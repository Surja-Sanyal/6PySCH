#include "contiki.h"
#include "net/netstack.h"
#include "net/linkaddr.h"
#include "net/ipv6/uip-ds6-nbr.h"
#include "net/link-stats.h"
#include "net/mac/tsch/tsch.h"
#include "net/mac/tsch/tsch-schedule.h"
#include "net/mac/tsch/tsch-slot-operation.h"
#include "net/routing/routing.h"
#include "net/routing/rpl-lite/rpl.h"
#include "net/routing/rpl-lite/rpl-neighbor.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "sys/log.h"
#include "lib/random.h"
#include "node.h"

#define LOG_MODULE "DEFINE-NODE"
#define LOG_LEVEL LOG_LEVEL_INFO

static struct ctimer sf_timer;
static struct tsch_slotframe *sf_min;
static uint8_t join_logged = 0;
static uint16_t parent_switches = 0;
static uint16_t last_ch_offset = 0xFFFF;
static uint32_t eb_tx_counter = 0;
static uint32_t slot_callbacks = 0;
static uint32_t sync_asn = 0;
static clock_time_t boot_clock = 0;
static clock_time_t sync_clock = 0;
static uint8_t node_failed = 0;

/* Real Energest counters separated into initial scan RX vs. associated-only slot-0 TX/RX */
static uint64_t boot_rx_ticks = 0;
static uint64_t prev_tx_ticks = 0;
static uint64_t prev_rx_ticks = 0;
static uint64_t slot0_tx_acc_ticks = 0;
static uint64_t slot0_rx_acc_ticks = 0;
static uint8_t was_associated = 0;

/* DEFINE static .bss state + 16-byte .rodata lookup table */
static uint32_t join_asn = 0;
static uint16_t prev_rank = 0;
static uint8_t ch_offset_u = 0;
static uint8_t parent_ch_offset = 0;
static uint8_t phi_u = 0;
static uint8_t sync_wait_cnt = 0;
static uint8_t sync_cooldown = 0;
static uint8_t parent_switched_flag = 0;
static uint8_t cached_h_u = 0;

/* Forward declaration */
static void set_slot0_link(uint16_t ch_offset, uint8_t link_opts, enum link_type ltype);

/* 16-entry uint8 lookup table for exp(-gamma) scaled by 255 (P_min = 0.15 -> 38/255) */
static const uint8_t exp_lut[16] = {
  255, 199, 155, 120, 94, 73, 57, 48, 45, 42, 40, 38, 38, 38, 38, 38
};

static uint16_t
isqrt_u16(uint16_t val)
{
  uint16_t res = 0;
  uint16_t bit = 1U << 14;
  while(bit > val) {
    bit >>= 2;
  }
  while(bit != 0) {
    if(val >= res + bit) {
      val -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

/* Contiki-NG Cooja stores simMoteID big-endian in the last 2 bytes of linkaddr_t */
static uint16_t
get_preferred_parent_id(void)
{
  if(curr_instance.used && curr_instance.dag.preferred_parent != NULL) {
    const linkaddr_t *ll = (const linkaddr_t *)rpl_neighbor_get_lladdr(curr_instance.dag.preferred_parent);
    if(ll != NULL) {
      return (uint16_t)(((uint16_t)ll->u8[LINKADDR_SIZE - 2] << 8) | (uint16_t)ll->u8[LINKADDR_SIZE - 1]);
    }
  }
  struct tsch_neighbor *ts = tsch_queue_get_time_source();
  if(ts != NULL) {
    const linkaddr_t *ts_addr = tsch_queue_get_nbr_address(ts);
    if(ts_addr != NULL) {
      return (uint16_t)(((uint16_t)ts_addr->u8[LINKADDR_SIZE - 2] << 8) | (uint16_t)ts_addr->u8[LINKADDR_SIZE - 1]);
    }
  }
  return 0;
}

static uint8_t
get_preferred_parent_tier(void)
{
  if(curr_instance.used && curr_instance.dag.preferred_parent != NULL) {
    uint16_t p_rank = curr_instance.dag.preferred_parent->rank;
    if(p_rank >= RPL_CONF_MIN_HOPRANKINC) {
      uint16_t raw_h = (p_rank / RPL_CONF_MIN_HOPRANKINC) - 1;
      return (raw_h > 255) ? 255 : (uint8_t)raw_h;
    }
  }
  return 0;
}

static int
get_active_neighbor_count(void)
{
  int rpl_n = rpl_neighbor_count();
  int ds6_n = (int)uip_ds6_nbr_num();
  return (rpl_n > ds6_n) ? rpl_n : ds6_n;
}

static uint32_t
get_unicast_tx_retries(void)
{
  uint32_t retries = 0;
  uip_ds6_nbr_t *nbr = uip_ds6_nbr_head();
  while(nbr != NULL) {
    const linkaddr_t *ll = (const linkaddr_t *)uip_ds6_nbr_get_ll(nbr);
    if(ll != NULL) {
      const struct link_stats *stats = link_stats_from_lladdr(ll);
      if(stats != NULL && stats->etx > 128) {
        retries += (uint32_t)((stats->etx - 128) / 64);
      }
    }
    nbr = uip_ds6_nbr_next(nbr);
  }
  return retries;
}

/* Tier-Parity Sub-Band ASSIGN mapper:
   - Root (tier == 0): base offset 0
   - Odd rank tiers (tier % 2 == 1): odd offsets {1, 3, ..., C-1}
   - Even non-root tiers (tier >= 2, even): non-zero even offsets {2, 4, ..., C-2}
   Guarantees disjoint sub-bands across adjacent tiers, 100% channel utilization
   even in 2-hop topologies, and zero non-root EBs on base cell (0,0) for C >= 4. */
static uint8_t
compute_subband_offset(uint16_t id, uint8_t tier)
{
  if(tier == 0) {
    return 0;
  }
#if NUM_CHANNELS <= 2
  (void)id;
  return (uint8_t)(tier % NUM_CHANNELS);
#else
  uint8_t half = (uint8_t)(NUM_CHANNELS / 2);
  uint32_t mix = (((uint32_t)id * 2654435761UL) >> 16) ^ (uint32_t)id;
  if((tier & 1U) == 1U) {
    return (uint8_t)(1U + 2U * (mix % half));
  } else {
    return (uint8_t)(2U + 2U * (mix % (half - 1U)));
  }
#endif
}

/* Parent-alias guard shift: avoids base cell 0 for non-root DEFINE/ASSIGN nodes when C >= 4 */
static uint8_t
guard_shift_offset(uint8_t off, uint8_t allow_zero)
{
#if NUM_CHANNELS <= 2
  (void)allow_zero;
  return (uint8_t)((off + 1U) % NUM_CHANNELS);
#else
  if(allow_zero) {
    return (uint8_t)((off + 1U) % NUM_CHANNELS);
  }
  return (uint8_t)(1U + (off % (NUM_CHANNELS - 1U)));
#endif
}

void
define_on_new_time_source(const struct tsch_neighbor *old, const struct tsch_neighbor *new)
{
  if(new != NULL) {
    sync_wait_cnt = 0;
    sync_cooldown = 0;
    parent_switched_flag = 1;
  }
  if(old != NULL && new != NULL && old != new) {
    parent_switches++;
    LOG_INFO("CSV_PARENT_SWITCH,id=%u,asn=%lu,switches=%u\n",
             (unsigned)node_id, (unsigned long)tsch_current_asn.ls4b, (unsigned)parent_switches);
  }
}

/* Single-radio timeslot-0 link arbiter */
static void
set_slot0_link(uint16_t ch_offset, uint8_t link_opts, enum link_type ltype)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(sf_min != NULL && !tsch_is_locked()) {
    struct tsch_link *l = tsch_schedule_get_link_by_timeslot(sf_min, 0);
    if(l != NULL) {
      if(ch_offset != last_ch_offset || link_opts != l->link_options || ltype != l->link_type) {
        l->channel_offset = (uint8_t)ch_offset;
        l->link_options = link_opts;
        l->link_type = ltype;
        last_ch_offset = ch_offset;
      }
    } else {
      tsch_schedule_add_link(sf_min, link_opts, ltype,
                             &tsch_broadcast_address, 0, (uint8_t)ch_offset, 1);
      last_ch_offset = ch_offset;
    }
  }
}

static uint8_t
compute_adapt_q_u8(uint32_t asn_now, int nbrs, uint8_t use_age, uint8_t use_dens, uint8_t use_phase)
{
  uint32_t phase_slots = use_phase ? ((uint32_t)phi_u * (uint32_t)SF_LENGTH) : 0;
  if(node_id != 1 && asn_now < join_asn + phase_slots) {
    return 0;
  }
  uint32_t delta_asn = use_age
                         ? ((asn_now > join_asn + phase_slots) ? (asn_now - join_asn - phase_slots) : 0)
                         : (uint32_t)(3 * SF_LENGTH);
  uint16_t eff_n = use_dens ? (uint16_t)(1 + (nbrs > 0 ? nbrs : 0)) : 1;
  uint16_t sqrt_scaled = isqrt_u16((uint16_t)(eff_n * 100));
  uint32_t gamma_q2 = (delta_asn * sqrt_scaled * 4UL) / ((uint32_t)NUM_CHANNELS * (uint32_t)SF_LENGTH * 10UL);
  uint8_t lut_idx = (gamma_q2 > 15) ? 15 : (uint8_t)gamma_q2;
  return (exp_lut[lut_idx] > 38) ? exp_lut[lut_idx] : 38;
}

static void
update_rank_tier(uint8_t do_immediate_refresh, uint8_t do_guard, uint8_t use_id_hash)
{
  uint16_t rank_inc = RPL_CONF_MIN_HOPRANKINC;
  uint8_t p_tier = get_preferred_parent_tier();
  uint16_t cur_rank = (curr_instance.used && curr_instance.dag.rank >= rank_inc)
                        ? (uint16_t)curr_instance.dag.rank
                        : (uint16_t)((p_tier + 2) * rank_inc);
  uint16_t diff_rank = (cur_rank > prev_rank) ? (cur_rank - prev_rank) : (prev_rank - cur_rank);

  if((do_immediate_refresh && parent_switched_flag) || prev_rank == 0 || diff_rank >= (rank_inc / 2)) {
    uint16_t raw_h = (cur_rank >= rank_inc) ? ((cur_rank / rank_inc) - 1) : 0;
    cached_h_u = (raw_h > 255) ? 255 : (uint8_t)raw_h;
    if(node_id == 1) {
      cached_h_u = 0;
    } else if(cached_h_u == 0) {
      cached_h_u = 1;
    }
    prev_rank = cur_rank;
    parent_switched_flag = 0;
  }

  uint16_t pid = get_preferred_parent_id();
  if(use_id_hash) {
    parent_ch_offset = (uint8_t)(((uint32_t)pid * 2654435761UL) % NUM_CHANNELS);
    ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
  } else {
    uint8_t eff_p_tier = (pid == 1) ? 0 : ((p_tier == 0) ? 1 : p_tier);
    parent_ch_offset = compute_subband_offset(pid, eff_p_tier);
    ch_offset_u = compute_subband_offset(node_id, cached_h_u);
  }
  if(do_guard && node_id != 1 && ch_offset_u == parent_ch_offset) {
    ch_offset_u = guard_shift_offset(ch_offset_u, use_id_hash);
  }
}

static void
apply_scheme_schedule(void)
{
  if(node_failed) {
    return;
  }
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(!tsch_is_associated || sf_min == NULL) {
    return;
  }

  uint32_t asn_now = tsch_current_asn.ls4b;
  uint32_t sf_num = (uint32_t)(asn_now / SF_LENGTH);

  /* Algorithm 1: Between TSCH sync and RPL DODAG join, keep (0,0) open on 3 of 4 slotframes
     for fast DIO reception and probe candidate EB offsets every 4th slotframe during a bounded window */
  if(!NETSTACK_ROUTING.node_has_joined() && node_id != 1) {
    uint32_t max_win_sfs = (NUM_CHANNELS * 2 < 16) ? (uint32_t)(NUM_CHANNELS * 2) : 16UL;
    uint32_t w_eb = max_win_sfs * (uint32_t)SF_LENGTH;
    uint8_t in_sweep_win = (sync_asn > 0 && asn_now >= sync_asn && (asn_now - sync_asn) < w_eb);
    uint8_t uses_alg1 = (SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 28) || (SYNC_SCHEME >= 33 && SYNC_SCHEME <= 36));
    if(in_sweep_win && (sf_num & 3) == 3 && uses_alg1) {
      uint8_t step = (uint8_t)((sf_num >> 2) % NUM_CHANNELS);
      uint8_t cand_off = step;
      if(SYNC_SCHEME == 27 || SYNC_SCHEME == 33) {
        cand_off = (uint8_t)((NUM_CHANNELS - 1) - step);
      } else if(SYNC_SCHEME == 34) {
        cand_off = (uint8_t)(random_rand() % NUM_CHANNELS);
      }
      set_slot0_link(cand_off, LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING, LINK_TYPE_NORMAL);
    } else {
      set_slot0_link(0, LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING, LINK_TYPE_NORMAL);
    }
    return;
  }

  slot_callbacks++;
  int nbrs = get_active_neighbor_count();

  uint8_t tx_prob_u8 = 255;
  uint8_t enable_keepalive = 0;
  uint8_t subslots = 1;
  uint8_t ka_allow_zero = 0;

  switch(SYNC_SCHEME) {
    case 1:  /* DEFINE (Full) */
    case 26: /* DEFINE (Lossy UDGM 85% PRR) */
    case 27: /* DEFINE (Reversed Sweep Order - same ASSIGN map, reversed Alg. 1 sweep) */
    case 28: /* DEFINE (Poisson Arrivals) */
    case 34: /* DEFINE (Random Sweep Order) */
    case 35: /* DEFINE (Late Joiners into Steady Network) */
    case 36: /* DEFINE (Parent Failure & Rejoin) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      break;

    case 25: /* DEFINE (No Keepalive) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 0;
      break;

    case 14: /* EASE + ADAPT (Grafted, Alg. 1+KA) */
      update_rank_tier(0, 0, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      ka_allow_zero = 1;
      break;

    case 15: /* EBMC + ADAPT (Grafted, Alg. 1+KA) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2 + (sf_num & 1));
      parent_ch_offset = (uint8_t)((get_preferred_parent_id() % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      ka_allow_zero = 1;
      break;

    case 16: /* ID-Hash + Guard + ADAPT (Control) */
    case 33: /* ID-Hash + Guard + ADAPT (Reversed Sweep) */
      update_rank_tier(1, 1, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      enable_keepalive = 1;
      ka_allow_zero = 1;
      break;

    case 17: /* ASSIGN (Hysteresis Only, No Refresh/Guard, 1S) */
      update_rank_tier(0, 0, 0);
      tx_prob_u8 = (uint8_t)(216 / EB_MULT);
      enable_keepalive = 1;
      break;

    case 18: /* ASSIGN (Refresh Only, No Guard, 1S) */
      update_rank_tier(1, 0, 0);
      tx_prob_u8 = (uint8_t)(216 / EB_MULT);
      enable_keepalive = 1;
      break;

    case 19: /* ASSIGN (Guard Only, Hysteresis, 1S) */
      update_rank_tier(0, 1, 0);
      tx_prob_u8 = (uint8_t)(216 / EB_MULT);
      enable_keepalive = 1;
      break;

    case 20: /* ASSIGN-Only (Refresh + Guard, 1S) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = (uint8_t)(216 / EB_MULT);
      enable_keepalive = 1;
      break;

    case 21: /* ADAPT (Age Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 0, 0);
      break;

    case 22: /* ADAPT (Density Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 0, 1, 0);
      break;

    case 23: /* ADAPT (No Phase, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 0);
      break;

    case 24: /* ADAPT-Only (Full, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      break;

    case 2:  /* 6TiSCH-MC (1S) */
      ch_offset_u = 0;
      tx_prob_u8 = (uint8_t)(255 / EB_MULT);
      break;

    case 11: /* 6TiSCH-MC (Tuned 4S) */
      ch_offset_u = 0;
      tx_prob_u8 = 64;
      break;

    case 12: /* 6TiSCH-MC (Default 26S) */
      ch_offset_u = 0;
      tx_prob_u8 = 10;
      break;

    case 3:  /* EDBA-SA */
      ch_offset_u = (uint8_t)(random_rand() % NUM_CHANNELS);
      tx_prob_u8 = (uint8_t)(255 / EB_MULT);
      break;

    case 4:  /* ECV-SA */
      ch_offset_u = (uint8_t)(node_id % NUM_CHANNELS);
      tx_prob_u8 = (uint8_t)(255 / EB_MULT);
      break;

    case 5:  /* ATP: 2 */
      ch_offset_u = (uint8_t)((sf_num % ATP_SUBSLOTS) % NUM_CHANNELS);
      tx_prob_u8 = (uint8_t)(255 / EB_MULT);
      subslots = ATP_SUBSLOTS;
      break;

    case 6:  /* TRGB */
    {
      uint8_t rgb_phase = (uint8_t)(sf_num % 3);
      uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
      ch_offset_u = (uint8_t)((rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS);
      uint8_t base_p = (rgb_phase == 0) ? 204 : ((rgb_phase == 1) ? 102 : 26);
      tx_prob_u8 = (uint8_t)(base_p / EB_MULT);
      break;
    }

    case 7:  /* CSMJ */
    {
      uint8_t csmj_be = (nbrs > 8) ? 2 : 1;
      ch_offset_u = 0;
      tx_prob_u8 = (uint8_t)((255 >> csmj_be) / EB_MULT);
      break;
    }

    case 8:  /* EASE [27]: textbook P_eb = 1 / (1 + nbr_u) */
    case 30: /* EASE (Poisson Arrivals) */
    case 32: /* EASE (Lossy UDGM 85% PRR) */
    {
      uint16_t raw_p = (255U / (uint16_t)(1 + (nbrs > 0 ? nbrs : 0))) / (uint16_t)EB_MULT;
      ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
      tx_prob_u8 = (raw_p < 1U) ? 1U : (uint8_t)raw_p;
      break;
    }

    case 9:  /* Pairing [19] */
    {
      uint8_t base_pair = (uint8_t)((node_id / 2) % (NUM_CHANNELS / 2));
      ch_offset_u = (sf_num & 1) ? (uint8_t)(base_pair + (NUM_CHANNELS / 2)) : base_pair;
      tx_prob_u8 = (uint8_t)(255 / EB_MULT);
      break;
    }

    case 10: /* EBMC (1S) [20] */
    case 31: /* EBMC (1S, Lossy UDGM 85% PRR) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = (uint8_t)(128 / EB_MULT);
      break;

    case 13: /* EBMC (Tuned 2S) */
    case 29: /* EBMC (Tuned 2S, Poisson Arrivals) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = 64;
      break;

    default:
      ch_offset_u = 0;
      tx_prob_u8 = 255;
      break;
  }

  /* State 1: Bounded Parent Time-Source Keep-Alive RX (30 s drift trigger, 10-SF probe, 20-SF cooldown) */
  if(enable_keepalive && node_id != 1) {
    clock_time_t now = clock_time();
    clock_time_t sync_age_ticks = (now > tsch_last_sync_time) ? (now - tsch_last_sync_time) : 0;
    if(sync_age_ticks < (clock_time_t)(30 * CLOCK_SECOND)) {
      sync_wait_cnt = 0;
      sync_cooldown = 0;
    } else if(sync_cooldown > 0) {
      sync_cooldown--;
    } else if(sync_wait_cnt < 10) {
      uint16_t pid = get_preferred_parent_id();
      uint8_t rx_off = (sync_wait_cnt < 8 || pid == 1)
                         ? parent_ch_offset
                         : guard_shift_offset(parent_ch_offset, ka_allow_zero);
      sync_wait_cnt++;
      set_slot0_link(rx_off, LINK_OPTION_RX | LINK_OPTION_TIME_KEEPING, LINK_TYPE_NORMAL);
      return;
    } else {
      sync_wait_cnt = 0;
      sync_cooldown = 20;
    }
  }

  uint32_t delta_sf = (asn_now > join_asn) ? ((asn_now - join_asn) / SF_LENGTH) : 0;

  /* Reserve at least 15% of slotframes for State 3 on (0,0) when ch_offset_u != 0 so DIOs are never starved */
  uint8_t eb_gate = (ch_offset_u != 0 && tx_prob_u8 > 216) ? 216 : tx_prob_u8;
  uint8_t r_draw = (uint8_t)(random_rand() & 0xFF);
  if(eb_gate > 0 && r_draw < eb_gate) {
    enum link_type st2_type = (ch_offset_u == 0) ? LINK_TYPE_ADVERTISING : LINK_TYPE_ADVERTISING_ONLY;
    set_slot0_link(ch_offset_u, LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING, st2_type);
    eb_tx_counter += subslots;
    uint16_t pid = get_preferred_parent_id();
    LOG_INFO("CSV_SLOT0_TX,id=%u,sf=%lu,ch=%u,tier=%u,parent=%u,sub=%u,dsf=%lu,q=%u\n",
             (unsigned)node_id, (unsigned long)sf_num, (unsigned)ch_offset_u,
             (unsigned)cached_h_u, (unsigned)pid, (unsigned)subslots,
             (unsigned long)delta_sf, (unsigned)tx_prob_u8);
  } else {
    set_slot0_link(0, LINK_OPTION_TX | LINK_OPTION_RX | LINK_OPTION_SHARED | LINK_OPTION_TIME_KEEPING, LINK_TYPE_NORMAL);
  }

  if((slot_callbacks % 10) == 0 || slot_callbacks == 1) {
    uint32_t slot0_tx_us = (uint32_t)((slot0_tx_acc_ticks * 1000000ULL) / ENERGEST_SECOND);
#if SYNC_SCHEME == 5 && ATP_SUBSLOTS > 1
    /* Account for physical 250-kbps airtime (1600 us per 50-byte EB) of the (s-1) extra ATP subslot EBs */
    slot0_tx_us += (uint32_t)((eb_tx_counter / ATP_SUBSLOTS) * (uint32_t)(ATP_SUBSLOTS - 1) * 1600UL);
#endif
    uint32_t slot0_rx_us = (uint32_t)((slot0_rx_acc_ticks * 1000000ULL) / ENERGEST_SECOND);
    LOG_INFO("CSV_ENERGEST,id=%u,sf=%lu,eb=%lu,tx_us=%lu,rx_us=%lu,nbrs=%d,switches=%u,retries=%lu\n",
             (unsigned)node_id, (unsigned long)slot_callbacks, (unsigned long)eb_tx_counter,
             (unsigned long)slot0_tx_us, (unsigned long)slot0_rx_us,
             nbrs, (unsigned)parent_switches, (unsigned long)get_unicast_tx_retries());
  }
}

static void
slotframe_callback(void *ptr)
{
  /* Accumulate Energest TX/RX strictly during associated slotframes so post-desync
     continuous rescanning never contaminates associated shared-slot control energy */
  if(join_logged && !node_failed) {
    energest_flush();
    uint64_t cur_tx = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    uint64_t cur_rx = energest_type_time(ENERGEST_TYPE_LISTEN);
    uint64_t d_tx = (cur_tx >= prev_tx_ticks) ? (cur_tx - prev_tx_ticks) : 0;
    uint64_t d_rx = (cur_rx >= prev_rx_ticks) ? (cur_rx - prev_rx_ticks) : 0;
    prev_tx_ticks = cur_tx;
    prev_rx_ticks = cur_rx;
    if(tsch_is_associated && was_associated) {
      slot0_tx_acc_ticks += d_tx;
      slot0_rx_acc_ticks += d_rx;
    }
    was_associated = tsch_is_associated ? 1 : 0;
  }

  apply_scheme_schedule();
  if(!node_failed) {
    ctimer_reset(&sf_timer);
  }
}

PROCESS(define_node_process, "DEFINE 6TiSCH Experiment Node");
AUTOSTART_PROCESSES(&define_node_process);

PROCESS_THREAD(define_node_process, ev, data)
{
  static struct etimer check_timer;
  static struct etimer arrival_timer;
  static struct etimer fail_timer;
  static uint16_t poll_ticks = 0;
  PROCESS_BEGIN();

  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
#if POISSON_ARRIVAL || LATE_JOINER
    NETSTACK_MAC.on();
#endif
    join_asn = 0;
    sync_asn = 0;
    boot_clock = clock_time();
    sync_clock = boot_clock;
    phi_u = 0;
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
    prev_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    prev_rx_ticks = boot_rx_ticks;
    slot0_tx_acc_ticks = 0;
    slot0_rx_acc_ticks = 0;
    was_associated = 1;
    join_logged = 1;
    apply_scheme_schedule();
  } else {
#if POISSON_ARRIVAL
    /* Homogeneous Poisson process (lambda = 5/s) conditioned on N_NODES arrivals:
       unordered arrival times are i.i.d. Uniform[0, N_NODES / 5] seconds */
    uint64_t window_ticks = ((uint64_t)N_NODES * (uint64_t)CLOCK_SECOND) / 5ULL;
    clock_time_t arrival_delay = (clock_time_t)(((uint64_t)random_rand() * window_ticks) >> 16);
    if(arrival_delay > 0) {
      etimer_set(&arrival_timer, arrival_delay);
      PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
    }
    boot_clock = clock_time();
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
    NETSTACK_MAC.on();
#elif LATE_JOINER
    /* 20% of pledges boot at t = 150 s into an already-formed steady network (P_min = 0.15 regime) */
    if((node_id % 5) == 0) {
      etimer_set(&arrival_timer, (clock_time_t)(150 * CLOCK_SECOND));
      PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
    }
    boot_clock = clock_time();
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
    NETSTACK_MAC.on();
#else
    (void)arrival_timer;
    boot_clock = clock_time();
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
#endif
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(!join_logged) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));
    poll_ticks++;

    if(tsch_is_associated && sync_asn == 0) {
      sync_asn = tsch_current_asn.ls4b;
      if(sync_asn == 0) {
        sync_asn = 1;
      }
      sync_clock = clock_time();
    }

    if(tsch_is_associated && NETSTACK_ROUTING.node_has_joined()) {
      clock_time_t join_clock = clock_time();
      last_ch_offset = 0xFFFF;
      join_asn = tsch_current_asn.ls4b;
      uint8_t max_phi = (NUM_CHANNELS < 4) ? NUM_CHANNELS : 4;
      phi_u = (uint8_t)(((node_id ^ (join_asn / SF_LENGTH)) + (random_rand() & 0x03)) % max_phi);
      parent_switched_flag = 1;

      energest_flush();
      prev_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
      prev_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
      slot0_tx_acc_ticks = 0;
      slot0_rx_acc_ticks = 0;
      was_associated = 1;

      uint32_t scan_rx_us = (uint32_t)(((prev_rx_ticks > boot_rx_ticks ? (prev_rx_ticks - boot_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
      uint32_t elapsed_sync_slots = (sync_clock >= boot_clock)
        ? (uint32_t)(((uint64_t)(sync_clock - boot_clock) * 100ULL) / CLOCK_SECOND)
        : 0;
      uint32_t elapsed_join_slots = (join_clock >= boot_clock)
        ? (uint32_t)(((uint64_t)(join_clock - boot_clock) * 100ULL) / CLOCK_SECOND)
        : 0;

      join_logged = 1;
      apply_scheme_schedule();

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,sync_asn=%lu,scan_rx_us=%lu,parent=%u,tier=%u,nbrs=%d\n",
               (unsigned)node_id,
               (unsigned long)elapsed_join_slots,
               (unsigned long)elapsed_sync_slots,
               (unsigned long)scan_rx_us,
               (unsigned)get_preferred_parent_id(),
               (unsigned)cached_h_u,
               get_active_neighbor_count());
    } else {
      if((poll_ticks % 100) == 0) {
        energest_flush();
        uint64_t cur_rx = energest_type_time(ENERGEST_TYPE_LISTEN);
        uint32_t cur_scan_us = (uint32_t)(((cur_rx > boot_rx_ticks ? (cur_rx - boot_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
        LOG_INFO("CSV_SCAN_ENERGEST,id=%u,scan_rx_us=%lu\n",
                 (unsigned)node_id, (unsigned long)cur_scan_us);
      }
      etimer_reset(&check_timer);
    }
  }

#if PARENT_FAILURE
  /* Controlled relay failure: 10% of non-root relays shut down at t = 120 s to test failover/rejoin */
  if(node_id != 1 && (node_id % 10) == 2) {
    clock_time_t now = clock_time();
    clock_time_t target = (clock_time_t)(120 * CLOCK_SECOND);
    if(target > now) {
      etimer_set(&fail_timer, target - now);
      PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&fail_timer));
    }
    node_failed = 1;
    ctimer_stop(&sf_timer);
    tsch_set_eb_period(0);
    tsch_schedule_remove_all_slotframes();
    NETSTACK_MAC.off();
    LOG_INFO("CSV_NODE_FAILED,id=%u,asn=%lu\n",
             (unsigned)node_id, (unsigned long)tsch_current_asn.ls4b);
  }
#else
  (void)fail_timer;
#endif

  PROCESS_END();
}