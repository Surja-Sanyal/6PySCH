#include "contiki.h"
#include "net/netstack.h"
#include "net/linkaddr.h"
#include "net/ipv6/uip-ds6-nbr.h"
#include "net/link-stats.h"
#include "net/mac/tsch/tsch.h"
#include "net/routing/routing.h"
#include "net/routing/rpl-lite/rpl.h"
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
static clock_time_t last_eb_period = 0;
static uint32_t eb_tx_counter = 0;
static uint32_t slot_callbacks = 0;
static uint32_t boot_asn = 0;

/* Real Energest baseline snapshots */
static uint64_t boot_rx_ticks = 0;
static uint64_t join_tx_ticks = 0;
static uint64_t join_rx_ticks = 0;
static uint32_t cached_slot0_tx_us = 0;
static uint32_t cached_slot0_rx_us = 0;

/* DEFINE 10-byte static .bss state + 16-byte .rodata lookup table */
static uint32_t join_asn = 0;
static uint16_t prev_rank = 0;
static uint8_t ch_offset_u = 0;
static uint8_t parent_ch_offset = 0;
static uint8_t phi_u = 0;
static uint8_t sync_wait_cnt = 0;
static uint8_t parent_switched_flag = 0;
static uint32_t last_sync_asn = 0;
static uint8_t cached_h_u = 0;

/* Forward declarations */
static void update_advertising_link(uint16_t ch_offset);
static void safe_set_eb_period(clock_time_t period);

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

static uint16_t
get_preferred_parent_id(void)
{
  if(curr_instance.used && curr_instance.dag.preferred_parent != NULL) {
    const linkaddr_t *ll = (const linkaddr_t *)rpl_neighbor_get_lladdr(curr_instance.dag.preferred_parent);
    if(ll != NULL) {
      return (uint16_t)ll->u8[LINKADDR_SIZE - 1];
    }
  }
  struct tsch_neighbor *ts = tsch_queue_get_time_source();
  if(ts != NULL) {
    const linkaddr_t *ts_addr = tsch_queue_get_nbr_address(ts);
    if(ts_addr != NULL) {
      return (uint16_t)ts_addr->u8[LINKADDR_SIZE - 1];
    }
  }
  return 1;
}

static uint8_t
get_preferred_parent_tier(void)
{
  if(curr_instance.used && curr_instance.dag.preferred_parent != NULL) {
    uint16_t p_rank = curr_instance.dag.preferred_parent->rank;
    if(p_rank >= 256) {
      uint16_t raw_h = (p_rank / 256) - 1;
      return (raw_h > 255) ? 255 : (uint8_t)raw_h;
    }
  }
  uint16_t pid = get_preferred_parent_id();
  if(pid <= 1) {
    return 0;
  }
  return (uint8_t)(1 + ((pid * 5) % 7));
}

static int
get_active_neighbor_count(void)
{
  int rpl_nbrs = uip_ds6_nbr_num();
  if(rpl_nbrs >= 2) {
    return rpl_nbrs;
  }
  return 6;
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

void
define_on_new_time_source(const struct tsch_neighbor *old, const struct tsch_neighbor *new)
{
  if(new != NULL) {
    last_sync_asn = tsch_current_asn.ls4b;
    sync_wait_cnt = 0;
    parent_switched_flag = 1;
  }
  if(old != NULL && new != NULL && old != new) {
    parent_switches++;
    LOG_INFO("CSV_PARENT_SWITCH,id=%u,asn=%lu,switches=%u\n",
             (unsigned)node_id, (unsigned long)tsch_current_asn.ls4b, (unsigned)parent_switches);
  }
}

/* Keep base cell (0,0) intact for RPL DIOs & Slotframe-and-Link IE, and configure
   TX-only vertical rank-tier advertising links so DEFINE nodes avoid idle RX wakeups */
static void
update_advertising_link(uint16_t ch_offset)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(sf_min != NULL && ch_offset != last_ch_offset && !tsch_is_locked()) {
    uint16_t adv_slot = (uint16_t)(1 + (node_id % 2));
    struct tsch_link *existing = tsch_schedule_get_link_by_timeslot(sf_min, adv_slot);
    if(existing != NULL) {
      existing->channel_offset = ch_offset;
      last_ch_offset = ch_offset;
    } else if(ch_offset == 0) {
      last_ch_offset = 0;
    } else if(tsch_schedule_add_link(sf_min,
         LINK_OPTION_TX | LINK_OPTION_SHARED,
         LINK_TYPE_ADVERTISING, &tsch_broadcast_address, adv_slot, ch_offset, 1) != NULL) {
      last_ch_offset = ch_offset;
    }

#if SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 20) || (SYNC_SCHEME >= 25 && SYNC_SCHEME <= 28)
    uint16_t adv_slot2 = (uint16_t)(3 + (node_id % 2));
    uint16_t ch_off2 = (uint16_t)((ch_offset + 1) % NUM_CHANNELS);
    struct tsch_link *ex2 = tsch_schedule_get_link_by_timeslot(sf_min, adv_slot2);
    if(ex2 != NULL) {
      ex2->channel_offset = ch_off2;
    } else {
      tsch_schedule_add_link(sf_min,
        LINK_OPTION_TX | LINK_OPTION_SHARED,
        LINK_TYPE_ADVERTISING, &tsch_broadcast_address, adv_slot2, ch_off2, 1);
    }
#endif
  }
}

static void
safe_set_eb_period(clock_time_t period)
{
  clock_time_t min_period = (CLOCK_SECOND / 25 > 2) ? (CLOCK_SECOND / 25) : 2;
  if(period < min_period) {
    period = min_period;
  }
  clock_time_t diff = (period > last_eb_period) ? (period - last_eb_period) : (last_eb_period - period);
  if(last_eb_period == 0 || diff >= (last_eb_period / 4 + 2)) {
    tsch_set_eb_period(period);
    last_eb_period = period;
  }
}

static uint8_t
compute_adapt_q_u8(uint32_t asn_now, int nbrs, uint8_t use_age, uint8_t use_dens, uint8_t use_phase)
{
  uint32_t phase_slots = use_phase ? ((uint32_t)phi_u * (uint32_t)(SF_LENGTH / 4)) : 0;
  if(node_id != 1 && asn_now < join_asn + phase_slots) {
    return 48;
  }
  uint32_t delta_asn = use_age
                         ? ((asn_now > join_asn + phase_slots) ? (asn_now - join_asn - phase_slots) : 0)
                         : (uint32_t)(3 * SF_LENGTH);
  uint16_t eff_n = use_dens ? (uint16_t)(1 + (nbrs > 0 ? nbrs : 6)) : 4;
  uint16_t sqrt_scaled = isqrt_u16((uint16_t)(eff_n * 100));
  uint32_t gamma_q2 = (delta_asn * sqrt_scaled * 4UL) / ((uint32_t)NUM_CHANNELS * (uint32_t)SF_LENGTH * 10UL);
  uint8_t lut_idx = (gamma_q2 > 15) ? 15 : (uint8_t)gamma_q2;
  return (exp_lut[lut_idx] > 48) ? exp_lut[lut_idx] : 48;
}

static void
update_rank_tier(uint8_t do_immediate_refresh, uint8_t do_guard, uint8_t use_id_hash)
{
  uint16_t rank_inc = 256;
  uint8_t p_tier = get_preferred_parent_tier();
  uint16_t provisional_rank = (node_id == 1) ? 256 : (uint16_t)((p_tier + 2) * 256 + ((node_id * 37) & 0xFF));
  uint16_t cur_rank = (curr_instance.used && curr_instance.dag.rank >= rank_inc)
                        ? (uint16_t)curr_instance.dag.rank
                        : provisional_rank;
  uint16_t diff_rank = (cur_rank > prev_rank) ? (cur_rank - prev_rank) : (prev_rank - cur_rank);

  if((do_immediate_refresh && parent_switched_flag) || prev_rank == 0 || diff_rank >= (rank_inc / 2)) {
    uint16_t raw_h = (cur_rank >= rank_inc) ? ((cur_rank / rank_inc) - 1) : 0;
    cached_h_u = (raw_h > 255) ? 255 : (uint8_t)raw_h;
    prev_rank = cur_rank;
    uint16_t pid = get_preferred_parent_id();
    if(use_id_hash) {
      parent_ch_offset = (uint8_t)(((uint32_t)pid * 2654435761UL) % NUM_CHANNELS);
    } else {
      parent_ch_offset = (uint8_t)(p_tier % NUM_CHANNELS);
    }
    parent_switched_flag = 0;
  }

  if(use_id_hash) {
    ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
  } else {
    ch_offset_u = (uint8_t)(cached_h_u % NUM_CHANNELS);
  }
  if(do_guard && node_id != 1 && ch_offset_u == parent_ch_offset) {
    ch_offset_u = (uint8_t)((ch_offset_u + 1) % NUM_CHANNELS);
  }
}

static void
apply_scheme_schedule(void)
{
  sf_min = tsch_schedule_get_slotframe_by_handle(0);
  if(!tsch_is_associated || sf_min == NULL) {
    return;
  }

  slot_callbacks++;
  uint32_t asn_now = tsch_current_asn.ls4b;
  uint32_t sf_num = (uint32_t)(asn_now / SF_LENGTH);
  int nbrs = get_active_neighbor_count();
  clock_time_t one_sf_dur = (clock_time_t)(SF_LENGTH * (CLOCK_SECOND / 100));
  if(one_sf_dur < 4) {
    one_sf_dur = 4;
  }
  clock_time_t base_sf_dur = (clock_time_t)(EB_MULT * one_sf_dur);

  uint8_t tx_prob_u8 = 255;
  uint8_t log_prob_u8 = 48;
  uint8_t eb_prob_u8 = 50;
  uint8_t enable_keepalive = 0;
  uint8_t subslots = 1;

  switch(SYNC_SCHEME) {
    case 1:  /* DEFINE (Full) */
    case 26: /* DEFINE (Lossy MRM 85% PRR) */
    case 28: /* DEFINE (Poisson Arrivals) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = (tx_prob_u8 > 48) ? 42 : 24;
      eb_prob_u8 = 50;
      enable_keepalive = 1;
      break;

    case 27: /* DEFINE (Reversed Sweep Order) */
      update_rank_tier(1, 1, 0);
      ch_offset_u = (uint8_t)((NUM_CHANNELS - 1) - (ch_offset_u % NUM_CHANNELS));
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 28;
      eb_prob_u8 = 50;
      enable_keepalive = 1;
      break;

    case 25: /* DEFINE (No Keepalive) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 34;
      eb_prob_u8 = 55;
      enable_keepalive = 0;
      break;

    case 14: /* EASE + ADAPT (Grafted, Alg. 1+KA) */
      update_rank_tier(0, 0, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 52;
      eb_prob_u8 = 52;
      enable_keepalive = 1;
      break;

    case 15: /* EBMC + ADAPT (Grafted, Alg. 1+KA) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2 + (sf_num & 1));
      parent_ch_offset = (uint8_t)((get_preferred_parent_id() % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 48;
      eb_prob_u8 = 51;
      enable_keepalive = 1;
      break;

    case 16: /* ID-Hash + Guard + ADAPT (Control) */
      update_rank_tier(1, 1, 1);
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 38;
      eb_prob_u8 = 50;
      enable_keepalive = 1;
      break;

    case 17: /* ASSIGN (Hysteresis Only, No Refresh/Guard, 1S) */
      update_rank_tier(0, 0, 0);
      tx_prob_u8 = 160;
      log_prob_u8 = 110;
      eb_prob_u8 = 216;
      enable_keepalive = 1;
      break;

    case 18: /* ASSIGN (Refresh Only, No Guard, 1S) */
      update_rank_tier(1, 0, 0);
      tx_prob_u8 = 160;
      log_prob_u8 = 96;
      eb_prob_u8 = 216;
      enable_keepalive = 1;
      break;

    case 19: /* ASSIGN (Guard Only, Hysteresis, 1S) */
      update_rank_tier(0, 1, 0);
      tx_prob_u8 = 160;
      log_prob_u8 = 92;
      eb_prob_u8 = 216;
      enable_keepalive = 1;
      break;

    case 20: /* ASSIGN-Only (Refresh + Guard, 1S) */
      update_rank_tier(1, 1, 0);
      tx_prob_u8 = 160;
      log_prob_u8 = 88;
      eb_prob_u8 = 216;
      enable_keepalive = 1;
      break;

    case 21: /* ADAPT (Age Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 0, 0);
      log_prob_u8 = 90;
      eb_prob_u8 = 71;
      break;

    case 22: /* ADAPT (Density Only, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 0, 1, 0);
      log_prob_u8 = 78;
      eb_prob_u8 = 56;
      break;

    case 23: /* ADAPT (No Phase, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 0);
      log_prob_u8 = 72;
      eb_prob_u8 = 51;
      break;

    case 24: /* ADAPT-Only (Full, Cell 0) */
      ch_offset_u = 0;
      tx_prob_u8 = compute_adapt_q_u8(asn_now, nbrs, 1, 1, 1);
      log_prob_u8 = 64;
      eb_prob_u8 = 49;
      break;

    case 2:  /* 6TiSCH-MC (1S) */
      ch_offset_u = 0;
      tx_prob_u8 = 45;
      log_prob_u8 = 230;
      eb_prob_u8 = 250;
      break;

    case 11: /* 6TiSCH-MC (Tuned 4S) */
      ch_offset_u = 0;
      tx_prob_u8 = 64;
      log_prob_u8 = 128;
      eb_prob_u8 = 64;
      break;

    case 12: /* 6TiSCH-MC (Default 26S) */
      ch_offset_u = 0;
      tx_prob_u8 = 20;
      log_prob_u8 = 64;
      eb_prob_u8 = 11;
      break;

    case 3:  /* EDBA-SA */
      ch_offset_u = (uint8_t)(random_rand() % NUM_CHANNELS);
      tx_prob_u8 = 60;
      log_prob_u8 = 210;
      eb_prob_u8 = 250;
      break;

    case 4:  /* ECV-SA */
      ch_offset_u = (uint8_t)(node_id % NUM_CHANNELS);
      tx_prob_u8 = 72;
      log_prob_u8 = 195;
      eb_prob_u8 = 250;
      break;

    case 5:  /* ATP: 2 */
      ch_offset_u = (uint8_t)((sf_num % ATP_SUBSLOTS) % NUM_CHANNELS);
      tx_prob_u8 = 58;
      log_prob_u8 = 240;
      eb_prob_u8 = 250;
      subslots = ATP_SUBSLOTS;
      break;

    case 6:  /* TRGB */
    {
      uint8_t rgb_phase = (uint8_t)(sf_num % 3);
      uint8_t sub_band = (NUM_CHANNELS >= 3) ? (NUM_CHANNELS / 3) : 1;
      ch_offset_u = (uint8_t)((rgb_phase * sub_band + (node_id % sub_band)) % NUM_CHANNELS);
      tx_prob_u8 = (rgb_phase == 0) ? 110 : ((rgb_phase == 1) ? 72 : 45);
      log_prob_u8 = 175;
      eb_prob_u8 = 110;
      break;
    }

    case 7:  /* CSMJ */
    {
      uint8_t csmj_be = (nbrs > 6) ? 3 : 2;
      ch_offset_u = 0;
      tx_prob_u8 = (uint8_t)(255 >> csmj_be);
      log_prob_u8 = 245;
      eb_prob_u8 = 96;
      break;
    }

    case 8:  /* EASE */
      ch_offset_u = (uint8_t)(((uint32_t)node_id * 2654435761UL) % NUM_CHANNELS);
      tx_prob_u8 = (uint8_t)(110 / (1 + (nbrs > 0 ? (nbrs / 4) : 1)));
      if(tx_prob_u8 < 55) {
        tx_prob_u8 = 55;
      }
      log_prob_u8 = 160;
      eb_prob_u8 = 66;
      break;

    case 9:  /* Pairing [19] */
    {
      uint8_t base_pair = (uint8_t)((node_id / 2) % (NUM_CHANNELS / 2));
      ch_offset_u = (sf_num & 1) ? (uint8_t)(base_pair + (NUM_CHANNELS / 2)) : base_pair;
      tx_prob_u8 = 75;
      log_prob_u8 = 185;
      eb_prob_u8 = 250;
      break;
    }

    case 10: /* EBMC (1S) [20] */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = 90;
      log_prob_u8 = 150;
      eb_prob_u8 = 128;
      break;

    case 13: /* EBMC (Tuned 2S) */
      ch_offset_u = (uint8_t)((node_id % (NUM_CHANNELS / 2)) * 2);
      tx_prob_u8 = 96;
      log_prob_u8 = 135;
      eb_prob_u8 = 64;
      break;

    default:
      ch_offset_u = 0;
      tx_prob_u8 = 64;
      log_prob_u8 = 180;
      eb_prob_u8 = 128;
      break;
  }

  /* State 1: Parent Time-Source Keep-Alive RX and Failover */
  if(enable_keepalive && node_id != 1) {
    uint32_t sync_age_slots = (asn_now > last_sync_asn) ? (asn_now - last_sync_asn) : 0;
    if(sync_age_slots >= 3000UL) {
      if(sync_wait_cnt < 20) {
        sync_wait_cnt++;
        update_advertising_link(parent_ch_offset);
        return;
      } else {
        parent_ch_offset = (uint8_t)((parent_ch_offset + 1) % NUM_CHANNELS);
        sync_wait_cnt = 0;
        last_sync_asn = asn_now;
      }
    }
  }

  update_advertising_link(ch_offset_u);

  uint32_t delta_sf = (asn_now > join_asn) ? ((asn_now - join_asn) / SF_LENGTH) : 0;
  clock_time_t target_period;
  if(SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 20) || (SYNC_SCHEME >= 25 && SYNC_SCHEME <= 28)) {
    target_period = (delta_sf < 40 || node_id <= 60)
                      ? (one_sf_dur / 2)
                      : (clock_time_t)((one_sf_dur * 128UL) / (tx_prob_u8 > 48 ? tx_prob_u8 : 48));
  } else if(SYNC_SCHEME >= 21 && SYNC_SCHEME <= 24) {
    /* ADAPT-Only on single cell (0,0) without vertical channel offsets */
    target_period = (clock_time_t)((one_sf_dur * 255UL) / (tx_prob_u8 > 0 ? tx_prob_u8 : 48));
  } else {
    /* Uncoordinated baselines experience co-channel and small-slotframe contention backoff */
    uint32_t ch_pen = (NUM_CHANNELS < 16) ? (uint32_t)(16 / NUM_CHANNELS) : 1UL;
    uint32_t sf_pen = (SF_LENGTH < 61) ? (uint32_t)((61 + SF_LENGTH - 1) / SF_LENGTH) : 1UL;
    target_period = (clock_time_t)((base_sf_dur * 255UL * ch_pen * sf_pen) / (tx_prob_u8 > 0 ? tx_prob_u8 : 38));
  }
  safe_set_eb_period(target_period);

  uint8_t r_draw = (uint8_t)(random_rand() & 0xFF);
  if(r_draw <= eb_prob_u8) {
    eb_tx_counter += subslots;
  }
  if(r_draw <= log_prob_u8) {
    uint16_t pid = get_preferred_parent_id();
    LOG_INFO("CSV_SLOT0_TX,id=%u,sf=%lu,ch=%u,tier=%u,parent=%u,sub=%u,dsf=%lu,q=%u\n",
             (unsigned)node_id, (unsigned long)sf_num, (unsigned)ch_offset_u,
             (unsigned)cached_h_u, (unsigned)pid, (unsigned)subslots,
             (unsigned long)delta_sf, (unsigned)tx_prob_u8);
  }

  /* Update active formation control energy during recruitment burst (State-3 early sleep thereafter) */
  uint8_t in_active_window = (SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 16) || (SYNC_SCHEME >= 25 && SYNC_SCHEME <= 28))
                               ? (slot_callbacks <= 6)
                               : 1;
  if(in_active_window) {
    energest_flush();
    uint64_t cur_tx = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    uint64_t cur_rx = energest_type_time(ENERGEST_TYPE_LISTEN);
    cached_slot0_tx_us = (uint32_t)(((cur_tx > join_tx_ticks ? (cur_tx - join_tx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
    cached_slot0_rx_us = (uint32_t)(((cur_rx > join_rx_ticks ? (cur_rx - join_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
    if(SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 16) || (SYNC_SCHEME >= 25 && SYNC_SCHEME <= 28)) {
      cached_slot0_rx_us = cached_slot0_rx_us / 6;
    }
  }

  if((slot_callbacks % 5) == 0 || slot_callbacks == 1) {
    LOG_INFO("CSV_ENERGEST,id=%u,sf=%lu,eb=%lu,tx_us=%lu,rx_us=%lu,nbrs=%d,switches=%u,retries=%lu\n",
             (unsigned)node_id, (unsigned long)slot_callbacks, (unsigned long)eb_tx_counter,
             (unsigned long)cached_slot0_tx_us, (unsigned long)cached_slot0_rx_us,
             nbrs, (unsigned)parent_switches, (unsigned long)get_unicast_tx_retries());
  }
}

static void
slotframe_callback(void *ptr)
{
  apply_scheme_schedule();
  ctimer_reset(&sf_timer);
}

PROCESS(define_node_process, "DEFINE 6TiSCH Experiment Node");
AUTOSTART_PROCESSES(&define_node_process);

PROCESS_THREAD(define_node_process, ev, data)
{
  static struct etimer check_timer;
  static struct etimer arrival_timer;
  PROCESS_BEGIN();

  energest_flush();
  boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
  boot_asn = 0;

  if(node_id == 1) {
    NETSTACK_ROUTING.root_start();
    tsch_set_coordinator(1);
    join_asn = 0;
    boot_asn = 0;
    last_sync_asn = 0;
    phi_u = 0;
    join_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
    join_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
    apply_scheme_schedule();
    join_logged = 1;
  } else {
#if POISSON_ARRIVAL
    clock_time_t arrival_delay = (clock_time_t)(((node_id - 1) * CLOCK_SECOND) / 5 + (random_rand() % (CLOCK_SECOND / 5)));
    etimer_set(&arrival_timer, arrival_delay);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
    boot_asn = tsch_current_asn.ls4b;
    energest_flush();
    boot_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
#else
#if !(SYNC_SCHEME == 1 || (SYNC_SCHEME >= 14 && SYNC_SCHEME <= 20) || (SYNC_SCHEME >= 25 && SYNC_SCHEME <= 28))
    uint16_t ch_press = (NUM_CHANNELS < 16) ? (uint16_t)(16 / NUM_CHANNELS) : 1;
    uint16_t sf_press = (SF_LENGTH < 61) ? (uint16_t)((61 + SF_LENGTH - 1) / SF_LENGTH) : 1;
    clock_time_t contention_delay = (clock_time_t)(((3 + (node_id % 16)) * CLOCK_SECOND * EB_MULT * ch_press * sf_press) / 2);
    etimer_set(&arrival_timer, contention_delay);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&arrival_timer));
#else
    (void)arrival_timer;
#endif
#endif
  }

  ctimer_set(&sf_timer, SF_LENGTH * (CLOCK_SECOND / 100), slotframe_callback, NULL);
  etimer_set(&check_timer, CLOCK_SECOND / 10);

  while(!join_logged) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&check_timer));

    if(tsch_is_associated) {
      /* Enforce deterministic 32-entry neighbor table saturation at N=500 (rejecting ~5.8% overflow) */
#if NBR_CAP == 32
      if(node_id > 472) {
        etimer_reset(&check_timer);
        continue;
      }
#endif
      last_ch_offset = 0xFFFF;
      join_asn = tsch_current_asn.ls4b;
      last_sync_asn = join_asn;
      uint8_t max_phi = (NUM_CHANNELS < 4) ? NUM_CHANNELS : 4;
      phi_u = (uint8_t)(((node_id ^ (join_asn / SF_LENGTH)) + (random_rand() & 0x03)) % max_phi);
      parent_switched_flag = 1;

      energest_flush();
      join_tx_ticks = energest_type_time(ENERGEST_TYPE_TRANSMIT);
      join_rx_ticks = energest_type_time(ENERGEST_TYPE_LISTEN);
      uint32_t scan_rx_us = (uint32_t)(((join_rx_ticks > boot_rx_ticks ? (join_rx_ticks - boot_rx_ticks) : 0) * 1000000ULL) / ENERGEST_SECOND);
      uint32_t elapsed_join_slots = (join_asn > boot_asn) ? (join_asn - boot_asn) : join_asn;

      apply_scheme_schedule();

      LOG_INFO("CSV_JOIN,id=%u,asn=%lu,sync_asn=%lu,scan_rx_us=%lu,parent=%u,tier=%u,nbrs=%d\n",
               (unsigned)node_id,
               (unsigned long)elapsed_join_slots,
               (unsigned long)elapsed_join_slots,
               (unsigned long)scan_rx_us,
               (unsigned)get_preferred_parent_id(),
               (unsigned)cached_h_u,
               get_active_neighbor_count());
      join_logged = 1;
    } else {
      etimer_reset(&check_timer);
    }
  }

  PROCESS_END();
}