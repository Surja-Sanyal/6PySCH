#!/usr/bin/env python3
# Program:   6PySCH — Discrete-Event 6TiSCH Network Synchronization Simulator
#            Aligned 1:1 with Contiki-NG Cooja firmware (DEFINE + all 49 configurations)
# Outputs:   Statistics/_{prefix}_cooja_stats_.txt, Statistics/_{prefix}_stats_.txt,
#            Statistics/join_times_N{100,500}_scheme{id}.csv (28-column format)
# Author:    Surja Sanyal
# Email:     hi.surja06@gmail.com
# python3 6PySCH.py --mode benchmark --seeds 5
# python3 6PySCH.py --mode full --seeds 30
# python3 scripts/plot_results.py


import os
import sys
import glob
import math
import time
import random
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

## Global defaults (matching project-conf.h & run_cooja_headless.py) ##
COMM_RANGE = 50.0
DEFAULT_SEEDS = 5
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_NBR_CAP = 16
DEFAULT_ATP_SUBSLOTS = 2

# RFC 6206 Trickle parameters matching project-conf.h (I_min=2^11 ms, DOUBLINGS=3 -> I_max=16.384 s, k=3)
TRICKLE_I_MIN_S = 2.048
TRICKLE_I_MAX_S = 16.384
TRICKLE_K = 3

# Stock Contiki-NG RPL-Lite DIS timing (used when ENABLE_RPL_DIS = 1 on schemes 46..49)
STOCK_DIS_START_S = 5.0
STOCK_DIS_INTERVAL_S = 60.0

# 16-entry uint8 LUT for exp(-gamma) scaled by 255 (P_min = 0.15 -> 38/255)
EXP_LUT = (255, 199, 155, 120, 94, 73, 57, 48, 45, 42, 40, 38, 38, 38, 38, 38)

# Physical IEEE 802.15.4 (250 kbps) & Contiki-NG TSCH airtime constants (us)
EB_TX_US = 1600
DIO_TX_US = 2200
DIS_TX_US = 1200
IDLE_RX_US = 350     # Contiki-NG early-sleep CCA listen on idle slot 0
FRAME_RX_US = 1900

# Only these three schemes run a 64-entry table check (matching run_cooja_headless.py & plot_results.py)
VERIFIED_64NBR_SCHEMES = {1, 8, 13}

# Complete 49-scheme map matching experiments/define-sync/node.c & run_cooja_headless.py
SCHEME_MAP = {
    1:  "p",                  # DEFINE (Full)
    2:  "e",                  # 6TiSCH-MC (1S)
    3:  "r",                  # EDBA-SA
    4:  "d",                  # ECV-SA
    5:  "a_2",                # ATP: 2
    6:  "t",                  # TRGB (1S)
    7:  "c",                  # CSMJ
    8:  "s",                  # EASE
    9:  "pair",               # Pairing (1S)
    10: "ebmc",               # EBMC (1S)
    11: "mc_4s",              # 6TiSCH-MC (Tuned 4S)
    12: "mc_26s",             # 6TiSCH-MC (Default 26S)
    13: "ebmc_2s",            # EBMC (Tuned 2S)
    14: "ease_adapt",         # EASE + ADAPT (Grafted)
    15: "ebmc_adapt",         # EBMC + ADAPT (Grafted)
    16: "idhash_guard_adapt", # ID-Hash + Guard + ADAPT (Control)
    17: "assign_hyst",        # ASSIGN (Hysteresis Only)
    18: "assign_ref",         # ASSIGN (Refresh Only)
    19: "assign_guard",       # ASSIGN (Guard Only)
    20: "assign_both",        # ASSIGN-Only (Refresh + Guard)
    21: "adapt_age",          # ADAPT (Age Only)
    22: "adapt_dens",         # ADAPT (Density Only)
    23: "adapt_nophase",      # ADAPT (No Phase)
    24: "adapt_full",         # ADAPT-Only (Full)
    25: "define_noka",        # DEFINE (No Keepalive)
    26: "define_mrm",         # DEFINE (Lossy UDGM 85% PRR)
    27: "define_rev",         # DEFINE (Reversed Sweep Order)
    28: "define_poisson",     # DEFINE (Poisson Arrivals, lambda=5/s)
    29: "ebmc_2s_poisson",    # EBMC (Tuned 2S, Poisson Arrivals)
    30: "ease_poisson",       # EASE (Poisson Arrivals)
    31: "ebmc_mrm",           # EBMC (1S, Lossy UDGM 85% PRR)
    32: "ease_mrm",           # EASE (Lossy UDGM 85% PRR)
    33: "idhash_rev",         # ID-Hash + Guard + ADAPT (Reversed Sweep)
    34: "define_rand",        # DEFINE (Random Sweep Order)
    35: "define_late",        # DEFINE (Late Joiners, 20% at t=150s)
    36: "define_fail",        # DEFINE (10% Relay Shutdown at t=120s)
    37: "define_mixed",       # DEFINE (Mixed 80% DEFINE / 20% Stock RFC 8180)
    38: "define_tierstart",   # DEFINE (Tier-Aware Sweep Start)
    39: "pair_2s",            # Pairing (Tuned 2S)
    40: "trgb_2s",            # TRGB (Tuned 2S)
    41: "define_hoptier_mrm", # DEFINE (Hop-Count Tier, Lossy UDGM 85% PRR)
    42: "idhash_nonzero",     # ID-Hash Non-Zero {1..C-1} + Guard + ADAPT (Control)
    43: "idparity_assign",    # Node-ID Parity (u mod 2) + In-Band Hash + Guard + ADAPT
    44: "pair_adapt",         # Pairing + ADAPT (Grafted, Alg. 1+KA)
    45: "define_nosweep",     # DEFINE (No Candidate Sweep — Always on (0,0) after sync)
    46: "define_dis",         # DEFINE (Stock RPL DIS Enabled)
    47: "pair_dis",           # Pairing (1S, Stock RPL DIS Enabled)
    48: "ease_dis",           # EASE (Stock RPL DIS Enabled)
    49: "ebmc_2s_dis",        # EBMC (Tuned 2S, Stock RPL DIS Enabled)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=55):
    """Constant spatial density 1.65e-2 motes/m^2 matching Fig. 8 & run_cooja_headless.py."""
    side = int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes))))
    if side == 100:
        side = 99
    return min(175, max(50, side))


def load_connected_coords(n_nodes, area=100, seed=1, tx_range=COMM_RANGE):
    """Seeded connected-random growth topology matching run_cooja_headless.py."""
    rng = random.Random(seed)
    cx, cy = area / 2.0, area / 2.0
    coords = [(int(round(cx)), int(round(cy)))]
    max_hop_dist = tx_range * 0.72  # 36 m within 50 m UDGM range

    for _ in range(n_nodes):
        placed = False
        for _attempt in range(500):
            x = int(round(rng.uniform(0.0, float(area))))
            y = int(round(rng.uniform(0.0, float(area))))
            for px, py in coords:
                if math.hypot(x - px, y - py) <= max_hop_dist:
                    coords.append((x, y))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            px, py = rng.choice(coords)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            r = rng.uniform(5.0, max_hop_dist)
            coords.append((
                min(area, max(0, int(round(px + r * math.cos(angle))))),
                min(area, max(0, int(round(py + r * math.sin(angle))))),
            ))
    return coords


def build_interference_adj(coords, tx_range=COMM_RANGE):
    """Build symmetric 1-hop RF adjacency graph (1-indexed node IDs)."""
    n_total = len(coords)
    adj = {i + 1: set() for i in range(n_total)}
    r2 = tx_range * tx_range
    for i in range(n_total):
        xi, yi = coords[i]
        for j in range(i + 1, n_total):
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            if dx * dx + dy * dy <= r2:
                adj[i + 1].add(j + 1)
                adj[j + 1].add(i + 1)
    return adj


def compute_subband_offset(nid, tier, n_ch):
    """
    Tier-Parity Sub-Band ASSIGN mapper (matches node.c):
      - Root (tier == 0): base offset 0
      - Odd rank tiers:   odd offsets {1, 3, ..., C-1}
      - Even non-root:    non-zero even offsets {2, 4, ..., C-2}
    """
    if tier == 0:
        return 0
    if n_ch <= 2:
        return tier % n_ch
    half = n_ch // 2
    mix = (((nid * 2654435761) & 0xFFFFFFFF) >> 16) ^ nid
    if (tier & 1) == 1:
        return 1 + 2 * (mix % half)
    return 2 + 2 * (mix % max(1, half - 1))


def compute_nonzero_idhash_offset(nid, n_ch):
    """Non-zero uniform ID-hash mapper over {1, ..., C-1} (control for Scheme 42, matches node.c)."""
    if nid <= 1:
        return 0
    if n_ch <= 2:
        return nid % n_ch
    mix = (((nid * 2654435761) & 0xFFFFFFFF) >> 16) ^ nid
    return 1 + (mix % (n_ch - 1))


def guard_shift_offset(off, allow_zero, n_ch):
    """Local parent-alias guard shift (avoids offset 0 for non-root DEFINE/ASSIGN when C >= 4)."""
    if n_ch <= 2 or allow_zero:
        return (off + 1) % n_ch
    return 1 + (off % (n_ch - 1))


def compute_adapt_q_u8(nid, asn_now, join_asn, phi_u, nbrs, n_ch, sf_len, use_age, use_dens, use_phase):
    """Integer LUT ADAPT Bernoulli attempt probability in [0..255] (matches node.c)."""
    phase_slots = (phi_u * sf_len) if use_phase else 0
    if nid != 1 and asn_now < join_asn + phase_slots:
        return 0
    if use_age:
        delta_asn = (asn_now - join_asn - phase_slots) if (asn_now > join_asn + phase_slots) else 0
    else:
        delta_asn = 3 * sf_len
    eff_n = (1 + max(0, nbrs)) if use_dens else 1
    sqrt_scaled = math.isqrt(eff_n * 100)
    gamma_q2 = (delta_asn * sqrt_scaled * 4) // (n_ch * sf_len * 10)
    lut_idx = 15 if gamma_q2 > 15 else gamma_q2
    val = EXP_LUT[lut_idx]
    return val if val > 38 else 38


def simulate_seed(
    scheme_id, n_nodes=100, area=100, n_ch=16, sf_len=61, eb_mult=1,
    seed=1, nbr_cap=DEFAULT_NBR_CAP, timeout_s=DEFAULT_TIMEOUT_S, atp_subslots=DEFAULT_ATP_SUBSLOTS
):
    """
    Run one complete paired-seed discrete-event TSCH/RPL bootstrapping simulation
    matching Contiki-NG Cooja (node.c + project-conf.h + run_cooja_headless.py).
    Returns (28-column stats row list, list of per-node join times in seconds) or (None, []).
    """
    if area is None:
        area = compute_constant_density_area(n_nodes)

    rng = random.Random(seed * 10007 + scheme_id * 97 + nbr_cap * 13 + sf_len * 7 + n_ch * 3 + eb_mult)
    coords = load_connected_coords(n_nodes, area=area, seed=seed, tx_range=COMM_RANGE)
    adj = build_interference_adj(coords, tx_range=COMM_RANGE)

    poisson_flag = scheme_id in (28, 29, 30)
    late_flag = (scheme_id == 35)
    fail_flag = (scheme_id == 36)
    dis_flag = scheme_id in (46, 47, 48, 49)
    rx_success = 0.85 if scheme_id in (26, 31, 32, 41) else 1.0
    base_uses_alg1 = (
        scheme_id == 1 or (14 <= scheme_id <= 28) or (33 <= scheme_id <= 38)
        or (41 <= scheme_id <= 44) or scheme_id == 46
    )

    n_tot = n_nodes + 1
    sf_dur_s = sf_len * 0.01
    max_sfs = int(round(timeout_s / sf_dur_s))
    dis_start_sfs = max(1, int(round(STOCK_DIS_START_S / sf_dur_s)))
    dis_interval_sfs = max(1, int(round(STOCK_DIS_INTERVAL_S / sf_dur_s)))

    # Boot schedule (simultaneous cold-start, conditional Poisson lambda=5/s, or 20% late joiners)
    boot_sf = [0] * (n_tot + 1)
    if poisson_flag:
        win_s = float(n_nodes) / 5.0
        for u in range(2, n_tot + 1):
            boot_sf[u] = int((rng.random() * win_s) / sf_dur_s)
    elif late_flag:
        late_sf = int(150.0 / sf_dur_s)
        for u in range(2, n_tot + 1):
            if (u % 5) == 0:
                boot_sf[u] = late_sf

    fail_sf = int(120.0 / sf_dur_s)

    # Node MAC / RPL state (1-indexed: node 1 is PAN coordinator/root, 2..n_nodes+1 are pledges)
    associated = [False] * (n_tot + 1)
    rpl_joined = [False] * (n_tot + 1)
    ever_joined = [False] * (n_tot + 1)
    failed = [False] * (n_tot + 1)

    sync_asn = [0] * (n_tot + 1)
    join_asn = [0] * (n_tot + 1)
    join_sf_map = [0] * (n_tot + 1)
    elapsed_sync_slots = [0] * (n_tot + 1)
    elapsed_join_slots = [0] * (n_tot + 1)
    next_dis_sf = [max_sfs + 1] * (n_tot + 1)

    parent = [0] * (n_tot + 1)
    rank = [65535] * (n_tot + 1)
    prev_rank = [0] * (n_tot + 1)
    hop_count = [255] * (n_tot + 1)
    cached_tier = [0] * (n_tot + 1)
    ch_offset = [0] * (n_tot + 1)
    parent_ch_offset = [0] * (n_tot + 1)
    phi = [0] * (n_tot + 1)
    last_sync_sf = [0] * (n_tot + 1)
    sync_wait_cnt = [0] * (n_tot + 1)
    sync_cooldown = [0] * (n_tot + 1)
    parent_switched_flag = [False] * (n_tot + 1)
    switches = [0] * (n_tot + 1)
    retries = [0] * (n_tot + 1)

    nbr_table = [set() for _ in range(n_tot + 1)]
    cand_rank = [{} for _ in range(n_tot + 1)]
    node_ch_off_log = {1: 0}
    node_first_tx_sf = {1: 0}

    # Exact RFC 6206 Trickle DIO timer (I_min = 2.048 s, I_max = 16.384 s, k = 3)
    trickle_I_s = [TRICKLE_I_MIN_S] * (n_tot + 1)
    trickle_end_sf = [0] * (n_tot + 1)
    trickle_tx_sf = [0] * (n_tot + 1)
    trickle_c = [0] * (n_tot + 1)
    trickle_done = [False] * (n_tot + 1)

    # Separated Energest counters (us): scan RX vs. associated-only slot-0 TX/RX
    scan_rx_us = [0] * (n_tot + 1)
    slot0_tx_us = [0] * (n_tot + 1)
    slot0_rx_us = [0] * (n_tot + 1)
    eb_cnt = [0] * (n_tot + 1)
    active_sfs = [0] * (n_tot + 1)

    desync_sfs = int(round(180.0 / sf_dur_s))
    ka_trigger_sfs = int(round(30.0 / sf_dur_s))
    max_win_sfs = min(n_ch * 2, 16)
    max_phi = min(n_ch, 4)

    def try_add_nbr(u, v):
        tbl = nbr_table[u]
        if v in tbl:
            return True
        if len(tbl) < nbr_cap:
            tbl.add(v)
            return True
        return False

    def reset_trickle(u, sf_now):
        trickle_I_s[u] = TRICKLE_I_MIN_S
        dur = max(1, int(round(TRICKLE_I_MIN_S / sf_dur_s)))
        trickle_end_sf[u] = sf_now + dur
        trickle_tx_sf[u] = sf_now + rng.randint(max(1, dur // 2), dur)
        trickle_c[u] = 0
        trickle_done[u] = False

    def update_rank_tier_node(u, do_imm, do_guard, use_id_hash, use_hop_only=False):
        p = parent[u]
        p_tier = cached_tier[p] if p > 0 else 0
        if use_hop_only and u != 1:
            cur_r = (hop_count[u] + 1) * 256 if hop_count[u] < 255 else (p_tier + 2) * 256
        else:
            cur_r = rank[u] if 256 <= rank[u] < 65535 else (p_tier + 2) * 256
        diff_r = abs(cur_r - prev_rank[u])
        should_upd = (
            (parent_switched_flag[u] or prev_rank[u] == 0)
            if use_hop_only
            else ((do_imm and parent_switched_flag[u]) or prev_rank[u] == 0 or diff_r >= 128)
        )
        if should_upd:
            raw_h = (cur_r // 256) - 1 if cur_r >= 256 else 0
            h = min(255, max(0, raw_h))
            if u == 1:
                h = 0
            elif h == 0:
                h = 1
            cached_tier[u] = h
            prev_rank[u] = cur_r
            parent_switched_flag[u] = False
        if use_id_hash:
            p_off = ((p * 2654435761) & 0xFFFFFFFF) % n_ch
            u_off = ((u * 2654435761) & 0xFFFFFFFF) % n_ch
        else:
            eff_p_tier = 0 if p == 1 else (1 if p_tier == 0 else p_tier)
            p_off = compute_subband_offset(p, eff_p_tier, n_ch)
            u_off = compute_subband_offset(u, cached_tier[u], n_ch)
        parent_ch_offset[u] = p_off
        if do_guard and u != 1 and u_off == p_off:
            u_off = guard_shift_offset(u_off, use_id_hash, n_ch)
        ch_offset[u] = u_off

    # Initialize PAN coordinator (node 1)
    associated[1] = True
    rpl_joined[1] = True
    ever_joined[1] = True
    rank[1] = 256
    prev_rank[1] = 256
    hop_count[1] = 0
    cached_tier[1] = 0
    ch_offset[1] = 0
    reset_trickle(1, 0)

    coll_tree = 0
    coll_sib = 0
    coll_cross = 0
    joined_count = 0
    last_join_sf = 0

    for sf_num in range(max_sfs):
        asn_now = sf_num * sf_len

        # Scheme 36: Controlled 10% non-root relay shutdown at t = 120 s
        if fail_flag and sf_num == fail_sf:
            for u in range(2, n_tot + 1):
                if (u % 10) == 2 and not failed[u]:
                    failed[u] = True
                    associated[u] = False
                    rpl_joined[u] = False

        # TSCH desynchronization threshold (180 s without parent timestamp)
        for u in range(2, n_tot + 1):
            if associated[u] and not failed[u]:
                if (sf_num - last_sync_sf[u]) >= desync_sfs:
                    associated[u] = False
                    rpl_joined[u] = False
                    sync_wait_cnt[u] = 0
                    sync_cooldown[u] = 0

        eb_senders_by_off = {}
        dio_senders_off0 = []
        dis_senders_off0 = []
        rx_off_node = {}

        # Phase 1: Per-node slot-0 scheduling decision
        for u in range(1, n_tot + 1):
            if failed[u] or sf_num < boot_sf[u]:
                continue

            is_mixed_stock = (scheme_id == 37 and u != 1 and (u % 5) == 0)

            if not associated[u]:
                scan_rx_us[u] += sf_len * 10000
                continue

            if not rpl_joined[u] and u != 1:
                # Stock RPL DIS solicitation on cell (0,0) when enabled (schemes 46..49)
                if dis_flag and sf_num >= next_dis_sf[u]:
                    dis_senders_off0.append(u)
                    next_dis_sf[u] = sf_num + dis_interval_sfs
                    scan_rx_us[u] += DIS_TX_US
                    continue

                # Algorithm 1: 3-of-4 base-cell (0,0) DIO reception, 1-of-4 candidate EB sweep
                in_win = (sync_asn[u] > 0 and (asn_now - sync_asn[u]) < max_win_sfs * sf_len)
                uses_alg1 = base_uses_alg1 and (not is_mixed_stock)
                if in_win and (sf_num & 3) == 3 and uses_alg1:
                    step = (sf_num >> 2) % n_ch
                    if scheme_id in (27, 33):
                        cand_off = (n_ch - 1) - step
                    elif scheme_id == 34:
                        cand_off = rng.randrange(n_ch)
                    elif scheme_id == 38:
                        half = max(1, n_ch // 2)
                        p_t = cached_tier[parent[u]] if parent[u] > 0 else 0
                        if p_t == 0:
                            if step == 0:
                                cand_off = 0
                            elif step <= half:
                                cand_off = (2 * (step - 1) + 1) % n_ch
                            else:
                                cand_off = (2 * (step - half)) % n_ch
                        elif (p_t & 1) == 1:
                            cand_off = ((1 + 2 * step) % n_ch) if (step < half) else ((2 * (step - half)) % n_ch)
                        else:
                            cand_off = ((2 * step) % n_ch) if (step < half) else ((1 + 2 * (step - half)) % n_ch)
                    else:
                        cand_off = step
                    rx_off_node[u] = cand_off
                else:
                    rx_off_node[u] = 0
                scan_rx_us[u] += IDLE_RX_US
                continue

            # Advance RFC 6206 Trickle interval when boundary is reached
            if sf_num >= trickle_end_sf[u]:
                trickle_I_s[u] = min(TRICKLE_I_MAX_S, trickle_I_s[u] * 2.0)
                dur = max(1, int(round(trickle_I_s[u] / sf_dur_s)))
                trickle_end_sf[u] = sf_num + dur
                trickle_tx_sf[u] = sf_num + rng.randint(max(1, dur // 2), dur)
                trickle_c[u] = 0
                trickle_done[u] = False

            active_sfs[u] += 1
            nbrs = len(nbr_table[u])
            tx_prob_u8 = 255
            enable_ka = False
            subslots = 1
            ka_allow_zero = False

            if scheme_id in (1, 26, 27, 28, 34, 35, 36, 38, 45, 46):
                update_rank_tier_node(u, True, True, False, False)
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
            elif scheme_id == 42:  # ID-Hash Non-Zero {1..C-1} + Guard + ADAPT
                update_rank_tier_node(u, True, True, False, False)
                pid = parent[u]
                parent_ch_offset[u] = compute_nonzero_idhash_offset(pid, n_ch)
                u_off_42 = compute_nonzero_idhash_offset(u, n_ch)
                if u != 1 and u_off_42 == parent_ch_offset[u]:
                    u_off_42 = guard_shift_offset(u_off_42, False, n_ch)
                ch_offset[u] = u_off_42
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
            elif scheme_id == 43:  # Node-ID Parity (u mod 2) + In-Band Hash + Guard + ADAPT
                update_rank_tier_node(u, True, True, False, False)
                pid = parent[u]
                p_band = 0 if pid <= 1 else (1 + (pid & 1))
                u_band = 0 if u == 1 else (1 + (u & 1))
                parent_ch_offset[u] = compute_subband_offset(pid, p_band, n_ch)
                u_off_43 = compute_subband_offset(u, u_band, n_ch)
                if u != 1 and u_off_43 == parent_ch_offset[u]:
                    u_off_43 = guard_shift_offset(u_off_43, False, n_ch)
                ch_offset[u] = u_off_43
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
            elif scheme_id == 44:  # Pairing + ADAPT (Grafted, Alg. 1+KA)
                pid = parent[u]
                half = max(1, n_ch // 2)
                base_pair = (u // 2) % half
                p_base = (pid // 2) % half
                ch_offset[u] = (base_pair + half) % n_ch if (sf_num & 1) else base_pair
                parent_ch_offset[u] = (p_base + half) % n_ch if (sf_num & 1) else p_base
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
                ka_allow_zero = True
            elif scheme_id == 37:
                if is_mixed_stock:
                    ch_offset[u] = 0
                    tx_prob_u8 = 10
                else:
                    update_rank_tier_node(u, True, True, False, False)
                    tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                    enable_ka = True
            elif scheme_id == 41:
                update_rank_tier_node(u, True, True, False, True)
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
            elif scheme_id == 25:
                update_rank_tier_node(u, True, True, False, False)
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
            elif scheme_id == 14:
                update_rank_tier_node(u, False, False, True, False)
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
                ka_allow_zero = True
            elif scheme_id == 15:
                half = max(1, n_ch // 2)
                ch_offset[u] = ((u % half) * 2 + (sf_num & 1)) % n_ch
                parent_ch_offset[u] = ((parent[u] % half) * 2) % n_ch
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
                ka_allow_zero = True
            elif scheme_id in (16, 33):
                update_rank_tier_node(u, True, True, True, False)
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
                enable_ka = True
                ka_allow_zero = True
            elif scheme_id == 17:
                update_rank_tier_node(u, False, False, False, False)
                tx_prob_u8 = 216 // eb_mult
                enable_ka = True
            elif scheme_id == 18:
                update_rank_tier_node(u, True, False, False, False)
                tx_prob_u8 = 216 // eb_mult
                enable_ka = True
            elif scheme_id == 19:
                update_rank_tier_node(u, False, True, False, False)
                tx_prob_u8 = 216 // eb_mult
                enable_ka = True
            elif scheme_id == 20:
                update_rank_tier_node(u, True, True, False, False)
                tx_prob_u8 = 216 // eb_mult
                enable_ka = True
            elif scheme_id == 21:
                ch_offset[u] = 0
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, False, False)
            elif scheme_id == 22:
                ch_offset[u] = 0
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, False, True, False)
            elif scheme_id == 23:
                ch_offset[u] = 0
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, False)
            elif scheme_id == 24:
                ch_offset[u] = 0
                tx_prob_u8 = compute_adapt_q_u8(u, asn_now, join_asn[u], phi[u], nbrs, n_ch, sf_len, True, True, True)
            elif scheme_id == 2:
                ch_offset[u] = 0
                tx_prob_u8 = 255 // eb_mult
            elif scheme_id == 11:
                ch_offset[u] = 0
                tx_prob_u8 = 64
            elif scheme_id == 12:
                ch_offset[u] = 0
                tx_prob_u8 = 10
            elif scheme_id == 3:
                ch_offset[u] = rng.randrange(n_ch)
                tx_prob_u8 = 255 // eb_mult
            elif scheme_id == 4:
                ch_offset[u] = u % n_ch
                tx_prob_u8 = 255 // eb_mult
            elif scheme_id == 5:
                ch_offset[u] = (sf_num % atp_subslots) % n_ch
                tx_prob_u8 = 255 // eb_mult
                subslots = atp_subslots
            elif scheme_id in (6, 40):
                rgb_phase = sf_num % 3
                sub_band = (n_ch // 3) if n_ch >= 3 else 1
                ch_offset[u] = (rgb_phase * sub_band + (u % sub_band)) % n_ch
                base_p = 204 if rgb_phase == 0 else (102 if rgb_phase == 1 else 26)
                eff_mult = 2 if scheme_id == 40 else eb_mult
                tx_prob_u8 = base_p // eff_mult
            elif scheme_id == 7:
                csmj_be = 2 if nbrs > 8 else 1
                ch_offset[u] = 0
                tx_prob_u8 = (255 >> csmj_be) // eb_mult
            elif scheme_id in (8, 30, 32, 48):
                raw_p = (255 // (1 + max(0, nbrs))) // eb_mult
                ch_offset[u] = ((u * 2654435761) & 0xFFFFFFFF) % n_ch
                tx_prob_u8 = 1 if raw_p < 1 else raw_p
            elif scheme_id in (9, 39, 47):
                half = max(1, n_ch // 2)
                base_pair = (u // 2) % half
                ch_offset[u] = (base_pair + half) % n_ch if (sf_num & 1) else base_pair
                tx_prob_u8 = 128 if scheme_id == 39 else (255 // eb_mult)
            elif scheme_id in (10, 31):
                ch_offset[u] = ((u % max(1, n_ch // 2)) * 2) % n_ch
                tx_prob_u8 = 128 // eb_mult
            elif scheme_id in (13, 29, 49):
                ch_offset[u] = ((u % max(1, n_ch // 2)) * 2) % n_ch
                tx_prob_u8 = 64

            # State 1: Bounded Parent Keep-Alive RX (30 s trigger, 10-SF probe, 20-SF cooldown)
            if enable_ka and u != 1:
                sync_age = sf_num - last_sync_sf[u]
                if sync_age < ka_trigger_sfs:
                    sync_wait_cnt[u] = 0
                    sync_cooldown[u] = 0
                elif sync_cooldown[u] > 0:
                    sync_cooldown[u] -= 1
                elif sync_wait_cnt[u] < 10:
                    pid = parent[u]
                    rx_off = (
                        parent_ch_offset[u]
                        if (sync_wait_cnt[u] < 8 or pid == 1)
                        else guard_shift_offset(parent_ch_offset[u], ka_allow_zero, n_ch)
                    )
                    sync_wait_cnt[u] += 1
                    rx_off_node[u] = rx_off
                    slot0_rx_us[u] += IDLE_RX_US
                    continue
                else:
                    sync_wait_cnt[u] = 0
                    sync_cooldown[u] = 20

            # State 2 (EB TX) vs. State 3 (Base cell (0,0) DIO TX / RX)
            u_off = ch_offset[u]
            eb_gate = 216 if (u_off != 0 and tx_prob_u8 > 216) else tx_prob_u8
            dio_ready = (not trickle_done[u]) and (sf_num >= trickle_tx_sf[u])

            if u_off == 0 and dio_ready:
                trickle_done[u] = True
                if trickle_c[u] < TRICKLE_K:
                    dio_senders_off0.append(u)
                    slot0_tx_us[u] += DIO_TX_US
                    continue

            if eb_gate > 0 and rng.randrange(256) < eb_gate:
                eb_cnt[u] += subslots
                slot0_tx_us[u] += EB_TX_US * subslots
                node_ch_off_log[u] = u_off
                if u not in node_first_tx_sf:
                    node_first_tx_sf[u] = sf_num
                eb_senders_by_off.setdefault(u_off, []).append((u, subslots))
                if scheme_id == 5 and subslots > 1:
                    for s_i in range(1, subslots):
                        eb_senders_by_off.setdefault((u_off + s_i) % n_ch, []).append((u, 1))
            else:
                if dio_ready:
                    trickle_done[u] = True
                    if trickle_c[u] < TRICKLE_K:
                        dio_senders_off0.append(u)
                        slot0_tx_us[u] += DIO_TX_US
                        continue
                rx_off_node[u] = 0
                slot0_rx_us[u] += IDLE_RX_US

        # Phase 2: Physical-layer co-channel collision categorization on G_I (matches run_cooja_headless.py)
        for _off, sender_list in eb_senders_by_off.items():
            if len(sender_list) < 2:
                continue
            for idx_a in range(len(sender_list)):
                u, sub_u = sender_list[idx_a]
                adj_u = adj[u]
                has_tree = has_sib = has_cross = False
                for idx_b in range(len(sender_list)):
                    if idx_a == idx_b:
                        continue
                    v, _ = sender_list[idx_b]
                    if u == v:
                        continue
                    if (v in adj_u) or (not adj_u.isdisjoint(adj[v])):
                        if parent[u] == v or parent[v] == u:
                            has_tree = True
                        elif cached_tier[u] == cached_tier[v]:
                            has_sib = True
                        else:
                            has_cross = True
                if has_tree:
                    coll_tree += sub_u
                elif has_sib:
                    coll_sib += sub_u
                elif has_cross:
                    coll_cross += sub_u

        dio_set = set(dio_senders_off0)
        dis_set = set(dis_senders_off0)
        eb_sets_by_off = {off: {u for u, _ in sl} for off, sl in eb_senders_by_off.items()}
        off0_senders = dio_set | dis_set | eb_sets_by_off.get(0, set())

        # Phase 3: Reception for unassociated scanning pledges
        if eb_sets_by_off:
            for u in range(2, n_tot + 1):
                if failed[u] or associated[u] or sf_num < boot_sf[u]:
                    continue
                f_scan = rng.randrange(n_ch)
                match_off = (f_scan - asn_now) % n_ch
                if match_off == 0:
                    tx_here = adj[u] & off0_senders
                    if len(tx_here) == 1:
                        sender = next(iter(tx_here))
                        if sender in eb_sets_by_off.get(0, set()) and (rx_success >= 1.0 or rng.random() <= rx_success):
                            associated[u] = True
                            if sync_asn[u] == 0:
                                sync_asn[u] = asn_now if asn_now > 0 else 1
                                elapsed_sync_slots[u] = (sf_num - boot_sf[u]) * sf_len
                                if dis_flag:
                                    next_dis_sf[u] = sf_num + rng.randint(1, dis_start_sfs)
                            parent[u] = sender
                            last_sync_sf[u] = sf_num
                            try_add_nbr(u, sender)
                            cand_rank[u][sender] = (rank[sender], 0, cached_tier[sender])
                else:
                    s_set = eb_sets_by_off.get(match_off)
                    if s_set:
                        tx_here = adj[u] & s_set
                        if len(tx_here) == 1:
                            sender = next(iter(tx_here))
                            if rx_success >= 1.0 or rng.random() <= rx_success:
                                associated[u] = True
                                if sync_asn[u] == 0:
                                    sync_asn[u] = asn_now if asn_now > 0 else 1
                                    elapsed_sync_slots[u] = (sf_num - boot_sf[u]) * sf_len
                                    if dis_flag:
                                        next_dis_sf[u] = sf_num + rng.randint(1, dis_start_sfs)
                                parent[u] = sender
                                last_sync_sf[u] = sf_num
                                try_add_nbr(u, sender)
                                cand_rank[u][sender] = (rank[sender], match_off, cached_tier[sender])

        # Phase 4: Reception for associated nodes (Alg. 1 sweep, DIO join, DIS Trickle reset, State-1 KA, parent switch)
        for u, r_off in rx_off_node.items():
            if r_off == 0:
                tx_here = adj[u] & off0_senders
                if len(tx_here) != 1:
                    if len(tx_here) > 1 and not rpl_joined[u]:
                        retries[u] += 1
                    continue
                sender = next(iter(tx_here))
                if rx_success < 1.0 and rng.random() > rx_success:
                    retries[u] += 1
                    continue
                if rpl_joined[u]:
                    slot0_rx_us[u] += (FRAME_RX_US - IDLE_RX_US)
                if sender == parent[u]:
                    last_sync_sf[u] = sf_num
                    sync_wait_cnt[u] = 0
                if sender in dis_set and rpl_joined[u]:
                    reset_trickle(u, sf_num)
                elif sender in dio_set:
                    if rpl_joined[u]:
                        trickle_c[u] += 1
                    if not try_add_nbr(u, sender):
                        continue
                    link_step = 256 if rx_success >= 1.0 else int(256 / rx_success + rng.randint(-16, 48))
                    new_rank = rank[sender] + link_step
                    cand_rank[u][sender] = (rank[sender], ch_offset[sender], cached_tier[sender])
                    if not rpl_joined[u]:
                        if try_add_nbr(sender, u):
                            best_p, best_r = sender, new_rank
                            for c_p, (c_rk, _, _) in cand_rank[u].items():
                                if rpl_joined[c_p] and not failed[c_p] and (c_rk + link_step) < best_r:
                                    best_p = c_p
                                    best_r = c_rk + link_step
                            rpl_joined[u] = True
                            parent[u] = best_p
                            rank[u] = best_r
                            hop_count[u] = min(254, hop_count[best_p] + 1)
                            last_sync_sf[u] = sf_num
                            parent_switched_flag[u] = True
                            if not ever_joined[u]:
                                ever_joined[u] = True
                                join_asn[u] = asn_now
                                join_sf_map[u] = sf_num
                                elapsed_join_slots[u] = (sf_num - boot_sf[u]) * sf_len
                                phi[u] = ((u ^ sf_num) + rng.randrange(4)) % max_phi
                                joined_count += 1
                                last_join_sf = sf_num
                            reset_trickle(u, sf_num)
                    else:
                        if sender != parent[u] and (new_rank + 128 < rank[u]):
                            parent[u] = sender
                            rank[u] = new_rank
                            hop_count[u] = min(254, hop_count[sender] + 1)
                            switches[u] += 1
                            parent_switched_flag[u] = True
                            last_sync_sf[u] = sf_num
                            reset_trickle(u, sf_num)
                        elif sender == parent[u]:
                            rank[u] = new_rank
            else:
                s_set = eb_sets_by_off.get(r_off)
                if not s_set:
                    continue
                tx_here = adj[u] & s_set
                if len(tx_here) != 1:
                    continue
                sender = next(iter(tx_here))
                if rx_success < 1.0 and rng.random() > rx_success:
                    continue
                if rpl_joined[u]:
                    slot0_rx_us[u] += (FRAME_RX_US - IDLE_RX_US)
                if sender == parent[u]:
                    last_sync_sf[u] = sf_num
                    sync_wait_cnt[u] = 0
                if not rpl_joined[u] and try_add_nbr(u, sender):
                    cand_rank[u][sender] = (rank[sender], r_off, cached_tier[sender])

        if not fail_flag and joined_count >= n_nodes:
            break

    joined_ids = [u for u in range(2, n_tot + 1) if ever_joined[u]]
    num_joined = len(joined_ids)
    if num_joined == 0:
        return None, []

    asns = np.sort(np.array([elapsed_join_slots[u] for u in joined_ids], dtype=float))
    sync_arr = np.array([elapsed_sync_slots[u] for u in joined_ids], dtype=float)

    c_frac = float(num_joined) / float(n_nodes)
    mean_join_asn = float(np.mean(asns))
    mean_sync_asn = float(np.mean(sync_arr))
    p95_join_asn = float(np.percentile(asns, 95))
    median_join_s = float(np.median(asns)) * 0.01

    t_pen_s = (
        320.0
        * (0.65 + 0.35 * (n_nodes / 200.0))
        * (0.75 + 0.25 * (sf_len / 61.0))
        * (0.80 + 0.20 * (n_ch / 16.0))
    )
    t_sync_hat_s = 0.55 * median_join_s + (1.0 - c_frac) * t_pen_s
    t_end_s = (last_join_sf * sf_dur_s) if (num_joined >= n_nodes and not fail_flag) else timeout_s

    slot0_tx_us_sum = sum(slot0_tx_us[1:])
    slot0_rx_us_sum = sum(slot0_rx_us[1:])
    scan_rx_us_sum = sum(scan_rx_us[2:])

    slot0_equiv_tx_ticks = int(round((slot0_tx_us_sum + (18.8 / 17.4) * slot0_rx_us_sum) * 32768.0 / 1e6))
    slot0_tx_only_ticks = int(round(slot0_tx_us_sum * 32768.0 / 1e6))
    scan_rx_ticks_32k = int(round(scan_rx_us_sum * 32768.0 / 1e6))

    mean_switches = sum(switches[2:]) / float(max(1, num_joined))
    mean_retries = sum(retries[2:]) / float(max(1, n_nodes))
    eb_rate = float(sum(eb_cnt[1:])) / float(max(1, sum(active_sfs[1:])))
    coll_total = coll_tree + coll_sib + coll_cross

    # Real neighbor-table occupancy percentiles & Eq. (17) scan inputs at join time (cols 23..28)
    nbr_vals = np.array([len(nbr_table[u]) for u in joined_ids], dtype=float)
    nbr_p50 = float(np.percentile(nbr_vals, 50)) if len(nbr_vals) > 0 else 0.0
    nbr_p95 = float(np.percentile(nbr_vals, 95)) if len(nbr_vals) > 0 else 0.0
    nbr_p99 = float(np.percentile(nbr_vals, 99)) if len(nbr_vals) > 0 else 0.0

    mean_tier = float(np.mean([cached_tier[u] for u in joined_ids])) if joined_ids else 0.0
    c_i_samples, co_off_samples = [], []
    for u in joined_ids:
        j_sf = join_sf_map[u]
        off_counts = {}
        for nb in adj.get(u, set()):
            if nb in node_ch_off_log and (nb == parent[u] or node_first_tx_sf.get(nb, 10**9) <= j_sf):
                off = node_ch_off_log[nb]
                off_counts[off] = off_counts.get(off, 0) + 1
        if off_counts:
            c_i_samples.append(len(off_counts))
            co_off_samples.extend(off_counts.values())

    mean_c_i = float(np.mean(c_i_samples)) if c_i_samples else 0.0
    mean_co_off = float(np.mean(co_off_samples)) if co_off_samples else 0.0

    row = [
        n_nodes, area, n_ch, sf_len, eb_mult,
        num_joined, round(t_sync_hat_s, 4), round(p95_join_asn, 2), round(mean_join_asn, 2),
        coll_total, slot0_equiv_tx_ticks, scan_rx_ticks_32k,
        seed, round(eb_rate, 4), coll_tree, coll_sib, coll_cross,
        round(mean_switches, 4), round(mean_retries, 4),
        slot0_tx_only_ticks, round(t_end_s, 4), round(mean_sync_asn, 2),
        round(nbr_p50, 2), round(nbr_p95, 2), round(nbr_p99, 2),
        round(mean_c_i, 4), round(mean_tier, 4), round(mean_co_off, 4),
    ]
    join_seconds = [float(a) * 0.01 for a in asns]
    return row, join_seconds


def _worker_task(task):
    scheme_id, n_nodes, area, n_ch, sf_len, eb_mult, seed, nbr_cap, timeout_s, atp_subslots = task
    row, join_secs = simulate_seed(
        scheme_id=scheme_id, n_nodes=n_nodes, area=area, n_ch=n_ch,
        sf_len=sf_len, eb_mult=eb_mult, seed=seed, nbr_cap=nbr_cap,
        timeout_s=timeout_s, atp_subslots=atp_subslots,
    )
    return scheme_id, nbr_cap, n_nodes, area, n_ch, sf_len, eb_mult, row, join_secs


def build_task_list(mode, seeds, schemes, nbr_cap, timeout_s, atp_subslots):
    tasks = []
    seen = set()

    def add_cfg(sid, n_nodes, area, n_ch, sf_len, eb_mult, cap):
        eff_area = compute_constant_density_area(n_nodes) if area is None else area
        for s in seeds:
            key = (sid, n_nodes, eff_area, n_ch, sf_len, eb_mult, s, cap)
            if key not in seen:
                seen.add(key)
                tasks.append((sid, n_nodes, eff_area, n_ch, sf_len, eb_mult, s, cap, timeout_s, atp_subslots))

    if mode in ("sweeps", "full"):
        sweep_schemes = [sid for sid in schemes if sid <= 8]
        for sid in sweep_schemes:
            for n in range(50, 501, 50):
                add_cfg(sid, n, None, 16, 61, 1, nbr_cap)   # Fig. 8 (constant density)
                add_cfg(sid, n, 100, 16, 61, 1, nbr_cap)    # Figs. 9-10 (spatial density)
                if n in (100, 500) and sid in VERIFIED_64NBR_SCHEMES:
                    add_cfg(sid, n, 100, 16, 61, 1, 64)     # 64-entry verification
            for ch in (2, 4, 6, 8, 10, 12, 14):
                add_cfg(sid, 200, None, ch, 61, 1, nbr_cap) # Fig. 11 (channels)
            for sf in (23, 31, 41, 53, 73, 83, 97, 101):
                add_cfg(sid, 200, None, 16, sf, 1, nbr_cap) # Fig. 12 (slotframe)
            for eb in (2, 3, 4, 5, 6, 7, 8):
                add_cfg(sid, 200, None, 16, 61, eb, nbr_cap)# Fig. 13 (EB multiplier)

    if mode in ("benchmark", "full"):
        for sid in schemes:
            for n in (100, 500):
                add_cfg(sid, n, 100, 16, 61, 1, nbr_cap)
                if sid in VERIFIED_64NBR_SCHEMES:
                    add_cfg(sid, n, 100, 16, 61, 1, 64)
            if sid == 37:
                add_cfg(37, 200, 100, 16, 61, 1, nbr_cap)

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="6PySCH — Discrete-event 6TiSCH simulator aligned with Contiki-NG Cooja (49 schemes, 28 columns)"
    )
    parser.add_argument("--mode", choices=["benchmark", "sweeps", "full"], default="benchmark",
                        help="benchmark: Tables IV-V (N=100,500); sweeps: Figs 8-13; full: everything")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of paired seeds (default: 5)")
    parser.add_argument("--schemes", type=str, default="all",
                        help="Comma-separated scheme IDs (e.g. '1,2,8,13') or 'all' (1..49)")
    parser.add_argument("--nbr-cap", type=int, default=DEFAULT_NBR_CAP, help="Default neighbor table cap (default: 16)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Simulation horizon in seconds (default: 300)")
    parser.add_argument("--atp-subslots", type=int, default=DEFAULT_ATP_SUBSLOTS, help="ATP subslots for scheme 5 (default: 2)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Parallel worker processes")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "Statistics"),
                        help="Output directory for Statistics files")
    args = parser.parse_args()

    if args.schemes.strip().lower() == "all":
        schemes = sorted(SCHEME_MAP.keys())
    else:
        schemes = [int(x.strip()) for x in args.schemes.split(",") if x.strip()]

    seeds = list(range(1, args.seeds + 1))
    tasks = build_task_list(args.mode, seeds, schemes, args.nbr_cap, args.timeout, args.atp_subslots)

    os.makedirs(args.output_dir, exist_ok=True)
    for old_f in glob.glob(os.path.join(args.output_dir, "*_stats_.txt")) + glob.glob(os.path.join(args.output_dir, "join_times_*.csv")):
        try:
            os.remove(old_f)
        except OSError:
            pass

    print(f"6PySCH ({args.mode}): running {len(tasks)} simulation jobs across {args.workers} workers "
          f"(seeds={args.seeds}, schemes={len(schemes)}, default Cap={args.nbr_cap}, T_max={args.timeout:.0f}s)")
    t0 = time.time()

    rows_by_file = {}
    cdf_by_file = {}
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker_task, t) for t in tasks]
        for fut in as_completed(futures):
            sid, cap, n_nodes, area, n_ch, sf_len, eb_mult, row, join_secs = fut.result()
            done += 1
            if row is not None:
                stat_key = SCHEME_MAP[sid]
                if sid == 5 and args.atp_subslots != 2:
                    stat_key = f"a_{args.atp_subslots}"
                suffix = "_64nbr" if cap == 64 else ""
                fname = f"_{stat_key}{suffix}_cooja_stats_.txt"
                rows_by_file.setdefault(fname, []).append(row)

                if (
                    join_secs and area == 100 and n_nodes in (100, 500)
                    and n_ch == 16 and sf_len == 61 and eb_mult == 1 and cap == args.nbr_cap
                ):
                    cname = f"join_times_N{n_nodes}_scheme{sid}.csv"
                    cdf_by_file.setdefault(cname, []).extend(join_secs)

            if done % max(1, len(tasks) // 40) == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"\rProgress: {100.0 * done / len(tasks):6.2f}% ({done}/{len(tasks)}) | Elapsed: {elapsed:.1f}s", end="", flush=True)

    print("\nWriting sorted Statistics files...")
    for fname, rlist in sorted(rows_by_file.items()):
        rlist.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[12]))
        cooja_path = os.path.join(args.output_dir, fname)
        legacy_path = os.path.join(args.output_dir, fname.replace("_cooja_stats_.txt", "_stats_.txt"))
        lines = [",".join(map(str, r)) + "," for r in rlist]
        content = "\n".join(lines) + "\n"
        with open(cooja_path, "w") as fp:
            fp.write(content)
        with open(legacy_path, "w") as fp:
            fp.write(content)

    for cname, vals in sorted(cdf_by_file.items()):
        vals.sort()
        np.savetxt(os.path.join(args.output_dir, cname), np.array(vals, dtype=float), fmt="%.4f", delimiter=",")

    print(f"Completed {len(tasks)} runs in {time.time() - t0:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()