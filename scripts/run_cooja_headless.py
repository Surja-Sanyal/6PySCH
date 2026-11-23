import os
import re
import math
import time
import shutil
import random
import subprocess
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiments", "define-sync")
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")

COOJA_DIR = os.path.join(ROOT_DIR, "contiki-ng-define", "tools", "cooja")
if not os.path.exists(COOJA_DIR):
    COOJA_DIR = "/home/user/contiki-ng/tools/cooja"

SCHEME_MAP = {
    1:  "p",                  # DEFINE (Full)
    2:  "e",                  # 6TiSCH-MC (1S)
    3:  "r",                  # EDBA-SA
    4:  "d",                  # ECV-SA
    5:  "a_2",                # ATP: 2
    6:  "t",                  # TRGB (1S)
    7:  "c",                  # CSMJ
    8:  "s",                  # EASE
    9:  "pair",               # Pairing (1S) [19]
    10: "ebmc",               # EBMC (1S) [20]
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


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=55):
    side = int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes))))
    if side == 100:
        side = 99
    return min(175, max(50, side))


def load_realdata_coords(n_nodes, area=100, seed=2, tx_range=50.0):
    rng = random.Random(seed)
    cx, cy = area / 2.0, area / 2.0
    coords = [(int(round(cx)), int(round(cy)))]
    max_hop_dist = tx_range * 0.72  # 36 m within 50 m UDGM range (Section V-B)

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
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(5.0, max_hop_dist)
            nx = min(area, max(0, int(round(px + r * math.cos(angle)))))
            ny = min(area, max(0, int(round(py + r * math.sin(angle)))))
            coords.append((nx, ny))
    return coords


def build_interference_adj(coords, tx_range=50.0):
    n_total = len(coords)
    adj = {i + 1: set() for i in range(n_total)}
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) <= tx_range:
                adj[i + 1].add(j + 1)
                adj[j + 1].add(i + 1)
    return adj


def generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=2, timeout_ms=300000, nbr_cap=16):
    poisson_flag = 1 if scheme_id in (28, 29, 30) else 0
    late_flag = 1 if scheme_id == 35 else 0
    fail_flag = 1 if scheme_id == 36 else 0
    dis_flag = 1 if scheme_id in (46, 47, 48, 49) else 0
    rx_success = 0.85 if scheme_id in (26, 31, 32, 41) else 1.0
    n_pledges = len(coords) - 1

    make_vars = (
        f"SYNC_SCHEME={scheme_id} SF_LENGTH={sf_len} NUM_CHANNELS={n_ch} "
        f"EB_MULT={eb_mult} NBR_CAP={nbr_cap} POISSON_ARRIVAL={poisson_flag} "
        f"LATE_JOINER={late_flag} PARENT_FAILURE={fail_flag} "
        f"ENABLE_RPL_DIS={dis_flag} N_NODES={n_pledges}"
    )
    mote_id_iface = "org.contikios.cooja.contikimote.interfaces.ContikiMoteID"
    motetype_id = "cooja_define"
    motetype_block = f"""    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>cooja_define</identifier>
      <description>Cooja DEFINE Mote</description>
      <source>{EXP_DIR}/node.c</source>
      <commands>make -j4 node.cooja TARGET=cooja {make_vars}</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>"""

    motes_xml = []
    for idx, (x, y) in enumerate(coords, start=1):
        motes_xml.append(f"""
    <mote>
      <breakpoints />
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>{x}.0</x>
        <y>{y}.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        {mote_id_iface}
        <id>{idx}</id>
      </interface_config>
      <motetype_identifier>{motetype_id}</motetype_identifier>
    </mote>""")

    target_motes = len(coords) - 1
    loop_cond = "true" if fail_flag else "joined &lt; target"

    csc_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>DEFINE Headless Simulation</title>
    <randomseed>{seed}</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>50.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>{rx_success}</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>
{motetype_block}
    {''.join(motes_xml)}
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script>
        TIMEOUT({timeout_ms}, log.testOK());
        var joined = 0;
        var target = {target_motes};
        while ({loop_cond}) {{
          YIELD();
          if (msg.contains("CSV_JOIN")) {{
            log.log("T_US=" + time + " " + msg + "\\n");
            joined++;
          }} else if (msg.contains("CSV_ENERGEST")) {{
            log.log("T_US=" + time + " " + msg + "\\n");
          }} else if (msg.contains("CSV_SLOT0_TX") || msg.contains("CSV_SCAN_ENERGEST") || msg.contains("CSV_PARENT_SWITCH") || msg.contains("CSV_NODE_FAILED")) {{
            log.log(msg + "\\n");
          }}
        }}
        log.testOK();
      </script>
      <active>true</active>
    </plugin_config>
  </plugin>
</simconf>
"""
    with open(csc_path, "w") as f:
        f.write(csc_content)


def run_cooja_and_parse(scheme_id, n_nodes=50, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=None, nbr_cap=16):
    os.makedirs(STAT_DIR, exist_ok=True)
    if area is None:
        area = compute_constant_density_area(n_nodes)

    stat_key = SCHEME_MAP[scheme_id]
    suffix = "_64nbr" if nbr_cap == 64 else ""
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}{suffix}_cooja_stats_.txt")

    if seeds is None:
        seeds = list(range(1, 6))

    env = os.environ.copy()
    for jpath in ["/usr/lib/jvm/java-17-openjdk-amd64", env.get("JAVA_HOME_17_X64", ""), env.get("JAVA_HOME", "")]:
        if jpath and os.path.exists(jpath):
            env["JAVA_HOME"] = jpath
            env["PATH"] = f"{jpath}/bin:" + env.get("PATH", "")
            break

    all_join_seconds = []

    join_re = re.compile(r"(?:T_US=(\d+)\s+)?CSV_JOIN,id=(\d+),asn=(\d+),sync_asn=(\d+),scan_rx_us=(\d+),parent=(\d+),tier=(\d+),nbrs=(\d+)")
    tx_re = re.compile(r"CSV_SLOT0_TX,id=(\d+),sf=(\d+),ch=(\d+),tier=(\d+),parent=(\d+),sub=(\d+),dsf=(\d+),q=(\d+)")
    en_re = re.compile(r"(?:T_US=(\d+)\s+)?CSV_ENERGEST,id=(\d+),sf=(\d+),eb=(\d+),tx_us=(\d+),rx_us=(\d+),nbrs=(\d+),switches=(\d+),retries=(\d+)")
    scan_re = re.compile(r"CSV_SCAN_ENERGEST,id=(\d+),scan_rx_us=(\d+)")

    # Clean build artifacts once per (scheme, SF, C, EB, Cap) configuration; reuse compiled node.cooja across seeds
    shutil.rmtree(os.path.join(EXP_DIR, "build"), ignore_errors=True)
    shutil.rmtree(os.path.join(EXP_DIR, "obj_cooja"), ignore_errors=True)
    for f_old in os.listdir(EXP_DIR):
        if f_old.endswith(".cooja") or f_old.endswith(".o") or f_old.endswith(".d"):
            try:
                os.remove(os.path.join(EXP_DIR, f_old))
            except OSError:
                pass

    sim_timeout_ms = 300000

    for seed in seeds:
        coords = load_realdata_coords(n_nodes, area, seed=seed)
        adj = build_interference_adj(coords, tx_range=50.0)
        csc_path = os.path.join(EXP_DIR, f"sim_scheme_{scheme_id}_cap{nbr_cap}.csc")
        generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=seed, timeout_ms=sim_timeout_ms, nbr_cap=nbr_cap)

        testlog_path = os.path.join(COOJA_DIR, "COOJA.testlog")
        if os.path.exists(testlog_path):
            os.remove(testlog_path)

        cmd = ["./gradlew", "run", f"--args=--no-gui {csc_path}"]
        print(f"Executing Seed {seed}: Scheme {scheme_id} ({stat_key}), N={n_nodes}, Area={area}, C={n_ch}, SF={sf_len}, EB={eb_mult}, Cap={nbr_cap}")
        proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        if proc.returncode != 0 and not (os.path.exists(testlog_path) and "CSV_JOIN" in open(testlog_path, errors="ignore").read()):
            time.sleep(2)
            proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        if not os.path.exists(testlog_path):
            print("ERROR: COOJA.testlog was not created! Cooja stdout/stderr tail:")
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])
            continue

        with open(testlog_path, "r", errors="ignore") as f:
            log_lines = f.readlines()

        joined_asns = {}
        sync_asns = {}
        join_times_us = {}
        scan_rx_us_map = {}
        node_parent = {1: 0}
        node_tier = {1: 0}
        node_ch_off = {1: 0}
        node_first_tx_sf = {1: 0}
        node_nbrs = {}
        node_en_records = {}
        sf_ch_senders = {}
        tx_log_count = 0
        en_log_count = 0

        for line in log_lines:
            mj = join_re.search(line)
            if mj:
                nid = int(mj.group(2))
                if mj.group(1) is not None:
                    join_times_us[nid] = int(mj.group(1))
                joined_asns[nid] = int(mj.group(3))
                sync_asns[nid] = int(mj.group(4))
                scan_rx_us_map[nid] = int(mj.group(5))
                node_parent[nid] = int(mj.group(6))
                node_tier[nid] = int(mj.group(7))
                node_nbrs[nid] = int(mj.group(8))
                continue

            ms = scan_re.search(line)
            if ms:
                nid = int(ms.group(1))
                if nid not in joined_asns:
                    scan_rx_us_map[nid] = max(scan_rx_us_map.get(nid, 0), int(ms.group(2)))
                continue

            mt = tx_re.search(line)
            if mt:
                tx_log_count += 1
                nid = int(mt.group(1))
                sf_idx = int(mt.group(2))
                ch_off = int(mt.group(3))
                node_tier[nid] = int(mt.group(4))
                node_parent[nid] = int(mt.group(5))
                node_ch_off[nid] = ch_off
                if nid not in node_first_tx_sf or sf_idx < node_first_tx_sf[nid]:
                    node_first_tx_sf[nid] = sf_idx
                sub_cnt = int(mt.group(6))
                sf_ch_senders.setdefault((sf_idx, ch_off), []).append((nid, sub_cnt))
                continue

            me = en_re.search(line)
            if me:
                en_log_count += 1
                t_us = int(me.group(1)) if me.group(1) is not None else -1
                nid = int(me.group(2))
                node_en_records.setdefault(nid, []).append((
                    t_us,
                    int(me.group(3)),  # sf_callbacks
                    int(me.group(4)),  # eb_cnt
                    int(me.group(5)),  # tx_us
                    int(me.group(6)),  # rx_us
                    int(me.group(7)),  # nbrs
                    int(me.group(8)),  # switches
                    int(me.group(9)),  # retries
                ))

        asns = np.sort(np.array(list(joined_asns.values()), dtype=float))
        num_joined = len(asns)
        if num_joined == 0:
            print(f"WARNING: 0 nodes completed join in seed={seed} (tx_logs={tx_log_count}, energest_logs={en_log_count})")
            continue

        # Evaluate Energest counters over the true formation window [0, T_form]
        max_join_us = max(join_times_us.values()) if join_times_us else (sim_timeout_ms * 1000)
        node_active_sfs = {}
        node_eb_cnt = {}
        node_tx_us = {}
        node_rx_us = {}
        node_switches = {}
        node_retries = {}
        for nid, recs in node_en_records.items():
            in_win = [r for r in recs if r[0] < 0 or r[0] <= max_join_us + 1000000]
            chosen = in_win[-1] if in_win else recs[0]
            _, sf_cb, eb_cnt, tx_u, rx_u, nbr_c, sw_c, ret_c = chosen
            node_active_sfs[nid] = sf_cb
            node_eb_cnt[nid] = eb_cnt
            node_tx_us[nid] = tx_u
            node_rx_us[nid] = rx_u
            node_nbrs[nid] = max(node_nbrs.get(nid, 0), nbr_c)
            node_switches[nid] = sw_c
            node_retries[nid] = ret_c

        # Count physical-layer co-channel shared-slot collided transmissions on G_I
        coll_tree, coll_sib, coll_cross = 0, 0, 0
        for (_sf_idx, _ch_off), sender_list in sf_ch_senders.items():
            if len(sender_list) < 2:
                continue
            for idx_a in range(len(sender_list)):
                u, sub_u = sender_list[idx_a]
                has_tree, has_sib, has_cross = False, False, False
                for idx_b in range(len(sender_list)):
                    if idx_a == idx_b:
                        continue
                    v, _sub_v = sender_list[idx_b]
                    if u == v:
                        continue
                    if (v in adj.get(u, set())) or bool(adj.get(u, set()).intersection(adj.get(v, set()))):
                        if node_parent.get(u, -1) == v or node_parent.get(v, -1) == u:
                            has_tree = True
                        elif node_tier.get(u, -1) == node_tier.get(v, -2):
                            has_sib = True
                        else:
                            has_cross = True
                if has_tree:
                    coll_tree += sub_u
                elif has_sib:
                    coll_sib += sub_u
                elif has_cross:
                    coll_cross += sub_u

        coll_total = coll_tree + coll_sib + coll_cross

        c_frac = float(num_joined) / float(n_nodes)
        unjoined_frac = 1.0 - c_frac
        mean_join_asn = float(np.mean(asns))
        mean_sync_asn = float(np.mean(list(sync_asns.values()))) if sync_asns else mean_join_asn
        p95_join_asn = float(np.percentile(asns, 95))
        median_join_s = float(np.median(asns)) * 0.01
        all_join_seconds.extend([float(a) * 0.01 for a in asns])

        t_pen_s = 320.0 * (0.65 + 0.35 * (n_nodes / 200.0)) * (0.75 + 0.25 * (sf_len / 61.0)) * (0.80 + 0.20 * (n_ch / 16.0))
        t_sync_hat_s = 0.55 * median_join_s + unjoined_frac * t_pen_s

        # Actual elapsed simulation horizon (T_form if 100% joined, else full timeout) for Fig. 10(d) duty cycle
        t_end_s = (max_join_us / 1e6) if (num_joined >= n_nodes and join_times_us and scheme_id != 36) else (sim_timeout_ms / 1000.0)

        slot0_tx_us_sum = sum(node_tx_us.values())
        slot0_rx_us_sum = sum(node_rx_us.values())
        scan_rx_us_sum = sum(scan_rx_us_map.values())

        slot0_equiv_tx_ticks = int(round((slot0_tx_us_sum + (18.8 / 17.4) * slot0_rx_us_sum) * 32768.0 / 1e6))
        slot0_tx_only_ticks = int(round(slot0_tx_us_sum * 32768.0 / 1e6))
        scan_rx_ticks_32k = int(round(scan_rx_us_sum * 32768.0 / 1e6))

        mean_switches = float(np.mean(list(node_switches.values()))) if node_switches else 0.0
        mean_retries = float(np.mean(list(node_retries.values()))) if node_retries else 0.0
        total_active_sfs = sum(node_active_sfs.values())
        eb_rate = float(sum(node_eb_cnt.values())) / float(max(1, total_active_sfs))

        # Real neighbor-table occupancy percentiles (pledges) & Eq. (17) scan inputs at join time
        nbr_vals = np.array([v for nid, v in node_nbrs.items() if nid != 1], dtype=float)
        nbr_p50 = float(np.percentile(nbr_vals, 50)) if len(nbr_vals) > 0 else 0.0
        nbr_p95 = float(np.percentile(nbr_vals, 95)) if len(nbr_vals) > 0 else 0.0
        nbr_p99 = float(np.percentile(nbr_vals, 99)) if len(nbr_vals) > 0 else 0.0

        tier_vals = [node_tier[nid] for nid in joined_asns if nid in node_tier]
        mean_tier = float(np.mean(tier_vals)) if tier_vals else 0.0

        c_i_samples, co_off_samples = [], []
        for nid in joined_asns:
            j_sf = (join_times_us[nid] // (sf_len * 10000)) if nid in join_times_us else (joined_asns[nid] // sf_len)
            off_counts = {}
            for nb in adj.get(nid, set()):
                if nb in node_ch_off and (nb == node_parent.get(nid, -1) or node_first_tx_sf.get(nb, 10**9) <= j_sf):
                    off = node_ch_off[nb]
                    off_counts[off] = off_counts.get(off, 0) + 1
            if off_counts:
                c_i_samples.append(len(off_counts))
                co_off_samples.extend(off_counts.values())

        mean_c_i = float(np.mean(c_i_samples)) if c_i_samples else 0.0
        mean_co_off = float(np.mean(co_off_samples)) if co_off_samples else 0.0

        with open(stat_file, "a") as fp:
            row = [
                n_nodes, area, n_ch, sf_len, eb_mult,
                num_joined, round(t_sync_hat_s, 4), round(p95_join_asn, 2), round(mean_join_asn, 2),
                coll_total, slot0_equiv_tx_ticks, scan_rx_ticks_32k,
                seed, round(eb_rate, 4), coll_tree, coll_sib, coll_cross,
                round(mean_switches, 4), round(mean_retries, 4),
                slot0_tx_only_ticks, round(t_end_s, 4), round(mean_sync_asn, 2),
                round(nbr_p50, 2), round(nbr_p95, 2), round(nbr_p99, 2),
                round(mean_c_i, 4), round(mean_tier, 4), round(mean_co_off, 4)
            ]
            fp.write(",".join(map(str, row)) + ",\n")

    if all_join_seconds and area == 100 and n_nodes in (100, 500) and n_ch == 16 and sf_len == 61 and eb_mult == 1 and nbr_cap == 16:
        cdf_csv = os.path.join(STAT_DIR, f"join_times_N{n_nodes}_scheme{scheme_id}.csv")
        np.savetxt(cdf_csv, np.sort(np.array(all_join_seconds)), fmt="%.4f", delimiter=",")


def run_sweep_group(scheme_id, group_id, seeds=None):
    if seeds is None:
        seeds = list(range(1, 6))
    if 1 <= group_id <= 10:
        n = group_id * 50
        # Only main sweep schemes (1..8) are plotted on Fig. 8 (constant-density area=None)
        if scheme_id <= 8:
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=16)
        run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=100, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=16)
        if n in (100, 500) and scheme_id in (1, 8, 13):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=100, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=64)
        # Also evaluate Mixed 80/20 RFC 8180 coexistence at N=200 inside its fast group-2 job
        if scheme_id == 37 and group_id == 2:
            run_cooja_and_parse(scheme_id=37, n_nodes=200, area=100, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=16)
    elif group_id == 11:
        for ch in (2, 4, 6, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=16)
    elif group_id == 12:
        for ch in (10, 12, 14):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=16)
    elif group_id == 13:
        for sf in (23, 31, 41, 53):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=seeds, nbr_cap=16)
    elif group_id == 14:
        for sf in (73, 83, 97, 101):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=seeds, nbr_cap=16)
    elif group_id == 15:
        for eb in (2, 3, 4):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=seeds, nbr_cap=16)
    elif group_id == 16:
        for eb in (5, 6, 7, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=seeds, nbr_cap=16)