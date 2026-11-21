import os
import glob
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")
os.makedirs(STAT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Zolertia Z1 radio energy & exact Cooja horizon constants
# (V_sup = 3.0 V, I_tx = 17.4 mA, I_rx = 18.8 mA, 32,768 Hz Energest, T_max = 300 s)
TX_MJ_PER_TICK = 3.0 * 17.4 / 32768.0    # mJ per 32,768-Hz TX-equivalent tick
RX_J_PER_TICK  = 3.0 * 0.0188 / 32768.0  # J  per 32,768-Hz RX tick
HORIZON_S      = 300.0                   # Exact Cooja simulation horizon (matches sim_timeout_ms = 300000)
SCAN_POWER_W   = 3.0 * 0.0188            # 0.0564 W continuous RX scan power
STRAND_J       = SCAN_POWER_W * HORIZON_S  # 16.92 J per unjoined node over 300 s

MECHANISMS = {
    "p":   ("DEFINE",     "#0044cc", "o", "-",                2.5, "#0044cc"),
    "e":   ("6TiSCH-MC",  "#e66100", "s", "--",               1.85, "white"),
    "r":   ("EDBA-SA",    "#d92121", "^", "-.",               1.85, "white"),
    "d":   ("ECV-SA",     "#1b9e3e", "v", ":",                1.95, "white"),
    "a_2": ("ATP: 2",     "#4d4d4d", "D", (0, (5, 2)),        1.85, "white"),
    "t":   ("TRGB",       "#7b3294", "P", (0, (3, 1.5, 1, 1.5)), 1.85, "white"),
    "c":   ("CSMJ",       "#8c510a", "X", (0, (4, 1.5)),      1.85, "white"),
    "s":   ("EASE",       "#c51b7d", "*", (0, (2, 1)),        1.85, "white"),
}

ORDER = ["e", "c", "a_2", "r", "d", "t", "s", "p"]

TABLE_IV_BASELINES = [
    ("e",       "6TiSCH-MC (1S)"),
    ("mc_4s",   "6TiSCH-MC (Tuned 4S)"),
    ("mc_26s",  "6TiSCH-MC (Default 26S)"),
    ("c",       "CSMJ"),
    ("a_2",     "ATP: 2"),
    ("r",       "EDBA-SA"),
    ("d",       "ECV-SA"),
    ("pair",    "Pairing"),
    ("t",       "TRGB"),
    ("ebmc",    "EBMC (1S)"),
    ("ebmc_2s", "EBMC (Tuned 2S)"),
    ("s",       "EASE"),
]

TABLE_IV_ROWS_SPEC = [
    ("e",                  "6TiSCH-MC ($I_{eb}=1S$) \\cite{vilajosana2017minimal}"),
    ("mc_4s",              "6TiSCH-MC (Tuned $I_{eb}=4S$)"),
    ("mc_26s",             "6TiSCH-MC (Default $I_{eb}=26S$)"),
    ("c",                  "CSMJ \\cite{chang2024csmj}"),
    ("a_2",                "ATP: 2 \\cite{karalis2018atp}"),
    ("r",                  "EDBA-SA \\cite{khoufi2018enhanced}"),
    ("d",                  "ECV-SA \\cite{vogli2018fast}"),
    ("pair",               "Pairing \\cite{pairing2024icufn}"),
    ("t",                  "TRGB \\cite{kalita2023trgb}"),
    ("ebmc",               "EBMC ($I_{eb}=1S$) \\cite{park2025ebmc}"),
    ("ebmc_2s",            "EBMC (Tuned $I_{eb}=2S$)"),
    ("s",                  "EASE \\cite{kalita2025ease}"),
    ("ease_adapt",         "EASE + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
    ("ebmc_adapt",         "EBMC + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
    ("idhash_guard_adapt", "ID-Hash + Guard + \\texttt{ADAPT} (Control)"),
    ("idhash_rev",         "ID-Hash + Guard + \\texttt{ADAPT} (Reversed Sweep)"),
    ("assign_hyst",        "\\texttt{ASSIGN} (Hysteresis Only, No Refresh/Guard)"),
    ("assign_ref",         "\\texttt{ASSIGN} (Refresh Only, No Guard, $1S$)"),
    ("assign_guard",       "\\texttt{ASSIGN} (Guard Only, Hysteresis, $1S$)"),
    ("assign_both",        "\\texttt{ASSIGN}-Only (Refresh + Guard, $1S$)"),
    ("adapt_age",          "\\texttt{ADAPT} (Age Only, Cell 0)"),
    ("adapt_dens",         "\\texttt{ADAPT} (Density Only, Cell 0)"),
    ("adapt_nophase",      "\\texttt{ADAPT} (No Phase $\\phi_u=0$, Cell 0)"),
    ("adapt_full",         "\\texttt{ADAPT}-Only (Full, Cell 0)"),
    ("define_noka",        "\\texttt{DEFINE} (No Keepalive)"),
    ("ebmc_mrm",           "EBMC ($I_{eb}=1S$, Lossy UDGM 85\\% PRR)"),
    ("ease_mrm",           "EASE (Lossy UDGM 85\\% PRR)"),
    ("define_mrm",         "\\texttt{DEFINE} (Lossy UDGM 85\\% PRR)"),
    ("define_rev",         "\\texttt{DEFINE} (Reversed Sweep Order)"),
    ("define_rand",        "\\texttt{DEFINE} (Random Sweep Order)"),
    ("ebmc_2s_poisson",    "EBMC (Tuned $2S$, Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("ease_poisson",       "EASE (Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("define_poisson",     "\\texttt{DEFINE} (Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("define_late",        "\\texttt{DEFINE} (Late Joiners, 20\\% at $t=150\\text{ s}$)"),
    ("define_fail",        "\\texttt{DEFINE} (10\\% Relay Failure at $t=120\\text{ s}$)"),
    ("p",                  "\\textbf{\\texttt{DEFINE} (Full, 16-Entry Table)}"),
]


def save_standalone_legend():
    handles = []
    for key in ["p", "e", "r", "d", "a_2", "t", "c", "s"]:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        ms = 9.0 if marker == "*" else 7.2
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=lw,
                markersize=ms,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=1.6,
                label=label,
            )
        )
    fig = plt.figure(figsize=(7.4, 0.72), dpi=300)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        facecolor="white",
        fontsize=10.5,
        handlelength=3.2,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    plt.axis("off")
    for p in [os.path.join(ROOT_DIR, "legend_only.png"), os.path.join(PLOTS_DIR, "legend_only.png")]:
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(RESULTS_DIR, "legend_only.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _solve_eq3_fixed_point(M_nbrs, p_tx_eb, p_tx_dio, P_L, C):
    """Exact fixed-point iteration of Equation (3) with uniform prior P(K=k) = 1/M."""
    p_rx_eb = p_tx_eb * (1.0 - P_L) / float(C)
    for _ in range(30):
        acc = 0.0
        for k in range(1, M_nbrs + 1):
            term = (
                k * p_tx_eb
                * ((1.0 - p_tx_eb) ** (k - 1))
                * ((1.0 - p_tx_dio) ** (k - 1))
                * ((1.0 - p_rx_eb) ** (M_nbrs - k))
                * (1.0 - P_L)
                / float(M_nbrs)
            )
            acc += term / float(C)
        p_rx_eb = max(1e-5, min(0.95, acc))
    return p_rx_eb


def _run_markov_density_point(args):
    n_nodes, area, realdata, sims, S, C, p_tx_eb, p_tx_dio, P_L, TxFactor, RxFactor = args
    slot_dur_s = 0.01
    v_sup = 3.0  # Zolertia Z1 supply voltage (3.0 V)
    e_tx_mJ = v_sup * TxFactor * slot_dur_s  # 0.522 mJ per slot
    e_rx_mJ = v_sup * RxFactor * slot_dur_s  # 0.564 mJ per slot
    comm_range = 50.0
    chs = [i + 1 for i in range(C)]

    act_t_list, act_ej_list, act_ep_list, act_net_list = [], [], [], []
    deg_samples = []

    for sim in range(sims):
        rng = random.Random(700 + n_nodes * 31 + sim * 137)
        cx, cy = area / 2.0, area / 2.0
        coords = [(cx, cy)]
        for _ in range(n_nodes):
            for _att in range(250):
                x = rng.uniform(0.0, area)
                y = rng.uniform(0.0, area)
                if any(math.hypot(x - px, y - py) <= comm_range * 0.85 for px, py in coords):
                    coords.append((x, y))
                    break
            else:
                coords.append((rng.uniform(10.0, area - 10.0), rng.uniform(10.0, area - 10.0)))

        nbrs = [[] for _ in range(n_nodes + 1)]
        for u in range(n_nodes + 1):
            for v in range(u + 1, n_nodes + 1):
                if math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1]) <= comm_range:
                    nbrs[u].append(v)
                    nbrs[v].append(u)

        deg_samples.extend([len(nbrs[u]) for u in range(1, n_nodes + 1)])

        # Unscaled Discrete-Event Slot-by-Slot 6TiSCH-MC Simulation (Actual)
        joined = [False] * (n_nodes + 1)
        joined[0] = True
        joined_list = [0]
        unjoined_set = set(range(1, n_nodes + 1))
        joined_asn = [0] * (n_nodes + 1)
        tx_events = [0] * (n_nodes + 1)
        rx_minimal_slots = [0] * (n_nodes + 1)

        asn = 0
        target_join = int(round(n_nodes * 0.80))
        max_slots = 180000
        while len(joined_list) - 1 < target_join and asn < max_slots:
            c_slot = chs[asn % C]
            tx_nodes = []
            for j_id in joined_list:
                if rng.random() <= (p_tx_eb + p_tx_dio):
                    tx_nodes.append(j_id)
                    tx_events[j_id] += 1
                else:
                    rx_minimal_slots[j_id] += 1

            if tx_nodes:
                tx_set = set(tx_nodes)
                new_joins = []
                for p_id in list(unjoined_set):
                    active_nbrs = [nb for nb in nbrs[p_id] if nb in tx_set]
                    if not active_nbrs:
                        continue
                    r_ch = rng.randint(1, C)
                    if r_ch == c_slot and len(active_nbrs) == 1 and rng.random() > P_L:
                        joined[p_id] = True
                        joined_asn[p_id] = asn
                        new_joins.append(p_id)
                for p_id in new_joins:
                    unjoined_set.remove(p_id)
                    joined_list.append(p_id)

            asn += S

        act_sync_s = asn * slot_dur_s
        ep_act_mJ = sum(
            e_rx_mJ * (joined_asn[i] if joined[i] else asn)
            for i in range(1, n_nodes + 1)
        ) / float(n_nodes)

        ej_act_mJ = sum(
            e_tx_mJ * tx_events[j] + e_rx_mJ * rx_minimal_slots[j]
            for j in joined_list
        ) / float(max(1, len(joined_list)))

        kc_sim = float(np.mean([len(nbrs[u]) / float(C) for u in range(1, n_nodes + 1)]))

        act_t_list.append(act_sync_s / 100.0)
        act_ej_list.append(ej_act_mJ / 100.0)
        act_ep_list.append(ep_act_mJ / 1000.0)
        act_net_list.append((ep_act_mJ + kc_sim * ej_act_mJ) / 1000.0)

    # Deterministic Analytical Markov Chain Evaluation (Equations 2-9)
    m_bar = max(4, int(round(float(np.mean(deg_samples)))))
    k_max = max(2, min(m_bar, int(round(math.sqrt(m_bar) * 2.5))))
    p_eb_val = _solve_eq3_fixed_point(k_max, p_tx_eb, p_tx_dio, P_L, C)
    p_jr_val = max(0.01, min(0.90, k_max * p_eb_val * (1.0 - p_tx_dio) * (1.0 - P_L)))
    p_dio_val = max(0.01, min(0.90, k_max * p_tx_dio * (1.0 - P_L)))

    pred_sync_slots = 1.0 + S * ((1.0 / p_eb_val) + (1.0 / p_jr_val) + (1.0 / p_dio_val))
    pred_sync_s = pred_sync_slots * slot_dur_s
    n_sf = pred_sync_slots / float(S)
    kc_bar = float(m_bar) / float(C)

    ep_pred_mJ = (
        e_rx_mJ * ((S / p_eb_val) + m_bar * (p_tx_eb + p_tx_dio) * n_sf)
        + (e_tx_mJ / p_eb_val + e_rx_mJ / p_jr_val)
        + (e_rx_mJ / p_dio_val)
    )
    ej_pred_mJ = (
        (e_tx_mJ * (p_tx_eb + p_tx_dio) + e_rx_mJ * kc_bar * p_jr_val) * n_sf
        + (kc_bar * (e_tx_mJ + e_rx_mJ) / p_jr_val)
    )

    pt_m = pred_sync_s / 100.0
    pep_m = ep_pred_mJ / 1000.0
    pej_m = ej_pred_mJ / 100.0
    pnet_m = (ep_pred_mJ + kc_bar * ej_pred_mJ) / 1000.0

    def mean_and_se(arr):
        m_val = float(np.mean(arr))
        se_val = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return m_val, se_val

    at_m, at_se = mean_and_se(act_t_list)
    aep_m, aep_se = mean_and_se(act_ep_list)
    aej_m, aej_se = mean_and_se(act_ej_list)
    anet_m, anet_se = mean_and_se(act_net_list)

    return [
        n_nodes, area,
        pt_m, 0.0, at_m, at_se,
        pep_m, 0.0, aep_m, aep_se,
        pej_m, 0.0, aej_m, aej_se,
        pnet_m, 0.0, anet_m, anet_se
    ]


def generate_markov_figure6_from_simulation():
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        realdata = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        realdata = list(range(1, 10000))

    S = 101
    C = 16
    i_eb = 4
    eb_prob = 0.1
    P_L = 0.2
    P_T = 0.2
    R_T = 16
    I_dio = 0.8
    TxFactor, RxFactor = 17.4, 18.8
    area = 100.0
    nodes_list = list(range(25, 500 + 1, 25))
    sims = 30

    p_tx_eb = eb_prob / float(i_eb)
    num_r = (2 ** R_T) * ((1.0 - P_T) ** R_T) * max(1.0, S / ((2 ** R_T) * I_dio))
    for r in range(R_T):
        num_r += P_T * (2 ** r) * ((1.0 - P_T) ** r) * max(1.0, S / ((2 ** r) * I_dio))
    den_r = P_T + (2 ** R_T) * ((1.0 - P_T) ** R_T) + sum(P_T * (2 ** s_idx) * ((1.0 - P_T) ** s_idx) for s_idx in range(1, R_T))
    p_tx_dio = min(0.12, (1.0 - P_L) * (1.0 - p_tx_eb) * (num_r / den_r) / float(S))

    tasks = [
        (n_nodes, area, realdata, sims, S, C, p_tx_eb, p_tx_dio, P_L, TxFactor, RxFactor)
        for n_nodes in nodes_list
    ]
    with ProcessPoolExecutor() as ex:
        m_rows = list(ex.map(_run_markov_density_point, tasks))

    m_data = np.array(sorted(m_rows, key=lambda r: r[0]), dtype=float)
    sorted_densities = [round(r[0] / r[1], 2) for r in m_data]

    def plot_markov_panel(y_pairs, legends, colors, markers, lstyles, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=300)
        max_y = 0.0
        for (y_mean, y_se), lbl, col, mk, ls in zip(y_pairs, legends, colors, markers, lstyles):
            if "Predicted" in lbl:
                max_y = max(max_y, float(np.max(y_mean)))
                ax.plot(
                    sorted_densities, y_mean,
                    label=lbl, color=col, marker=mk, linestyle=ls,
                    linewidth=2.0, markersize=6.0, markerfacecolor="white",
                    markeredgecolor=col, markeredgewidth=1.5, alpha=0.95
                )
            else:
                max_y = max(max_y, float(np.max(y_mean + y_se)))
                ax.errorbar(
                    sorted_densities, y_mean, yerr=y_se,
                    label=lbl, color=col, marker=mk, linestyle=ls,
                    linewidth=2.0, markersize=6.0, markerfacecolor="white",
                    markeredgecolor=col, markeredgewidth=1.5,
                    capsize=2.0, elinewidth=0.9, alpha=0.95
                )
        ax.set_xlabel(r"Spatial density ($\times 10^{-2}$ motes/m$^2$)", fontsize=12.0)
        ax.set_ylabel(ylabel, fontsize=12.0)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.tick_params(labelsize=11)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylim(0.0, 1.25 * max_y)
        ax.legend(loc="upper left", ncol=1, fontsize=13.5, frameon=True, edgecolor="#333333")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, fname), format="pdf", bbox_inches="tight")
        plt.close(fig)

    plot_markov_panel(
        [(m_data[:, 2], m_data[:, 3]), (m_data[:, 4], m_data[:, 5])],
        ["Predicted time", "Actual time"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average sync time ($\times 10^2$ s)", "markov_densitywise_sync_time.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 14], m_data[:, 15]), (m_data[:, 6], m_data[:, 7]), (m_data[:, 16], m_data[:, 17]), (m_data[:, 8], m_data[:, 9])],
        ["Predicted network energy", "Predicted pledge energy", "Actual network energy", "Actual pledge energy"],
        ["#0044cc", "#1b9e3e", "#e66100", "#d92121"], ["s", "D", "o", "p"], ["-", "-.", "--", ":"],
        r"Average energy (J)", "markov_densitywise_total_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 10], m_data[:, 11]), (m_data[:, 12], m_data[:, 13])],
        ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy ($\times 10^2$ mJ)", "markov_densitywise_joined_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 6], m_data[:, 7]), (m_data[:, 8], m_data[:, 9])],
        ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy (J)", "markov_densitywise_pledge_energy.pdf"
    )


def generate_wavefront_cdf_plot():
    """Plot empirical Cooja joining-time CDFs directly from real multi-seed per-node timestamps."""
    csv_100 = os.path.join(STAT_DIR, "join_times_N100_scheme1.csv")
    csv_500 = os.path.join(STAT_DIR, "join_times_N500_scheme1.csv")
    if not (os.path.exists(csv_100) and os.path.exists(csv_500)):
        return

    emp_100 = np.sort(np.atleast_1d(np.loadtxt(csv_100, delimiter=",")))
    emp_500 = np.sort(np.atleast_1d(np.loadtxt(csv_500, delimiter=",")))
    if len(emp_100) == 0 or len(emp_500) == 0:
        return

    y_100 = np.linspace(0.0, 100.0, len(emp_100))
    y_500 = np.linspace(0.0, 100.0, len(emp_500))
    step_100 = max(1, len(emp_100) // 18)
    step_500 = max(1, len(emp_500) // 18)

    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    ax.plot(emp_100, y_100, color="#0044cc", linestyle="-", linewidth=2.2,
            marker="o", markevery=step_100, markersize=5.8, markerfacecolor="white", markeredgewidth=1.5,
            label=f"DEFINE Cooja CDF ($N=100$, med={np.median(emp_100):.1f}s, $P_{{95}}$={np.percentile(emp_100, 95):.1f}s)")
    ax.plot(emp_500, y_500, color="#d92121", linestyle="--", linewidth=2.2,
            marker="s", markevery=step_500, markersize=5.8, markerfacecolor="white", markeredgewidth=1.5,
            label=f"DEFINE Cooja CDF ($N=500$, med={np.median(emp_500):.1f}s, $P_{{95}}$={np.percentile(emp_500, 95):.1f}s)")

    ax.set_xlabel("Conditional joining time (s)", fontsize=12.0)
    ax.set_ylabel("Cumulative joined-node CDF (%)", fontsize=12.0)
    ax.set_ylim(0.0, 102.0)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "Fig_wavefront_cdf_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "wavefront_cdf_comparison.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_raw_data(prefix):
    """
    Load Cooja rows from Statistics/. Prefer 22-column multi-seed rows generated by
    run_cooja_headless.py over 21-, 19-, or legacy 12-column files.
    """
    candidates = (
        glob.glob(os.path.join(STAT_DIR, f"_{prefix}_cooja_stats_.txt"))
        + glob.glob(os.path.join(STAT_DIR, f"_{prefix}_stats_.txt"))
    )
    best_data = None
    for path in candidates:
        rows_22, rows_21, rows_19, rows_12 = [], [], [], []
        with open(path, "r") as f:
            for line in f:
                parts = [float(p.strip()) for p in line.strip().split(",") if p.strip()]
                if len(parts) >= 22:
                    rows_22.append(parts[:22])
                elif len(parts) >= 21:
                    rows_21.append(parts[:21])
                elif len(parts) >= 19:
                    rows_19.append(parts[:19])
                elif len(parts) >= 12:
                    rows_12.append(parts[:12])
        rows = rows_22 if rows_22 else (rows_21 if rows_21 else (rows_19 if rows_19 else rows_12))
        if not rows:
            continue
        data = np.array(rows, dtype=float)
        non_zero = data[data[:, 5] > 0]
        arr = non_zero if len(non_zero) > 0 else data
        if (
            best_data is None
            or (arr.shape[1] > best_data.shape[1])
            or (arr.shape[1] == best_data.shape[1] and len(arr) > len(best_data))
        ):
            best_data = arr
    return best_data


def compute_per_seed_metrics(sub):
    """Compute exact per-seed normalized time, energy, and separated duty-cycle arrays over HORIZON_S = 300 s."""
    n_nodes = np.maximum(1.0, sub[:, 0])
    n_joined_with_root = np.maximum(1.0, sub[:, 5] + 1.0)
    c_frac = np.clip(sub[:, 5] / n_nodes, 0.0, 1.0)
    comp_pct = c_frac * 100.0
    m_join_s = sub[:, 8] * 0.01  # ASN slots (10 ms) -> seconds
    p95_s = sub[:, 7] * 0.01
    m_sync_s = (sub[:, 21] * 0.01) if sub.shape[1] > 21 else m_join_s
    post_sync_s = np.maximum(0.0, m_join_s - m_sync_s)

    # Exact restricted mean synchronization time over the 300-s Cooja horizon (in seconds)
    rmst_300_s = c_frac * m_join_s + (1.0 - c_frac) * HORIZON_S

    tx_ticks = sub[:, 19] if sub.shape[1] > 19 else sub[:, 10]
    tx_mj_per_joined = (tx_ticks * TX_MJ_PER_TICK) / n_joined_with_root
    shared_mj_per_joined = (sub[:, 10] * TX_MJ_PER_TICK) / n_joined_with_root
    scan_j_per_pledge = (sub[:, 11] * RX_J_PER_TICK) / n_nodes

    # Exact 300-s bootstrap + stranding energy (J/node):
    # joined nodes pay scan RX + per-joined-node shared-slot control energy; stranded nodes pay 300s * 56.4mW = 16.92 J
    e300_j_per_node = c_frac * (SCAN_POWER_W * m_join_s + shared_mj_per_joined * 1e-3) + (1.0 - c_frac) * STRAND_J

    t_end_s = sub[:, 20] if sub.shape[1] > 20 else np.maximum(1.0, sub[:, 6])
    slot0_active_s = sub[:, 10] / 32768.0
    total_active_s = (sub[:, 10] + sub[:, 11]) / 32768.0

    # Separate associated-node slot-0 duty from scanning-inclusive formation duty (closes F2)
    slot0_duty_pct = np.minimum(100.0, (slot0_active_s / np.maximum(1.0, n_joined_with_root * np.maximum(1.0, t_end_s))) * 100.0)
    formation_duty_pct = np.minimum(100.0, (total_active_s / np.maximum(1.0, (n_nodes + 1.0) * np.maximum(1.0, t_end_s))) * 100.0)

    return {
        "comp_pct": comp_pct,
        "m_join_s": m_join_s,
        "m_sync_s": m_sync_s,
        "post_sync_s": post_sync_s,
        "p95_s": p95_s,
        "rmst_300_s": rmst_300_s,
        "tx_mj_per_joined": tx_mj_per_joined,
        "scan_j_per_pledge": scan_j_per_pledge,
        "shared_mj_per_joined": shared_mj_per_joined,
        "slot0_duty_pct": slot0_duty_pct,
        "formation_duty_pct": formation_duty_pct,
        "e300_j_per_node": e300_j_per_node,
        "t_end_s": t_end_s,
    }


def get_metric_series_with_se(prefix, x_col, fixed_filters, metric_fn, mode="default"):
    """Evaluate metric_fn strictly per seed before computing multi-seed mean and SE."""
    data = load_raw_data(prefix)
    if data is None:
        return None, None, None
    mask = np.ones(len(data), dtype=bool)
    for col, val in fixed_filters.items():
        mask &= (data[:, col] == val)

    if mode == "densitywise" and np.any(mask & (data[:, 1] == 100)):
        mask &= (data[:, 1] == 100)
    elif mode == "motewise" and np.any(mask & (data[:, 1] != 100)):
        mask &= (data[:, 1] != 100)

    filtered = data[mask]
    if len(filtered) == 0:
        return None, None, None
    by_x = {}
    for row in filtered:
        by_x.setdefault(float(row[x_col]), []).append(row)

    xs, means, ses = [], [], []
    for k in sorted(by_x.keys()):
        arr = np.array(by_x[k], dtype=float)
        vals = np.asarray(metric_fn(arr), dtype=float)
        xs.append(k)
        means.append(float(np.mean(vals)))
        if len(vals) > 1:
            ses.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))))
        else:
            ses.append(0.0)
    return np.array(xs), np.array(means), np.array(ses)


def plot_sweep(
    x_col, fixed_filters, col_idx, xlabel, ylabel, png_name, pdf_name=None,
    x_scale=1.0, y_scale=1.0, is_pct=False, is_rmst=False, is_tx_energy=False,
    is_rx_energy=False, is_total_energy=False, is_duty=False, mode="default"
):
    def metric_fn(arr):
        m = compute_per_seed_metrics(arr)
        if is_pct:
            return m["comp_pct"]
        if is_rmst:
            return m["rmst_300_s"]
        if is_tx_energy:
            return m["tx_mj_per_joined"] * y_scale
        if is_rx_energy:
            return m["scan_j_per_pledge"] * y_scale
        if is_total_energy:
            return m["e300_j_per_node"] * y_scale
        if is_duty:
            return m["formation_duty_pct"]
        return arr[:, col_idx] * y_scale

    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    plotted_any = False
    for key in ORDER:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        x_raw, y, y_se = get_metric_series_with_se(key, x_col, fixed_filters, metric_fn, mode=mode)
        if x_raw is None or len(x_raw) == 0:
            continue
        plotted_any = True
        x = x_raw * x_scale

        ms = 8.5 if marker == "*" else 6.4
        z = 10 if key == "p" else 4
        if np.any(y_se > 0):
            ax.errorbar(
                x, y, yerr=y_se,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                capsize=1.8, elinewidth=0.85, alpha=1.0, zorder=z,
            )
        else:
            ax.plot(
                x, y,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                alpha=1.0, zorder=z,
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=12.0)
    ax.set_ylabel(ylabel, fontsize=12.0)
    ax.tick_params(labelsize=11)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        fig.savefig(os.path.join(out_dir, png_name), dpi=300, bbox_inches="tight")
    if pdf_name:
        fig.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf", bbox_inches="tight")
    plt.close(fig)


def select_benchmark_rows(data, n_target):
    if data is None:
        return np.empty((0, 12))
    base_mask = (data[:, 0] == n_target) & (data[:, 2] == 16) & (data[:, 3] == 61) & (data[:, 4] == 1)
    m = data[base_mask & (data[:, 1] == 100)]
    return m if len(m) > 0 else data[base_mask]


def wilcoxon_two_sided(x, y):
    """Exact (n<=14) or continuity-corrected normal approx two-sided paired Wilcoxon signed-rank p-value (no scipy)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    i = 0
    tie_corr = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_d[order[j + 1]] - abs_d[order[i]]) <= 1e-12:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_corr += t * (t * t - 1)
        i = j + 1
    w_stat = min(float(np.sum(ranks[d > 0])), float(np.sum(ranks[d < 0])))
    if n <= 14 and tie_corr == 0.0:
        max_s = n * (n + 1) // 2
        dp = np.zeros(max_s + 1, dtype=np.int64)
        dp[0] = 1
        for r in range(1, n + 1):
            dp[r:] += dp[:-r].copy()
        tail = int(np.sum(dp[:int(round(w_stat)) + 1]))
        return min(1.0, 2.0 * tail / float(1 << n))
    mean_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_corr) / 24.0
    if var_w <= 0.0:
        return 1.0
    z = max(0.0, (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w))
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def holm_adjust(pvals):
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=float)
    idx = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, orig_i in enumerate(idx):
        val = min(1.0, pvals[orig_i] * (m - rank))
        running_max = max(running_max, val)
        adj[orig_i] = running_max
    return adj


def generate_latex_table_iv_and_stats():
    def ms_str(arr, fmt="%.2f"):
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return f"${fmt % m} \\pm {fmt % se}$"

    def format_row(prefix, label, n_target, is_64_only=False):
        d16 = load_raw_data(prefix)
        m16 = select_benchmark_rows(d16, n_target)
        if len(m16) == 0:
            return None

        met = compute_per_seed_metrics(m16)
        m_join_100s = met["m_join_s"] / 100.0
        p95_100s = met["p95_s"] / 100.0
        coll_10k = m16[:, 9] * 1e-4
        eb_rate = float(np.mean(m16[:, 13])) if m16.shape[1] > 13 else 1.0

        if is_64_only:
            c16_str = "---"
            c64_str = ms_str(met["comp_pct"], "%.1f")
        else:
            c16_str = ms_str(met["comp_pct"], "%.1f")
            d64 = load_raw_data(f"{prefix}_64nbr")
            m64 = select_benchmark_rows(d64, n_target)
            if len(m64) > 0:
                comp64 = (m64[:, 5] / float(n_target)) * 100.0
                c64_str = ms_str(comp64, "%.1f")
            else:
                c64_str = "---"

        return (
            f"{label} & ${eb_rate:.2f}$ & {ms_str(met['rmst_300_s'], '%.1f')} & {ms_str(m_join_100s)} & "
            f"{ms_str(p95_100s)} & {ms_str(coll_10k)} & {c16_str} & "
            f"{c64_str} & {ms_str(met['shared_mj_per_joined'], '%.1f')} & {ms_str(met['e300_j_per_node'], '%.2f')} \\\\"
        )

    out_lines = [
        "% Auto-generated strictly from multi-seed Cooja logs in Statistics/",
        "% Exact Cooja horizon T_max = 300 s (sim_timeout_ms = 300000); default NBR_CAP = 16 vs. 64",
        "% Columns: Scheme | EB Rate | RMST_300 (s) | Mean m_join (x10^4 sl) | P95 (x10^4 sl) | Collisions (x10^4) | 16-Nbr Comp (%) | 64-Entry Comp (%) | e_shared (mJ/node) | Boot+Strand E_300 (J/node)"
    ]
    csv_lines = [
        "n_nodes,prefix,nbr_cap,seed,eb_rate,rmst_300_s,m_join_s,m_sync_s,post_sync_s,p95_s,"
        "collisions,coll_tree,coll_sib,coll_cross,switches,comp_pct,"
        "tx_mj_per_joined,scan_j_per_pledge,e_shared_mj_per_joined,slot0_duty_pct,formation_duty_pct,e300_j_per_node,t_end_s"
    ]
    report_lines = [f"=== DEFINE Multi-Seed Statistical & Provenance Report (T_max = {HORIZON_S:.0f} s, default NBR_CAP = 16) ==="]

    for n_val in (100, 500):
        out_lines.append(f"% --- N = {n_val} ---")
        report_lines.append(f"\n--- Benchmark N = {n_val} (100m x 100m, C=16, S=61, I_eb=1S, T_max={HORIZON_S:.0f}s) ---")
        for prefix, lbl in TABLE_IV_ROWS_SPEC:
            line = format_row(prefix, lbl, n_val)
            if line:
                out_lines.append(line)
            for cap_val, pfx in ((16, prefix), (64, f"{prefix}_64nbr")):
                rows = select_benchmark_rows(load_raw_data(pfx), n_val)
                if len(rows) == 0:
                    continue
                met = compute_per_seed_metrics(rows)
                for idx in range(len(rows)):
                    seed_id = int(round(rows[idx, 12])) if rows.shape[1] > 12 else (idx + 1)
                    eb_r = rows[idx, 13] if rows.shape[1] > 13 else 1.0
                    c_tree = rows[idx, 14] if rows.shape[1] > 14 else 0.0
                    c_sib = rows[idx, 15] if rows.shape[1] > 15 else 0.0
                    c_cross = rows[idx, 16] if rows.shape[1] > 16 else 0.0
                    sw = rows[idx, 17] if rows.shape[1] > 17 else 0.0
                    csv_lines.append(
                        f"{n_val},{prefix},{cap_val},{seed_id},{eb_r:.4f},"
                        f"{met['rmst_300_s'][idx]:.2f},{met['m_join_s'][idx]:.3f},"
                        f"{met['m_sync_s'][idx]:.3f},{met['post_sync_s'][idx]:.3f},{met['p95_s'][idx]:.3f},"
                        f"{rows[idx, 9]:.1f},{c_tree:.1f},{c_sib:.1f},{c_cross:.1f},{sw:.4f},{met['comp_pct'][idx]:.2f},"
                        f"{met['tx_mj_per_joined'][idx]:.3f},{met['scan_j_per_pledge'][idx]:.3f},"
                        f"{met['shared_mj_per_joined'][idx]:.3f},{met['slot0_duty_pct'][idx]:.4f},"
                        f"{met['formation_duty_pct'][idx]:.3f},{met['e300_j_per_node'][idx]:.3f},{met['t_end_s'][idx]:.2f}"
                    )
                if cap_val == 16:
                    c_tree_m = float(np.mean(rows[:, 14])) if rows.shape[1] > 14 else 0.0
                    c_sib_m = float(np.mean(rows[:, 15])) if rows.shape[1] > 15 else 0.0
                    c_cross_m = float(np.mean(rows[:, 16])) if rows.shape[1] > 16 else 0.0
                    sw_m = float(np.mean(rows[:, 17])) if rows.shape[1] > 17 else 0.0
                    report_lines.append(
                        f"  {prefix:18s} (n={len(rows):2d}): comp={np.mean(met['comp_pct']):5.1f}% | "
                        f"RMST300={np.mean(met['rmst_300_s']):5.1f}s | m_join={np.mean(met['m_join_s']):5.1f}s "
                        f"(sync={np.mean(met['m_sync_s']):5.1f}s+rpl={np.mean(met['post_sync_s']):5.1f}s) | "
                        f"coll(T/S/X)={c_tree_m:.0f}/{c_sib_m:.0f}/{c_cross_m:.0f} | sw={sw_m:.2f} | "
                        f"slot0_DC={np.mean(met['slot0_duty_pct']):.3f}% | form_DC={np.mean(met['formation_duty_pct']):5.2f}% | "
                        f"e_sh={np.mean(met['shared_mj_per_joined']):6.1f}mJ | E300={np.mean(met['e300_j_per_node']):5.2f}J"
                    )

        if n_val == 500:
            line64 = format_row("p_64nbr", "\\textbf{\\texttt{DEFINE} (Full, 64-Entry Table)}", 500, is_64_only=True)
            if line64:
                out_lines.append(line64)

        # Paired two-sided Wilcoxon + Holm step-down across all 12 baseline rows
        p_rows = select_benchmark_rows(load_raw_data("p"), n_val)
        if len(p_rows) > 0 and p_rows.shape[1] > 12:
            p_met = compute_per_seed_metrics(p_rows)
            p_by_seed = {
                int(round(p_rows[i, 12])): (p_met["comp_pct"][i], p_met["rmst_300_s"][i], p_met["m_join_s"][i])
                for i in range(len(p_rows))
            }
            for metric_idx, m_name in enumerate(("DODAG Completion (%)", "RMST_300 (s)", "Cond. Mean Join Time (s)")):
                tested_names, raw_p = [], []
                for b_pfx, b_lbl in TABLE_IV_BASELINES:
                    b_rows = select_benchmark_rows(load_raw_data(b_pfx), n_val)
                    if len(b_rows) == 0 or b_rows.shape[1] <= 12:
                        continue
                    b_met = compute_per_seed_metrics(b_rows)
                    b_by_seed = {
                        int(round(b_rows[i, 12])): (b_met["comp_pct"][i], b_met["rmst_300_s"][i], b_met["m_join_s"][i])
                        for i in range(len(b_rows))
                    }
                    common = sorted(set(p_by_seed.keys()) & set(b_by_seed.keys()))
                    if len(common) >= 3:
                        x_arr = [p_by_seed[s][metric_idx] for s in common]
                        y_arr = [b_by_seed[s][metric_idx] for s in common]
                        tested_names.append(f"{b_lbl} (n={len(common)})")
                        raw_p.append(wilcoxon_two_sided(x_arr, y_arr))
                if raw_p:
                    adj_p = holm_adjust(raw_p)
                    report_lines.append(f"  [Paired Wilcoxon + Holm ({len(raw_p)} baselines) — {m_name}]: max p_Holm = {np.max(adj_p):.6g}")
                    for nm, rp, ap in zip(tested_names, raw_p, adj_p):
                        report_lines.append(f"    vs {nm:28s}: p_raw={rp:.6g}, p_Holm={ap:.6g}")

    with open(os.path.join(RESULTS_DIR, "table_iv_real.tex"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "table_iv_per_seed.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "stats_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    save_standalone_legend()
    generate_markov_figure6_from_simulation()
    generate_wavefront_cdf_plot()
    generate_latex_table_iv_and_stats()

    rmst_lbl = r"$\mathrm{RMST}_{300}$ (s)"
    mjoin_lbl = r"Cond. mean join time ($\times 10^4$ slots)"
    coll_lbl = r"Collisions ($\times 10^4$)"
    comp_lbl = r"DODAG completion ($\%$)"

    f_nodes = {2: 16, 3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6, "Total pledges", rmst_lbl,  "Fig9a_total_sync_time.png", "motewise_total_time.pdf", is_rmst=True, mode="motewise")
    plot_sweep(0, f_nodes, 8, "Total pledges", mjoin_lbl, "Fig9b_avg_sync_time.png",   "motewise_average_neighbour_time.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 9, "Total pledges", coll_lbl,  "Fig9c_collisions.png",      "motewise_average_collisions.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 5, "Total pledges", comp_lbl,  "Fig9d_dodag_completion.png","motewise_dodag_completion.pdf", is_pct=True, mode="motewise")

    dens_lbl = r"Spatial density ($\times 10^{-2}$ motes/m$^2$)"
    plot_sweep(0, f_nodes, 6, dens_lbl, rmst_lbl,  "Fig10a_density_total_time.png",  "densitywise_total_time.pdf", x_scale=0.01, is_rmst=True, mode="densitywise")
    plot_sweep(0, f_nodes, 8, dens_lbl, mjoin_lbl, "Fig10b_density_avg_time.png",    "densitywise_average_neighbour_time.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 9, dens_lbl, coll_lbl,  "Fig10c_density_collisions.png",  "densitywise_average_collisions.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 5, dens_lbl, comp_lbl,  "Fig10d_density_completion.png",  "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True, mode="densitywise")

    plot_sweep(0, f_nodes, 19, dens_lbl, r"Tx radio energy (mJ/node)",               "Fig11a_tx_energy.png",    "densitywise_transmission_energy.pdf", x_scale=0.01, is_tx_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 11, dens_lbl, r"Rx scan energy (J/node)",                 "Fig11b_rx_energy.png",    "densitywise_reception_energy.pdf", x_scale=0.01, is_rx_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Total formation energy $E_{300}$ (J/node)", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, is_total_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Formation duty cycle ($\%$)",             "Fig11d_duty_cycle.png",   "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True, mode="densitywise")

    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", rmst_lbl,  "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf", is_rmst=True)
    plot_sweep(2, f_ch, 8, "Total channels", mjoin_lbl, "Fig12b_ch_avg_sync_time.png",   "channelwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 9, "Total channels", coll_lbl,  "Fig12c_ch_collisions.png",      "channelwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 5, "Total channels", comp_lbl,  "Fig12d_ch_dodag_completion.png","channelwise_dodag_completion.pdf", is_pct=True)

    f_sf = {0: 200, 2: 16, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size (slots)", rmst_lbl,  "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf", is_rmst=True)
    plot_sweep(3, f_sf, 8, "Slotframe size (slots)", mjoin_lbl, "Fig13b_sf_avg_sync_time.png",   "slotframewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 9, "Slotframe size (slots)", coll_lbl,  "Fig13c_sf_collisions.png",      "slotframewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 5, "Slotframe size (slots)", comp_lbl,  "Fig13d_sf_dodag_completion.png","slotframewise_dodag_completion.pdf", is_pct=True)

    f_eb = {0: 200, 2: 16, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval multiplier", rmst_lbl,  "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf", is_rmst=True)
    plot_sweep(4, f_eb, 8, "EB interval multiplier", mjoin_lbl, "Fig14b_eb_avg_sync_time.png",   "eb_intervalwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 9, "EB interval multiplier", coll_lbl,  "Fig14c_eb_collisions.png",      "eb_intervalwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 5, "EB interval multiplier", comp_lbl,  "Fig14d_eb_dodag_completion.png","eb_intervalwise_dodag_completion.pdf", is_pct=True)