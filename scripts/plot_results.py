import os
import glob
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
TX_MJ_PER_TICK = 3.0 * 17.4 / 32768.0      # mJ per 32,768-Hz TX-equivalent tick
RX_J_PER_TICK  = 3.0 * 0.0188 / 32768.0    # J  per 32,768-Hz RX tick
HORIZON_S      = 300.0                     # Exact Cooja simulation horizon (matches sim_timeout_ms = 300000)
SCAN_POWER_W   = 3.0 * 0.0188              # 0.0564 W continuous RX scan power
STRAND_J       = SCAN_POWER_W * HORIZON_S  # 16.92 J continuous-RX stranding upper bound over 300 s

# Only these three schemes were evaluated with a full 30-seed NBR_CAP=64 campaign
VERIFIED_64NBR_PREFIXES = {"p", "s", "ebmc_2s"}

MECHANISMS = {
    "p":   ("DEFINE",     "#0044cc", "o", "-",                   2.5,  "#0044cc"),
    "e":   ("6TiSCH-MC",  "#e66100", "s", "--",                  1.85, "white"),
    "r":   ("EDBA-SA",    "#d92121", "^", "-.",                  1.85, "white"),
    "d":   ("ECV-SA",     "#1b9e3e", "v", ":",                   1.95, "white"),
    "a_2": ("ATP: 2",     "#4d4d4d", "D", (0, (5, 2)),           1.85, "white"),
    "t":   ("TRGB",       "#7b3294", "P", (0, (3, 1.5, 1, 1.5)), 1.85, "white"),
    "c":   ("CSMJ",       "#8c510a", "X", (0, (4, 1.5)),         1.85, "white"),
    "s":   ("EASE",       "#c51b7d", "*", (0, (2, 1)),           1.85, "white"),
}

ORDER = ["e", "c", "a_2", "r", "d", "t", "s", "p"]

# Exact 14 baseline rows in Table IV (1-to-1 alignment for paired Wilcoxon + Holm step-down)
TABLE_IV_BASELINES = [
    ("e",       "6TiSCH-MC (1S)"),
    ("mc_4s",   "6TiSCH-MC (Tuned 4S)"),
    ("mc_26s",  "6TiSCH-MC (Default 26S)"),
    ("c",       "CSMJ"),
    ("a_2",     "ATP: 2"),
    ("r",       "EDBA-SA"),
    ("d",       "ECV-SA"),
    ("pair",    "Pairing (1S)"),
    ("pair_2s", "Pairing (Tuned 2S)"),
    ("t",       "TRGB (1S)"),
    ("trgb_2s", "TRGB (Tuned 2S)"),
    ("ebmc",    "EBMC (1S)"),
    ("ebmc_2s", "EBMC (Tuned 2S)"),
    ("s",       "EASE"),
]

TABLE_IV_LATEX_SPEC = [
    ("e",       "6TiSCH-MC ($I_{eb}=1S$) \\cite{vilajosana2017minimal}"),
    ("mc_4s",   "6TiSCH-MC (Tuned $I_{eb}=4S$)"),
    ("mc_26s",  "6TiSCH-MC (Default $I_{eb}=26S$)"),
    ("c",       "CSMJ \\cite{chang2024csmj}"),
    ("a_2",     "ATP: 2 \\cite{karalis2018atp}"),
    ("r",       "EDBA-SA \\cite{khoufi2018enhanced}"),
    ("d",       "ECV-SA \\cite{vogli2018fast}"),
    ("pair",    "Pairing ($I_{eb}=1S$) \\cite{pairing2024icufn}"),
    ("pair_2s", "Pairing (Tuned $I_{eb}=2S$)"),
    ("t",       "TRGB ($I_{eb}=1S$) \\cite{kalita2023trgb}"),
    ("trgb_2s", "TRGB (Tuned $I_{eb}=2S$)"),
    ("ebmc",    "EBMC ($I_{eb}=1S$) \\cite{park2025ebmc}"),
    ("ebmc_2s", "EBMC (Tuned $I_{eb}=2S$)"),
    ("s",       "EASE \\cite{kalita2025ease}"),
    ("p",       "\\textbf{\\texttt{DEFINE} (Full)}"),
]

TABLE_V_LATEX_SPEC = [
    ("ease_adapt",         "EASE + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
    ("ebmc_adapt",         "EBMC + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
    ("pair_adapt",         "Pairing + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
    ("idhash_guard_adapt", "ID-Hash + Guard + \\texttt{ADAPT} (Asc.\\ Sweep)"),
    ("idhash_rev",         "ID-Hash + Guard + \\texttt{ADAPT} (Rev.\\ Sweep)"),
    ("idhash_nonzero",     "ID-Hash Non-Zero + Guard + \\texttt{ADAPT}"),
    ("idparity_assign",    "Node-ID Parity + In-Band Hash + Guard + \\texttt{ADAPT}"),
    ("define_rev",         "\\texttt{DEFINE} (Reversed Sweep Order)"),
    ("define_rand",        "\\texttt{DEFINE} (Random Sweep Order)"),
    ("define_tierstart",   "\\texttt{DEFINE} (Tier-Aware Sweep Start)"),
    ("define_nosweep",     "\\texttt{DEFINE} (No Candidate Sweep)"),
    ("assign_hyst",        "\\texttt{ASSIGN}-Only (No Guard, Hyst./Ref., $1S$)"),
    ("assign_guard",       "\\texttt{ASSIGN}-Only (With Guard, Hyst./Ref., $1S$)"),
    ("adapt_age",          "\\texttt{ADAPT} (Age Only, Cell 0)"),
    ("adapt_dens",         "\\texttt{ADAPT} (Density Only, Cell 0)"),
    ("adapt_nophase",      "\\texttt{ADAPT} (No Phase $\\phi_u=0$, Cell 0)"),
    ("adapt_full",         "\\texttt{ADAPT}-Only (Full, Cell 0)"),
    ("define_noka",        "\\texttt{DEFINE} (No Keepalive)"),
    ("ebmc_mrm",           "EBMC ($I_{eb}=1S$, Lossy UDGM $85\\%$ PRR)"),
    ("ease_mrm",           "EASE (Lossy UDGM $85\\%$ PRR)"),
    ("define_hoptier_mrm", "\\texttt{DEFINE} (Hop-Count Tier, Lossy $85\\%$ PRR)"),
    ("define_mrm",         "\\texttt{DEFINE} (Lossy UDGM $85\\%$ PRR)"),
    ("ebmc_2s_poisson",    "EBMC (Tuned $2S$, Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("ease_poisson",       "EASE (Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("define_poisson",     "\\texttt{DEFINE} (Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
    ("define_late",        "\\texttt{DEFINE} (Late Joiners, $20\\%$ at $t=150\\text{ s}$)"),
    ("define_fail",        "\\texttt{DEFINE} (10\\% Relay Shutdown at $t=120\\text{ s}$)"),
    ("define_mixed",       "\\texttt{DEFINE} (Mixed $80\\%$ DEFINE / $20\\%$ Stock $26S$)"),
    ("define_dis",         "\\texttt{DEFINE} (Stock RPL DIS Enabled)"),
    ("pair_dis",           "Pairing ($I_{eb}=1S$, Stock RPL DIS Enabled)"),
    ("ease_dis",           "EASE (Stock RPL DIS Enabled)"),
    ("ebmc_2s_dis",        "EBMC (Tuned $2S$, Stock RPL DIS Enabled)"),
]

ALL_49_PREFIXES = (
    [p for p, _ in TABLE_IV_LATEX_SPEC]
    + [p for p, _ in TABLE_V_LATEX_SPEC]
    + ["assign_ref", "assign_both"]
)


def _t_crit_95(n):
    tbl = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
        10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074,
        24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    }
    return tbl.get(n, 1.960)


def save_standalone_legend():
    handles = []
    for key in ["p", "e", "r", "d", "a_2", "t", "c", "s"]:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        ms = 9.5 if marker == "*" else 7.6
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=lw + 0.3,
                markersize=ms,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=1.6,
                label=label,
            )
        )
    fig = plt.figure(figsize=(6.0, 0.72), dpi=600)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        facecolor="white",
        fontsize=13.5,
        handlelength=2.6,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    plt.axis("off")
    for p in [os.path.join(ROOT_DIR, "legend_only.png"), os.path.join(PLOTS_DIR, "legend_only.png")]:
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(RESULTS_DIR, "legend_only.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.03)
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
        fig, ax = plt.subplots(figsize=(4.5, 3.25), dpi=600)
        max_y = 0.0
        for (y_mean, y_se), lbl, col, mk, ls in zip(y_pairs, legends, colors, markers, lstyles):
            if "Pred" in lbl:
                max_y = max(max_y, float(np.max(y_mean)))
                ax.plot(
                    sorted_densities, y_mean,
                    label=lbl, color=col, marker=mk, linestyle=ls,
                    linewidth=2.3, markersize=6.6, markerfacecolor="white",
                    markeredgecolor=col, markeredgewidth=1.6, alpha=0.95
                )
            else:
                max_y = max(max_y, float(np.max(y_mean + y_se)))
                ax.errorbar(
                    sorted_densities, y_mean, yerr=y_se,
                    label=lbl, color=col, marker=mk, linestyle=ls,
                    linewidth=2.3, markersize=6.6, markerfacecolor="white",
                    markeredgecolor=col, markeredgewidth=1.6,
                    capsize=2.2, elinewidth=1.0, alpha=0.95
                )
        ax.set_xlabel(r"Density ($\times 10^{-2}$ motes/m$^2$)", fontsize=19.0, labelpad=3)
        ax.set_ylabel(ylabel, fontsize=19.0, labelpad=3)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=17.0)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylim(0.0, (1.38 if len(legends) > 2 else 1.25) * max_y)
        leg_fs = 14.5 if len(legends) > 2 else 16.5
        ax.legend(
            loc="upper left", ncol=1, fontsize=leg_fs, frameon=True,
            edgecolor="#333333", handlelength=1.8, handletextpad=0.4,
            borderpad=0.25, labelspacing=0.18
        )
        fig.tight_layout(pad=0.25)
        fig.savefig(os.path.join(RESULTS_DIR, fname), format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    plot_markov_panel(
        [(m_data[:, 2], m_data[:, 3]), (m_data[:, 4], m_data[:, 5])],
        ["Predicted", "Actual"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Sync time ($\times 10^2$ s)", "markov_densitywise_sync_time.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 14], m_data[:, 15]), (m_data[:, 6], m_data[:, 7]), (m_data[:, 16], m_data[:, 17]), (m_data[:, 8], m_data[:, 9])],
        ["Pred. net.", "Pred. pledge", "Act. net.", "Act. pledge"],
        ["#0044cc", "#1b9e3e", "#e66100", "#d92121"], ["s", "D", "o", "p"], ["-", "-.", "--", ":"],
        "Avg. energy (J)", "markov_densitywise_total_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 10], m_data[:, 11]), (m_data[:, 12], m_data[:, 13])],
        ["Predicted", "Actual"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Energy ($\times 10^2$ mJ)", "markov_densitywise_joined_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 6], m_data[:, 7]), (m_data[:, 8], m_data[:, 9])],
        ["Predicted", "Actual"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        "Avg. energy (J)", "markov_densitywise_pledge_energy.pdf"
    )


def _analytic_wavefront_cdf(t_grid, p_scan, sf_s, t_rpl_mean, horizon_s=HORIZON_S):
    """
    Diagnostic convolution CDF of T_join = T_scan + T_rpl, conditioned on T_join <= horizon_s:
      - T_scan ~ Geom(p_scan) * sf_s from Eq. (17) evaluated on logged (c_i, n_w, P_eb^eff)
      - T_rpl  ~ Exp(1 / t_rpl_mean) from logged post-sync RPL join stage
    """
    p_scan = max(1e-4, min(0.95, float(p_scan)))
    max_k = int(math.ceil(horizon_s / sf_s)) + 1
    ks = np.arange(1, max_k + 1, dtype=float)
    pmf = ((1.0 - p_scan) ** (ks - 1.0)) * p_scan
    t_scan = ks * sf_s
    lam_rpl = 1.0 / max(0.5, float(t_rpl_mean))
    dt = np.maximum(0.0, t_grid[:, None] - t_scan[None, :])
    active = (t_grid[:, None] >= t_scan[None, :]).astype(float)
    raw_cdf = np.sum(pmf[None, :] * active * (1.0 - np.exp(-lam_rpl * dt)), axis=1)
    dt_h = np.maximum(0.0, horizon_s - t_scan)
    norm = float(np.sum(pmf * (t_scan <= horizon_s) * (1.0 - np.exp(-lam_rpl * dt_h))))
    return (raw_cdf / max(1e-9, norm)) * 100.0


def load_raw_data(prefix):
    """Load Cooja rows from Statistics/, preferring 28-column multi-seed logs."""
    candidates = (
        glob.glob(os.path.join(STAT_DIR, f"_{prefix}_cooja_stats_.txt"))
        + glob.glob(os.path.join(STAT_DIR, f"_{prefix}_stats_.txt"))
    )
    best_data = None
    for path in candidates:
        rows_28, rows_22, rows_21, rows_19, rows_12 = [], [], [], [], []
        with open(path, "r") as f:
            for line in f:
                parts = [float(p.strip()) for p in line.strip().split(",") if p.strip()]
                if len(parts) >= 28:
                    rows_28.append(parts[:28])
                elif len(parts) >= 22:
                    rows_22.append(parts[:22])
                elif len(parts) >= 21:
                    rows_21.append(parts[:21])
                elif len(parts) >= 19:
                    rows_19.append(parts[:19])
                elif len(parts) >= 12:
                    rows_12.append(parts[:12])
        rows = rows_28 if rows_28 else (rows_22 if rows_22 else (rows_21 if rows_21 else (rows_19 if rows_19 else rows_12)))
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


def select_benchmark_rows(data, n_target):
    if data is None:
        return np.empty((0, 12))
    base_mask = (data[:, 0] == n_target) & (data[:, 2] == 16) & (data[:, 3] == 61) & (data[:, 4] == 1)
    m = data[base_mask & (data[:, 1] == 100)]
    return m if len(m) > 0 else data[base_mask]


def compute_per_seed_metrics(sub):
    """Compute per-seed time, measured Energest energy E_meas, stranding bound E^ub_300, and duty cycles."""
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

    # Actual measured Energest radio energy per node across [0, T_end] (pre-sync scan + associated slot-0 TX/RX)
    e_meas_j_per_node = (
        (n_nodes / (n_nodes + 1.0)) * scan_j_per_pledge
        + (n_joined_with_root / (n_nodes + 1.0)) * (shared_mj_per_joined * 1e-3)
    )

    # Worst-case continuous-RX stranding energy upper bound (J/node)
    e300_j_per_node = c_frac * (SCAN_POWER_W * m_join_s + shared_mj_per_joined * 1e-3) + (1.0 - c_frac) * STRAND_J

    t_end_s = sub[:, 20] if sub.shape[1] > 20 else np.maximum(1.0, sub[:, 6])
    slot0_active_s = sub[:, 10] / 32768.0
    total_active_s = (sub[:, 10] + sub[:, 11]) / 32768.0

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
        "e_meas_j_per_node": e_meas_j_per_node,
        "slot0_duty_pct": slot0_duty_pct,
        "formation_duty_pct": formation_duty_pct,
        "e300_j_per_node": e300_j_per_node,
        "t_end_s": t_end_s,
    }


def generate_wavefront_cdf_plot():
    """Plot empirical Cooja joining-time CDFs vs. Eq. (17) local-scan * measured-t_rpl diagnostic CDFs."""
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
    step_100 = max(1, len(emp_100) // 16)
    step_500 = max(1, len(emp_500) // 16)

    fig, ax = plt.subplots(figsize=(5.0, 3.35), dpi=600)
    ax.plot(
        emp_100, y_100, color="#0044cc", linestyle="-", linewidth=2.3,
        marker="o", markevery=step_100, markersize=6.0, markerfacecolor="white", markeredgewidth=1.6,
        label=f"Cooja $N=100$ (med {np.median(emp_100):.1f}s, $P_{{95}}$ {np.percentile(emp_100, 95):.1f}s)"
    )
    ax.plot(
        emp_500, y_500, color="#d92121", linestyle="-", linewidth=2.3,
        marker="s", markevery=step_500, markersize=6.0, markerfacecolor="white", markeredgewidth=1.6,
        label=f"Cooja $N=500$ (med {np.median(emp_500):.1f}s, $P_{{95}}$ {np.percentile(emp_500, 95):.1f}s)"
    )

    p_data = load_raw_data("p")
    t_grid = np.linspace(0.0, HORIZON_S, 301)
    for n_val, col, ls in ((100, "#0044cc", "--"), (500, "#d92121", ":")):
        rows = select_benchmark_rows(p_data, n_val)
        if len(rows) > 0 and rows.shape[1] > 27:
            met = compute_per_seed_metrics(rows)
            c_bar = float(np.mean(rows[:, 25]))
            n_w = max(1.0, float(np.mean(rows[:, 27])))
            eb_eff = max(0.01, min(0.95, float(np.mean(rows[:, 13]))))
            t_rpl = float(np.mean(met["post_sync_s"]))
            p_scan = (c_bar / 16.0) * n_w * eb_eff * ((1.0 - eb_eff) ** max(0.0, n_w - 1.0))
            if p_scan > 1e-4:
                y_pred = _analytic_wavefront_cdf(t_grid, p_scan, 0.61, t_rpl, HORIZON_S)
                ax.plot(
                    t_grid, y_pred, color=col, linestyle=ls, linewidth=1.8, alpha=0.88,
                    label=f"Eq. (17) diag. $N={n_val}$ ($\\bar{{c}}_i={c_bar:.1f}, n_\\omega={n_w:.1f}$)"
                )

    ax.set_xlabel("Conditional joining time (s)", fontsize=14.0, labelpad=3)
    ax.set_ylabel("Joined-node CDF (%)", fontsize=14.0, labelpad=3)
    ax.set_xlim(0.0, HORIZON_S)
    ax.set_ylim(0.0, 102.0)
    ax.tick_params(labelsize=12.5)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=10.5, frameon=True, edgecolor="#333333", handlelength=2.0, borderpad=0.3, labelspacing=0.22)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(PLOTS_DIR, "Fig_wavefront_cdf_comparison.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(RESULTS_DIR, "wavefront_cdf_comparison.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(4.5, 3.25), dpi=600)
    plotted_any = False
    for key in ORDER:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        x_raw, y, y_se = get_metric_series_with_se(key, x_col, fixed_filters, metric_fn, mode=mode)
        if x_raw is None or len(x_raw) == 0:
            continue
        plotted_any = True
        x = x_raw * x_scale

        ms = 9.0 if marker == "*" else 6.8
        lw_use = lw + 0.35
        z = 10 if key == "p" else 4
        if np.any(y_se > 0):
            ax.errorbar(
                x, y, yerr=y_se,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw_use, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                capsize=1.8, elinewidth=0.85, alpha=1.0, zorder=z,
            )
        else:
            ax.plot(
                x, y,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw_use, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                alpha=1.0, zorder=z,
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=19.0, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=19.0, labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=17.0)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout(pad=0.25)
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        fig.savefig(os.path.join(out_dir, png_name), dpi=300, bbox_inches="tight", pad_inches=0.02)
    if pdf_name:
        fig.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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


def generate_latex_tables_and_stats():
    def ms_str(arr, fmt="%.1f"):
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return f"${fmt % m} \\pm {fmt % se}$"

    def pad_collapsed(arr, target_n, fill_val):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < target_n:
            return np.concatenate([arr, np.full(target_n - len(arr), fill_val, dtype=float)])
        return arr

    t4_lines = [
        "% Auto-generated strictly from 30-seed Cooja logs in Statistics/ (no stale 5-seed 64nbr rows)",
        "% Exact Cooja horizon T_max = 300 s; collapsed zero-join seeds counted as 0% and RMST_300 = 300 s",
        "% Columns: Scheme | EB/SF | RMST_300 (s) | TSCH Sync m_sync (s) | Mean m_join (s) | Cond. P95 (s) | Collisions (x10^4) | Comp. (%) | EB TX e_tx (mJ/node) | Measured E_meas (J/node)"
    ]
    t5_lines = [
        "% Auto-generated strictly from 30-seed Cooja logs in Statistics/",
        "% Columns: Variant | EB/SF (500) | RMST_300 s (500) | m_join s (500) | P95 s (500) | Coll x10^4 (500) | Comp % (500) | E_meas J (500) | Comp % (100) | RMST_300 s (100)"
    ]
    csv_lines = [
        "n_nodes,prefix,nbr_cap,seed,eb_rate,rmst_300_s,m_join_s,m_sync_s,post_sync_s,p95_s,"
        "collisions,coll_tree,coll_sib,coll_cross,switches,comp_pct,"
        "tx_mj_per_joined,scan_j_per_pledge,e_shared_mj_per_joined,e_meas_j_per_node,"
        "slot0_duty_pct,formation_duty_pct,e300_ub_j_per_node,t_end_s,"
        "nbr_p50,nbr_p95,nbr_p99,mean_c_i,mean_tier,mean_co_off"
    ]
    report_lines = [f"=== DEFINE Multi-Seed Statistical & Provenance Report (T_max = {HORIZON_S:.0f} s, default NBR_CAP = 16) ==="]

    for n_val in (100, 500):
        p_ref = select_benchmark_rows(load_raw_data("p"), n_val)
        target_seeds = len(p_ref) if len(p_ref) > 0 else 30

        t4_lines.append(f"% --- N = {n_val} (target_seeds = {target_seeds}) ---")
        report_lines.append(f"\n--- Benchmark N = {n_val} (100m x 100m, C=16, S=61, I_eb=1S, T_max={HORIZON_S:.0f}s, target_n={target_seeds}) ---")

        # Table IV rows (14 baselines + DEFINE, directly in seconds with TSCH Sync m_sync)
        for prefix, lbl in TABLE_IV_LATEX_SPEC:
            rows = select_benchmark_rows(load_raw_data(prefix), n_val)
            if len(rows) == 0:
                continue
            met = compute_per_seed_metrics(rows)
            comp_full = pad_collapsed(met["comp_pct"], target_seeds, 0.0)
            rmst_full = pad_collapsed(met["rmst_300_s"], target_seeds, HORIZON_S)
            eb_rate = float(np.mean(rows[:, 13])) if rows.shape[1] > 13 else 1.0
            t4_lines.append(
                f"{lbl} & ${eb_rate:.2f}$ & {ms_str(rmst_full, '%.1f')} & "
                f"{ms_str(met['m_sync_s'], '%.1f')} & {ms_str(met['m_join_s'], '%.1f')} & "
                f"{ms_str(met['p95_s'], '%.1f')} & {ms_str(rows[:, 9] * 1e-4, '%.2f')} & "
                f"{ms_str(comp_full, '%.1f')} & {ms_str(met['tx_mj_per_joined'], '%.2f')} & "
                f"{ms_str(met['e_meas_j_per_node'], '%.2f')} \\\\"
            )

        # Per-seed CSV and summary report across all 49 configurations (strictly verified 30-seed 64nbr only)
        for prefix in ALL_49_PREFIXES:
            caps_to_export = [(16, prefix)]
            if prefix in VERIFIED_64NBR_PREFIXES:
                caps_to_export.append((64, f"{prefix}_64nbr"))
            for cap_val, pfx in caps_to_export:
                rows = select_benchmark_rows(load_raw_data(pfx), n_val)
                if len(rows) == 0:
                    continue
                if cap_val == 64 and len(rows) < target_seeds:
                    continue
                met = compute_per_seed_metrics(rows)
                for idx in range(len(rows)):
                    seed_id = int(round(rows[idx, 12])) if rows.shape[1] > 12 else (idx + 1)
                    eb_r = rows[idx, 13] if rows.shape[1] > 13 else 1.0
                    c_tree = rows[idx, 14] if rows.shape[1] > 14 else 0.0
                    c_sib = rows[idx, 15] if rows.shape[1] > 15 else 0.0
                    c_cross = rows[idx, 16] if rows.shape[1] > 16 else 0.0
                    sw = rows[idx, 17] if rows.shape[1] > 17 else 0.0
                    n_p50 = rows[idx, 22] if rows.shape[1] > 27 else 0.0
                    n_p95 = rows[idx, 23] if rows.shape[1] > 27 else 0.0
                    n_p99 = rows[idx, 24] if rows.shape[1] > 27 else 0.0
                    c_i_v = rows[idx, 25] if rows.shape[1] > 27 else 0.0
                    h_bar = rows[idx, 26] if rows.shape[1] > 27 else 0.0
                    n_om = rows[idx, 27] if rows.shape[1] > 27 else 0.0
                    csv_lines.append(
                        f"{n_val},{prefix},{cap_val},{seed_id},{eb_r:.4f},"
                        f"{met['rmst_300_s'][idx]:.2f},{met['m_join_s'][idx]:.3f},"
                        f"{met['m_sync_s'][idx]:.3f},{met['post_sync_s'][idx]:.3f},{met['p95_s'][idx]:.3f},"
                        f"{rows[idx, 9]:.1f},{c_tree:.1f},{c_sib:.1f},{c_cross:.1f},{sw:.4f},{met['comp_pct'][idx]:.2f},"
                        f"{met['tx_mj_per_joined'][idx]:.3f},{met['scan_j_per_pledge'][idx]:.3f},"
                        f"{met['shared_mj_per_joined'][idx]:.3f},{met['e_meas_j_per_node'][idx]:.3f},"
                        f"{met['slot0_duty_pct'][idx]:.4f},{met['formation_duty_pct'][idx]:.3f},"
                        f"{met['e300_j_per_node'][idx]:.3f},{met['t_end_s'][idx]:.2f},"
                        f"{n_p50:.2f},{n_p95:.2f},{n_p99:.2f},{c_i_v:.4f},{h_bar:.4f},{n_om:.4f}"
                    )
                if cap_val == 16:
                    comp_full = pad_collapsed(met["comp_pct"], target_seeds, 0.0)
                    rmst_full = pad_collapsed(met["rmst_300_s"], target_seeds, HORIZON_S)
                    c_tree_m = float(np.mean(rows[:, 14])) if rows.shape[1] > 14 else 0.0
                    c_sib_m = float(np.mean(rows[:, 15])) if rows.shape[1] > 15 else 0.0
                    c_cross_m = float(np.mean(rows[:, 16])) if rows.shape[1] > 16 else 0.0
                    sw_m = float(np.mean(rows[:, 17])) if rows.shape[1] > 17 else 0.0
                    occ_str = (
                        f" | nbr(50/95/99)={np.mean(rows[:, 22]):.1f}/{np.mean(rows[:, 23]):.1f}/{np.mean(rows[:, 24]):.1f}"
                        f" | c_i={np.mean(rows[:, 25]):.2f} h={np.mean(rows[:, 26]):.2f} n_w={np.mean(rows[:, 27]):.2f}"
                        if rows.shape[1] > 27 else ""
                    )
                    report_lines.append(
                        f"  {prefix:18s} (n={len(rows):2d}/{target_seeds:2d}): comp={np.mean(comp_full):5.1f}% | "
                        f"RMST300={np.mean(rmst_full):5.1f}s | m_join={np.mean(met['m_join_s']):5.1f}s "
                        f"(sync={np.mean(met['m_sync_s']):5.1f}s+rpl={np.mean(met['post_sync_s']):5.1f}s) | "
                        f"coll(T/S/X)={c_tree_m:.0f}/{c_sib_m:.0f}/{c_cross_m:.0f} | sw={sw_m:.2f} | "
                        f"slot0_DC={np.mean(met['slot0_duty_pct']):.3f}% | form_DC={np.mean(met['formation_duty_pct']):5.2f}% | "
                        f"tx={np.mean(met['tx_mj_per_joined']):5.2f}mJ | E_meas={np.mean(met['e_meas_j_per_node']):5.2f}J | "
                        f"E_ub={np.mean(met['e300_j_per_node']):5.2f}J{occ_str}"
                    )

        # Paired two-sided Wilcoxon + Holm step-down + 95% CIs across all 14 baselines (collapsed seeds included on comp & RMST)
        if len(p_ref) > 0 and p_ref.shape[1] > 12:
            p_met = compute_per_seed_metrics(p_ref)
            p_seeds = sorted(int(round(p_ref[i, 12])) for i in range(len(p_ref)))
            p_by_seed = {
                int(round(p_ref[i, 12])): (
                    p_met["comp_pct"][i],
                    p_met["rmst_300_s"][i],
                    p_ref[i, 9],
                    p_met["tx_mj_per_joined"][i],
                    p_met["e_meas_j_per_node"][i],
                )
                for i in range(len(p_ref))
            }
            metric_defs = [
                (0, "DODAG Completion (%)", True,  0.0),
                (1, "RMST_300 (s)",         True,  HORIZON_S),
                (2, "Collisions",           False, None),
                (3, "EB TX Energy (mJ)",    False, None),
                (4, "Measured E_meas (J)",  False, None),
            ]
            for m_idx, m_name, fill_collapsed, fill_val in metric_defs:
                tested_names, raw_p, diffs, cis = [], [], [], []
                for b_pfx, b_lbl in TABLE_IV_BASELINES:
                    b_rows = select_benchmark_rows(load_raw_data(b_pfx), n_val)
                    if len(b_rows) == 0 or b_rows.shape[1] <= 12:
                        continue
                    b_met = compute_per_seed_metrics(b_rows)
                    b_by_seed = {
                        int(round(b_rows[i, 12])): (
                            b_met["comp_pct"][i],
                            b_met["rmst_300_s"][i],
                            b_rows[i, 9],
                            b_met["tx_mj_per_joined"][i],
                            b_met["e_meas_j_per_node"][i],
                        )
                        for i in range(len(b_rows))
                    }
                    if fill_collapsed:
                        x_arr = np.array([p_by_seed[s][m_idx] for s in p_seeds], dtype=float)
                        y_arr = np.array([b_by_seed[s][m_idx] if s in b_by_seed else fill_val for s in p_seeds], dtype=float)
                        n_eff = len(p_seeds)
                    else:
                        common = sorted(set(p_by_seed.keys()) & set(b_by_seed.keys()))
                        if len(common) < 3:
                            continue
                        x_arr = np.array([p_by_seed[s][m_idx] for s in common], dtype=float)
                        y_arr = np.array([b_by_seed[s][m_idx] for s in common], dtype=float)
                        n_eff = len(common)
                    d_arr = x_arr - y_arr
                    se_d = float(np.std(d_arr, ddof=1) / math.sqrt(n_eff)) if n_eff > 1 else 0.0
                    tested_names.append(f"{b_lbl} (n={n_eff})")
                    raw_p.append(wilcoxon_two_sided(x_arr, y_arr))
                    diffs.append(float(np.mean(d_arr)))
                    cis.append(_t_crit_95(n_eff) * se_d)
                if raw_p:
                    adj_p = holm_adjust(raw_p)
                    report_lines.append(f"  [Paired Wilcoxon + Holm ({len(raw_p)} baselines) — {m_name}]: max p_Holm = {np.max(adj_p):.6g}")
                    for nm, md, ci, rp, ap in zip(tested_names, diffs, cis, raw_p, adj_p):
                        report_lines.append(f"    vs {nm:30s}: diff={md:+9.2f} +/- {ci:7.2f} (95% CI) | p_raw={rp:.6g}, p_Holm={ap:.6g}")

            # Paired tests for DEFINE vs Round-7 attribution controls
            ctrl_lines = []
            for c_pfx, c_lbl in (
                ("idhash_nonzero",  "ID-Hash Non-Zero + Guard + ADAPT"),
                ("idparity_assign", "Node-ID Parity + In-Band Hash + ADAPT"),
                ("pair_adapt",      "Pairing + ADAPT (Grafted)"),
                ("define_nosweep",  "DEFINE (No Candidate Sweep)"),
            ):
                c_rows = select_benchmark_rows(load_raw_data(c_pfx), n_val)
                if len(c_rows) == 0 or c_rows.shape[1] <= 12:
                    continue
                c_met = compute_per_seed_metrics(c_rows)
                c_by_seed = {int(round(c_rows[i, 12])): (c_met["comp_pct"][i], c_met["rmst_300_s"][i]) for i in range(len(c_rows))}
                common = sorted(set(p_by_seed.keys()) & set(c_by_seed.keys()))
                if len(common) >= 3:
                    dc = np.array([p_by_seed[s][0] - c_by_seed[s][0] for s in common])
                    dr = np.array([p_by_seed[s][1] - c_by_seed[s][1] for s in common])
                    pc = wilcoxon_two_sided([p_by_seed[s][0] for s in common], [c_by_seed[s][0] for s in common])
                    pr = wilcoxon_two_sided([p_by_seed[s][1] for s in common], [c_by_seed[s][1] for s in common])
                    ctrl_lines.append(
                        f"    vs {c_lbl:38s} (n={len(common)}): d_comp={dc.mean():+5.2f} pts (p={pc:.4g}) | d_RMST={dr.mean():+6.2f} s (p={pr:.4g})"
                    )
            if ctrl_lines:
                report_lines.append("  [Paired Wilcoxon — DEFINE vs Round-7 Attribution Controls]:")
                report_lines.extend(ctrl_lines)

            # Paired tests + Holm step-down for Stock RPL DIS Sensitivity (3 baselines)
            dd_rows = select_benchmark_rows(load_raw_data("define_dis"), n_val)
            if len(dd_rows) > 0 and dd_rows.shape[1] > 12:
                dd_met = compute_per_seed_metrics(dd_rows)
                dd_by_seed = {int(round(dd_rows[i, 12])): (dd_met["comp_pct"][i], dd_met["rmst_300_s"][i]) for i in range(len(dd_rows))}
                dis_peers, dis_n, dis_dc, dis_dr, dis_pc, dis_pr = [], [], [], [], [], []
                for b_pfx, b_lbl in (("pair_dis", "Pairing 1S (DIS)"), ("ease_dis", "EASE (DIS)"), ("ebmc_2s_dis", "EBMC 2S (DIS)")):
                    b_rows = select_benchmark_rows(load_raw_data(b_pfx), n_val)
                    if len(b_rows) == 0 or b_rows.shape[1] <= 12:
                        continue
                    b_met = compute_per_seed_metrics(b_rows)
                    b_by_seed = {int(round(b_rows[i, 12])): (b_met["comp_pct"][i], b_met["rmst_300_s"][i]) for i in range(len(b_rows))}
                    common = sorted(set(dd_by_seed.keys()) & set(b_by_seed.keys()))
                    if len(common) >= 3:
                        dis_peers.append(b_lbl)
                        dis_n.append(len(common))
                        dis_dc.append(float(np.mean([dd_by_seed[s][0] - b_by_seed[s][0] for s in common])))
                        dis_dr.append(float(np.mean([dd_by_seed[s][1] - b_by_seed[s][1] for s in common])))
                        dis_pc.append(wilcoxon_two_sided([dd_by_seed[s][0] for s in common], [b_by_seed[s][0] for s in common]))
                        dis_pr.append(wilcoxon_two_sided([dd_by_seed[s][1] for s in common], [b_by_seed[s][1] for s in common]))
                if dis_peers:
                    adj_pc = holm_adjust(dis_pc)
                    adj_pr = holm_adjust(dis_pr)
                    report_lines.append("  [Paired Wilcoxon + Holm (3 baselines) — Stock RPL DIS Sensitivity]:")
                    for nm, nc, dc, dr, pc, apc, pr, apr in zip(dis_peers, dis_n, dis_dc, dis_dr, dis_pc, adj_pc, dis_pr, adj_pr):
                        report_lines.append(
                            f"    DEFINE (DIS) vs {nm:18s} (n={nc}): d_comp={dc:+6.2f} pts (p={pc:.4g}, p_Holm={apc:.4g}) | "
                            f"d_RMST={dr:+6.2f} s (p={pr:.4g}, p_Holm={apr:.4g})"
                        )

    # Auto-generate Table V directly in seconds
    target_500 = max(1, len(select_benchmark_rows(load_raw_data("p"), 500)))
    target_100 = max(1, len(select_benchmark_rows(load_raw_data("p"), 100)))
    for pfx, lbl in TABLE_V_LATEX_SPEC:
        r5 = select_benchmark_rows(load_raw_data(pfx), 500)
        r1 = select_benchmark_rows(load_raw_data(pfx), 100)
        if len(r5) == 0 or len(r1) == 0:
            continue
        m5 = compute_per_seed_metrics(r5)
        m1 = compute_per_seed_metrics(r1)
        eb5 = float(np.mean(r5[:, 13])) if r5.shape[1] > 13 else 1.0
        t5_lines.append(
            f"{lbl} & ${eb5:.2f}$ & {ms_str(pad_collapsed(m5['rmst_300_s'], target_500, HORIZON_S), '%.1f')} & "
            f"{ms_str(m5['m_join_s'], '%.1f')} & {ms_str(m5['p95_s'], '%.1f')} & "
            f"{ms_str(r5[:, 9] * 1e-4, '%.2f')} & {ms_str(pad_collapsed(m5['comp_pct'], target_500, 0.0), '%.1f')} & "
            f"{ms_str(m5['e_meas_j_per_node'], '%.2f')} & "
            f"{ms_str(pad_collapsed(m1['comp_pct'], target_100, 0.0), '%.1f')} & "
            f"{ms_str(pad_collapsed(m1['rmst_300_s'], target_100, HORIZON_S), '%.1f')} \\\\"
        )

    # Mixed 80/20 RFC 8180 coexistence check at N=200
    mixed_200 = select_benchmark_rows(load_raw_data("define_mixed"), 200)
    if len(mixed_200) > 0:
        met200 = compute_per_seed_metrics(mixed_200)
        report_lines.append(
            f"\n--- Mixed 80/20 Coexistence Check (N = 200, 100m x 100m, n={len(mixed_200)}) ---\n"
            f"  define_mixed: comp={np.mean(met200['comp_pct']):5.1f}% | RMST300={np.mean(met200['rmst_300_s']):5.1f}s | "
            f"m_join={np.mean(met200['m_join_s']):5.1f}s (sync={np.mean(met200['m_sync_s']):5.1f}s+rpl={np.mean(met200['post_sync_s']):5.1f}s) | "
            f"E_meas={np.mean(met200['e_meas_j_per_node']):5.2f}J"
        )

    with open(os.path.join(RESULTS_DIR, "table_iv_real.tex"), "w") as f:
        f.write("\n".join(t4_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "table_v_real.tex"), "w") as f:
        f.write("\n".join(t5_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "table_iv_per_seed.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "stats_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    save_standalone_legend()
    generate_markov_figure6_from_simulation()
    generate_wavefront_cdf_plot()
    generate_latex_tables_and_stats()

    rmst_lbl  = r"$\mathrm{RMST}_{300}$ (s)"
    mjoin_lbl = r"Cond. join ($\times 10^4$ sl)"
    coll_lbl  = r"Collisions ($\times 10^4$)"
    comp_lbl  = r"Completion ($\%$)"
    dens_lbl  = r"Density ($\times 10^{-2}$ motes/m$^2$)"

    f_nodes = {2: 16, 3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6, "Total pledges", rmst_lbl,  "Fig9a_total_sync_time.png", "motewise_total_time.pdf", is_rmst=True, mode="motewise")
    plot_sweep(0, f_nodes, 8, "Total pledges", mjoin_lbl, "Fig9b_avg_sync_time.png",   "motewise_average_neighbour_time.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 9, "Total pledges", coll_lbl,  "Fig9c_collisions.png",      "motewise_average_collisions.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 5, "Total pledges", comp_lbl,  "Fig9d_dodag_completion.png","motewise_dodag_completion.pdf", is_pct=True, mode="motewise")

    plot_sweep(0, f_nodes, 6, dens_lbl, rmst_lbl,  "Fig10a_density_total_time.png",  "densitywise_total_time.pdf", x_scale=0.01, is_rmst=True, mode="densitywise")
    plot_sweep(0, f_nodes, 8, dens_lbl, mjoin_lbl, "Fig10b_density_avg_time.png",    "densitywise_average_neighbour_time.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 9, dens_lbl, coll_lbl,  "Fig10c_density_collisions.png",  "densitywise_average_collisions.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 5, dens_lbl, comp_lbl,  "Fig10d_density_completion.png",  "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True, mode="densitywise")

    plot_sweep(0, f_nodes, 19, dens_lbl, r"Tx energy (mJ/node)",                  "Fig11a_tx_energy.png",    "densitywise_transmission_energy.pdf", x_scale=0.01, is_tx_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 11, dens_lbl, r"Scan energy (J/node)",                 "Fig11b_rx_energy.png",    "densitywise_reception_energy.pdf", x_scale=0.01, is_rx_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Bound $E^{\mathrm{ub}}_{300}$ (J/node)", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, is_total_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Duty cycle ($\%$)",                    "Fig11d_duty_cycle.png",   "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True, mode="densitywise")

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