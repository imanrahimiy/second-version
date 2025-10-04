
# ================================================================
# baseline_cplex_saa_edited.py
# CPLEX/DOcplex time-indexed MILP + Monte Carlo SAA + OOS evaluation
# FIXED: Out-of-sample (OOS) NPV now subtracts mining cost for apples-to-apples bias.
# Stable output path next to this script.
# ================================================================

import os, math, time
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from docplex.mp.model import Model
except Exception as e:
    raise RuntimeError("DOcplex is required. Install with: pip install docplex (and ensure CPLEX is available)") from e

# -------------------------
# Configuration
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = dict(
    # problem size
    num_blocks=500,
    num_periods=6,
    # Monte Carlo / SAA
    S_in_list=[10, 30, 50],   # sensitivity
    R=5,                      # replications per S_in
    S_out=500,                # out-of-sample scenarios for evaluation only
    # economics & plant
    discount_rate=0.08,
    gold_price=1190.0,            # USD/oz
    mining_cost_per_ton=2.5,      # USD/t
    plant_avail_hours_per_period=7000.0,
    mine_capacity_per_period_tons=2.0e7,
    # operating modes
    modes={
        "A": {"rate": 250.0, "cost": 21.4, "recovery": 0.83, "diorite_frac": 0.65, "silicified_frac": 0.35},
        "B": {"rate": 200.0, "cost": 24.9, "recovery": 0.83, "diorite_frac": 0.45, "silicified_frac": 0.55},
    },
    # uncertainty
    grade_sigma=0.18,         # lognormal sigma for grade
    price_sigma=0.00,         # set >0.0 to jitter price per scenario
    # solver
    mip_gap=0.02,
    time_limit=3600,
    # reproducibility
    seed=42,
    # output dir (next to script)
    outdir=os.path.join(BASE_DIR, "baseline_outputs")
)
os.makedirs(CFG["outdir"], exist_ok=True)

# -------------------------
# Synthetic data generator
# -------------------------
def generate_blocks(num_blocks: int, seed: int):
    """
    Two-lithology synthetic deposit: diorite(0) vs silicified(1).
    Returns: df (mass, rock), base_grade vector, predecessors list[list[int]]
    """
    rng = np.random.default_rng(seed)
    masses = np.full(num_blocks, 15000.0)  # tons per block
    rock_type = rng.integers(0, 2, size=num_blocks)  # 0/1
    base_grade = np.where(
        rock_type == 0,
        rng.normal(1.5, 0.3, num_blocks),   # g/t diorite
        rng.normal(1.2, 0.25, num_blocks)   # g/t silicified
    )
    # simple layered precedence: dependencies on previous indices
    preds = [[] for _ in range(num_blocks)]
    for b in range(num_blocks):
        if b - 10 >= 0:
            preds[b].append(b - 10)
        if b - 11 >= 0:
            preds[b].append(b - 11)
    df = pd.DataFrame({"mass": masses, "rock": rock_type})
    return df, base_grade, preds

# -------------------------
# Scenario / Monte Carlo generators
# -------------------------
def draw_grade_scenarios(base_grade: np.ndarray, S: int, sigma: float, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Lognormal grade shocks applied to base grade (ensure non-negative grades)."""
    shock = rng.lognormal(mean=0.0, sigma=sigma, size=(S, len(base_grade)))
    arr = np.clip(base_grade * shock, 0.01, None)
    return {s: arr[s, :] for s in range(S)}

def draw_price_scenarios(base_price: float, S: int, sigma: float, rng: np.random.Generator) -> Dict[int, float]:
    """Optional lognormal gold price per scenario. Set sigma=0 to keep fixed."""
    if sigma <= 0:
        return {s: base_price for s in range(S)}
    p = base_price * rng.lognormal(mean=0.0, sigma=sigma, size=S)
    return {s: float(p[s]) for s in range(S)}

# -------------------------
# Core MILP builder (time-indexed)
# -------------------------
def build_milp_timeindexed(
    df_blocks: pd.DataFrame,
    preds: List[List[int]],
    grades_s: Dict[int, np.ndarray],
    price_s: Dict[int, float],
    num_periods: int,
    modes: Dict[str, Dict],
    discount_rate: float,
    mining_cost_per_ton: float,
    plant_avail_hours_per_period: float,
    mine_capacity_per_period_tons: float,
    mip_gap: float,
    time_limit: int,
    log_output: bool = False
):
    """
    Build & solve unified MILP:
      x[b,t] binary mining
      m[b,s,o,t] processing tons
      capacity, precedence, blend, cumulative mine-then-process
    Returns: model, solution, x_solution_dict
    """
    B = range(len(df_blocks))
    T = range(num_periods)
    S = range(len(grades_s))
    modes_list = list(modes.keys())
    M = df_blocks["mass"].to_numpy()
    R = df_blocks["rock"].to_numpy()
    dcf = [math.pow(1.0 + discount_rate, -(t + 1)) for t in T]

    mdl = Model("OPMPS_TimeIndexed")
    mdl.parameters.mip.tolerances.mipgap = mip_gap
    mdl.parameters.timelimit = time_limit

    # Vars
    x = mdl.binary_var_dict(((b, t) for b in B for t in T), name="x")
    m = mdl.continuous_var_dict(((b, s, o, t) for b in B for s in S for o in modes_list for t in T), lb=0, name="m")
    Mtot = mdl.continuous_var_dict(((s, o, t) for s in S for o in modes_list for t in T), lb=0, name="Mtot")

    # Constraints
    for b in B:
        mdl.add_constraint(mdl.sum(x[b, t] for t in T) <= 1, ctname=f"mine_once_{b}")

    for t in T:
        mdl.add_constraint(mdl.sum(M[b] * x[b, t] for b in B) <= mine_capacity_per_period_tons, ctname=f"cap_mine_{t}")

    for b in B:
        for p in preds[b]:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(x[p, tt] for tt in range(t + 1)) >= mdl.sum(x[b, tt] for tt in range(t + 1)),
                    ctname=f"prec_{p}_{b}_{t}"
                )

    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(m[b, s, o, t] / modes[o]["rate"] for b in B for o in modes_list) <= plant_avail_hours_per_period,
                ctname=f"cap_proc_{s}_{t}"
            )

    for b in B:
        Mb = M[b]
        for s in S:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(m[b, s, o, tt] for o in modes_list for tt in range(t + 1))
                    <= Mb * mdl.sum(x[b, tt] for tt in range(t + 1)),
                    ctname=f"mine_then_process_{b}_{s}_{t}"
                )

    for s in S:
        for o in modes_list:
            for t in T:
                mdl.add_constraint(Mtot[s, o, t] == mdl.sum(m[b, s, o, t] for b in B), ctname=f"def_Mtot_{s}_{o}_{t}")
                dio_frac = modes[o]["diorite_frac"]
                sil_frac = modes[o]["silicified_frac"]
                dio_mass = mdl.sum(m[b, s, o, t] for b in B if R[b] == 0)
                sil_mass = mdl.sum(m[b, s, o, t] for b in B if R[b] == 1)
                mdl.add_constraint(dio_mass == dio_frac * Mtot[s, o, t], ctname=f"blend_dio_{s}_{o}_{t}")
                mdl.add_constraint(sil_mass == sil_frac * Mtot[s, o, t], ctname=f"blend_sil_{s}_{o}_{t}")

    mining_cost = mdl.sum(M[b] * x[b, t] * dcf[t] * mining_cost_per_ton for b in B for t in T)

    proc_val_terms = []
    for s in S:
        p = price_s[s]
        for o in modes_list:
            rec = modes[o]["recovery"]
            pc = modes[o]["cost"]
            for b in B:
                oz_per_ton = grades_s[s][b] / 31.1035  # g/t -> oz/t
                rev_per_ton = oz_per_ton * p * rec
                net_per_ton = rev_per_ton - pc
                for t in T:
                    proc_val_terms.append(net_per_ton * m[b, s, o, t] * dcf[t])

    processing_value = (1.0 / len(S)) * mdl.sum(proc_val_terms)
    mdl.maximize(processing_value - mining_cost)

    sol = mdl.solve(log_output=log_output)
    x_star = {(b, t): (sol.get_value(x[b, t]) if sol else 0.0) for b in B for t in T}
    return mdl, sol, x_star

# -------------------------
# Mining cost for fixed schedule (used in OOS)
# -------------------------
def mining_cost_from_x(df_blocks, x_star, discount_rate, mining_cost_per_ton, num_periods):
    from math import pow
    M = df_blocks["mass"].to_numpy()
    dcf = [pow(1.0 + discount_rate, -(t+1)) for t in range(num_periods)]
    total = 0.0
    for b in range(len(M)):
        for t in range(num_periods):
            total += M[b] * x_star[b, t] * dcf[t] * mining_cost_per_ton
    return total

# -------------------------
# Processing-only LP for OOS evaluation (x fixed)
# -------------------------
def evaluate_oos_processing_lp(
    df_blocks: pd.DataFrame,
    preds: List[List[int]],
    x_star: Dict[Tuple[int, int], float],
    grades_s: Dict[int, np.ndarray],
    price_s: Dict[int, float],
    num_periods: int,
    modes: Dict[str, Dict],
    discount_rate: float,
    plant_avail_hours_per_period: float
) -> float:
    """Fix x to x_star. Solve LP only on m[b,s,o,t] for S_out scenarios. Return average discounted processing value."""
    B = range(len(df_blocks))
    T = range(num_periods)
    S = range(len(grades_s))
    modes_list = list(modes.keys())
    M = df_blocks["mass"].to_numpy()
    R = df_blocks["rock"].to_numpy()
    dcf = [math.pow(1.0 + discount_rate, -(t + 1)) for t in T]

    mdl = Model("OOS_ProcessingOnly")
    m = mdl.continuous_var_dict(((b, s, o, t) for b in B for s in S for o in modes_list for t in T), lb=0, name="m")
    Mtot = mdl.continuous_var_dict(((s, o, t) for s in S for o in modes_list for t in T), lb=0, name="Mtot")

    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(m[b, s, o, t] / modes[o]["rate"] for b in B for o in modes_list) <= plant_avail_hours_per_period,
                ctname=f"cap_proc_{s}_{t}"
            )

    for b in B:
        Mb = M[b]
        for s in S:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(m[b, s, o, tt] for o in modes_list for tt in range(t + 1))
                    <= Mb * sum(x_star[b, tt] for tt in range(t + 1)),
                    ctname=f"mine_then_process_{b}_{s}_{t}"
                )

    for s in S:
        for o in modes_list:
            for t in T:
                mdl.add_constraint(Mtot[s, o, t] == mdl.sum(m[b, s, o, t] for b in B), ctname=f"def_Mtot_{s}_{o}_{t}")
                dio_frac = modes[o]["diorite_frac"]
                sil_frac = modes[o]["silicified_frac"]
                dio_mass = mdl.sum(m[b, s, o, t] for b in B if R[b] == 0)
                sil_mass = mdl.sum(m[b, s, o, t] for b in B if R[b] == 1)
                mdl.add_constraint(dio_mass == dio_frac * Mtot[s, o, t], ctname=f"blend_dio_{s}_{o}_{t}")
                mdl.add_constraint(sil_mass == sil_frac * Mtot[s, o, t], ctname=f"blend_sil_{s}_{o}_{t}")

    proc_val_terms = []
    for s in S:
        p = price_s[s]
        for o in modes_list:
            rec = modes[o]["recovery"]
            pc = modes[o]["cost"]
            for b in B:
                oz_per_ton = grades_s[s][b] / 31.1035
                rev_per_ton = oz_per_ton * p * rec
                net_per_ton = rev_per_ton - pc
                for t in T:
                    proc_val_terms.append(net_per_ton * m[b, s, o, t] * dcf[t])

    processing_value = (1.0 / len(S)) * mdl.sum(proc_val_terms)
    sol = mdl.solve(log_output=False)
    return float(sol.objective_value) if sol else float("nan")

# -------------------------
# SAA driver + metrics + plots
# -------------------------
def percentile(a: List[float], q: float) -> float:
    return float(np.percentile(np.array(a), q))

def cvar_alpha(losses: List[float], alpha: float) -> float:
    L = sorted(losses, reverse=True)  # worst to best
    k = max(1, int(len(L) * alpha/100.0))
    return float(np.mean(L[:k]))

def run_saa_baseline(cfg: Dict):
    rng = np.random.default_rng(cfg["seed"])
    df_blocks, base_grade, preds = generate_blocks(cfg["num_blocks"], cfg["seed"])
    rows = []

    for S_in in cfg["S_in_list"]:
        for r in range(cfg["R"]):
            grades_in = draw_grade_scenarios(base_grade, S_in, cfg["grade_sigma"], rng)
            price_in  = draw_price_scenarios(cfg["gold_price"], S_in, cfg["price_sigma"], rng)

            t0 = time.time()
            mdl, sol, x_star = build_milp_timeindexed(
                df_blocks=df_blocks,
                preds=preds,
                grades_s=grades_in,
                price_s=price_in,
                num_periods=cfg["num_periods"],
                modes=cfg["modes"],
                discount_rate=cfg["discount_rate"],
                mining_cost_per_ton=cfg["mining_cost_per_ton"],
                plant_avail_hours_per_period=cfg["plant_avail_hours_per_period"],
                mine_capacity_per_period_tons=cfg["mine_capacity_per_period_tons"],
                mip_gap=cfg["mip_gap"],
                time_limit=cfg["time_limit"],
                log_output=False
            )
            t1 = time.time()
            npv_in = sol.objective_value if sol else float("nan")
            gap = getattr(sol.solve_details, "mip_relative_gap", float("nan")) if sol else float("nan")
            runtime = t1 - t0

            grades_out = draw_grade_scenarios(base_grade, cfg["S_out"], cfg["grade_sigma"], rng)
            price_out  = draw_price_scenarios(cfg["gold_price"], cfg["S_out"], cfg["price_sigma"], rng)

            npv_proc_out = evaluate_oos_processing_lp(
                df_blocks=df_blocks, preds=preds, x_star=x_star,
                grades_s=grades_out, price_s=price_out,
                num_periods=cfg["num_periods"], modes=cfg["modes"],
                discount_rate=cfg["discount_rate"],
                plant_avail_hours_per_period=cfg["plant_avail_hours_per_period"]
            )
            mining_cost_x = mining_cost_from_x(
                df_blocks, x_star,
                cfg["discount_rate"], cfg["mining_cost_per_ton"], cfg["num_periods"]
            )
            npv_out = npv_proc_out - mining_cost_x

            rows.append(dict(
                S_in=S_in, rep=r, npv_in=npv_in, npv_out=npv_out,
                bias=npv_in - npv_out, runtime_sec=runtime, gap=gap
            ))
            print(f"[S_in={S_in} rep={r}] NPV_in={npv_in:,.2f}  NPV_out={npv_out:,.2f}  "
                  f"bias={npv_in-npv_out:,.2f}  time={runtime:.1f}s  gap={gap}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg["outdir"], "baseline_metrics.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", os.path.abspath(out_csv))
    return df

def make_plots(df: pd.DataFrame, cfg: Dict):
    agg = df.groupby("S_in").agg(
        m_out=("npv_out", "mean"),
        s_out=("npv_out", "std"),
        m_bias=("bias", "mean"),
        s_bias=("bias", "std"),
        m_time=("runtime_sec", "mean")
    ).reset_index()
    z = 1.96
    ci_low = agg["m_out"] - z * agg["s_out"] / np.sqrt(cfg["R"])
    ci_high = agg["m_out"] + z * agg["s_out"] / np.sqrt(cfg["R"])

    plt.figure()
    plt.fill_between(agg["S_in"], ci_low, ci_high, alpha=0.2, label="95% CI (NPV_out)")
    plt.plot(agg["S_in"], agg["m_out"], marker="o", label="Mean NPV_out")
    plt.xlabel("In-sample scenarios (S_in)")
    plt.ylabel("Out-of-sample NPV")
    plt.title("SAA Stability (CPLEX Baseline)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(cfg["outdir"], "fig_saa_stability.png")
    plt.savefig(p1, dpi=180)
    print("Wrote:", os.path.abspath(p1))

    plt.figure()
    plt.plot(agg["S_in"], agg["m_bias"], marker="o")
    plt.xlabel("In-sample scenarios (S_in)")
    plt.ylabel("Bias = NPV_in - NPV_out")
    plt.title("In-sample Optimism (Bias) vs S_in")
    plt.tight_layout()
    p2 = os.path.join(cfg["outdir"], "fig_bias_vs_sin.png")
    plt.savefig(p2, dpi=180)
    print("Wrote:", os.path.abspath(p2))

    plt.figure()
    plt.plot(agg["S_in"], agg["m_time"], marker="o")
    plt.xlabel("In-sample scenarios (S_in)")
    plt.ylabel("Solve time (s)")
    plt.title("Runtime scaling (CPLEX Baseline)")
    plt.tight_layout()
    p3 = os.path.join(cfg["outdir"], "fig_runtime_vs_sin.png")
    plt.savefig(p3, dpi=180)
    print("Wrote:", os.path.abspath(p3))

    plt.figure()
    plt.hist(df["npv_out"].dropna().values, bins=30, alpha=0.8)
    plt.xlabel("Out-of-sample NPV")
    plt.ylabel("Frequency")
    plt.title("NPV_out Distribution (All Runs)")
    plt.tight_layout()
    p4 = os.path.join(cfg["outdir"], "fig_npv_out_hist.png")
    plt.savefig(p4, dpi=180)
    print("Wrote:", os.path.abspath(p4))

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    df_metrics = run_saa_baseline(CFG)

    npvs = df_metrics["npv_out"].dropna().tolist()
    if len(npvs) >= 5:
        p10 = float(np.percentile(npvs, 10))
        p50 = float(np.percentile(npvs, 50))
        p90 = float(np.percentile(npvs, 90))
        losses = [p50 - v for v in npvs]
        losses_sorted = sorted(losses, reverse=True)
        k = max(1, int(len(losses_sorted) * 0.10))
        cvar10 = float(np.mean(losses_sorted[:k]))
        out_csv = os.path.join(CFG["outdir"], "baseline_risk_summary.csv")
        pd.DataFrame([dict(P10=p10, P50=p50, P90=p90, CVaR10=cvar10)]).to_csv(out_csv, index=False)
        print("Wrote:", os.path.abspath(out_csv))

    make_plots(df_metrics, CFG)
    print("Artifacts written to:", os.path.abspath(CFG["outdir"]))
