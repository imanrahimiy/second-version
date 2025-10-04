
from docplex.mp.model import Model
import numpy as np
import pandas as pd
from math import pow

def generate_blocks(num_blocks=1000, num_periods=6, num_scenarios=10, seed=42):
    rng = np.random.default_rng(seed)
    masses = np.full(num_blocks, 15000.0)  # tons
    rock_type = rng.integers(0, 2, size=num_blocks)  # 0=diorite, 1=silicified
    base_grade = np.where(rock_type==0, rng.normal(1.5, 0.3, num_blocks),
                          rng.normal(1.2, 0.25, num_blocks))
    grades = {s: np.maximum(0.05, base_grade * rng.lognormal(0, 0.15, num_blocks))
              for s in range(num_scenarios)}
    preds = [[] for _ in range(num_blocks)]
    for b in range(num_blocks):
        if b-10 >= 0:
            preds[b].append(b-10)
        if b-11 >= 0:
            preds[b].append(b-11)
    df = pd.DataFrame({"mass": masses, "rock": rock_type})
    return df, grades, preds

def build_and_solve(num_blocks=1000, num_periods=6, num_scenarios=10,
                    discount_rate=0.08, gold_price=1190.0,
                    operational_modes=None,
                    mine_capacity_per_ton=2.0e7,
                    plant_avail_hours=7000.0,
                    mip_gap=0.02, time_limit=3600, seed=42):
    if operational_modes is None:
        operational_modes = {
            "A": {"rate": 250.0, "cost": 21.4, "recovery": 0.83, "diorite_frac": 0.65, "silicified_frac": 0.35},
            "B": {"rate": 200.0, "cost": 24.9, "recovery": 0.83, "diorite_frac": 0.45, "silicified_frac": 0.55},
        }
    modes = list(operational_modes.keys())
    T = range(num_periods)
    S = range(num_scenarios)
    B = range(num_blocks)

    blocks, grades, preds = generate_blocks(num_blocks, num_periods, num_scenarios, seed)
    mass = blocks["mass"].to_numpy()
    rock = blocks["rock"].to_numpy()
    dcf = [pow(1.0 + discount_rate, -(t+1)) for t in T]

    mdl = Model("OPMPS_time_indexed")

    x = mdl.binary_var_dict(((b,t) for b in B for t in T), name="x")
    m = mdl.continuous_var_dict(((b,s,o,t) for b in B for s in S for o in modes for t in T),
                                lb=0, name="m")
    M_tot = mdl.continuous_var_dict(((s,o,t) for s in S for o in modes for t in T),
                                    lb=0, name="Mtot")

    for b in B:
        mdl.add_constraint(mdl.sum(x[b,t] for t in T) <= 1, ctname=f"mine_once_{b}")

    for t in T:
        mdl.add_constraint(mdl.sum(mass[b]*x[b,t] for b in B) <= mine_capacity_per_ton,
                           ctname=f"cap_mine_{t}")

    for b in B:
        for p in preds[b]:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(x[p,tt] for tt in range(t+1)) >= mdl.sum(x[b,tt] for tt in range(t+1)),
                    ctname=f"prec_{p}_{b}_{t}"
                )

    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(m[b,s,o,t]/operational_modes[o]["rate"] for b in B for o in modes)
                <= plant_avail_hours,
                ctname=f"cap_proc_{s}_{t}"
            )

    for b in B:
        Mb = mass[b]
        for s in S:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(m[b,s,o,tt] for o in modes for tt in range(t+1)) <=
                    Mb * mdl.sum(x[b,tt] for tt in range(t+1)),
                    ctname=f"mine_then_process_{b}_{s}_{t}"
                )

    for s in S:
        for o in modes:
            for t in T:
                mdl.add_constraint(
                    M_tot[s,o,t] == mdl.sum(m[b,s,o,t] for b in B),
                    ctname=f"def_Mtot_{s}_{o}_{t}"
                )
                dio_frac = operational_modes[o]["diorite_frac"]
                sil_frac = operational_modes[o]["silicified_frac"]
                dio_mass = mdl.sum(m[b,s,o,t] for b in B if rock[b]==0)
                sil_mass = mdl.sum(m[b,s,o,t] for b in B if rock[b]==1)
                mdl.add_constraint(dio_mass == dio_frac * M_tot[s,o,t], ctname=f"blend_dio_{s}_{o}_{t}")
                mdl.add_constraint(sil_mass == sil_frac * M_tot[s,o,t], ctname=f"blend_sil_{s}_{o}_{t}")

    mining_cost = mdl.sum(mass[b]*x[b,t]*dcf[t]*2.5 for b in B for t in T)

    proc_value_terms = []
    for s in S:
        for o in modes:
            rec = operational_modes[o]["recovery"]
            pc = operational_modes[o]["cost"]
            for b in B:
                oz_per_ton = grades[s][b] / 31.1035
                revenue_per_ton = oz_per_ton * gold_price * rec
                net_per_ton = revenue_per_ton - pc
                for t in T:
                    proc_value_terms.append(net_per_ton * m[b,s,o,t] * dcf[t])
    processing_value = (1.0/num_scenarios) * mdl.sum(proc_value_terms)

    mdl.maximize(processing_value - mining_cost)

    mdl.parameters.mip.tolerances.mipgap = mip_gap
    mdl.parameters.timelimit = time_limit

    sol = mdl.solve(log_output=True)
    return mdl, sol

if __name__ == "__main__":
    mdl, sol = build_and_solve()
    if sol:
        print("Objective (NPV):", sol.objective_value)
