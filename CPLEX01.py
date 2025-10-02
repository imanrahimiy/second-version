import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docplex.mp.model import Model
import warnings
warnings.filterwarnings('ignore')


class MineOptimizationDSS:
    def __init__(self, case_study_params=None):
        """Initialize parameters and containers."""
        defaults = {
            'num_blocks': 500,
            'num_periods': 6,
            'num_scenarios': 10,
            'block_weight': 15375,              # tons
            'discount_rate': 0.08,
            'gold_price': 1190,                 # $/oz
            'mining_cost': 20.5,                # $/ton
            'mining_capacity': 6_500_000,       # ton/period
            'processing_availability': 8075,    # hours/period
            'operational_modes': {
                'A': {'processing_rate': 250, 'processing_cost': 21.4,
                      'recovery_rate': 0.83, 'diorite_fraction': 0.65,
                      'silicified_fraction': 0.35},
                'B': {'processing_rate': 200, 'processing_cost': 24.9,
                      'recovery_rate': 0.83, 'diorite_fraction': 0.45,
                      'silicified_fraction': 0.55}
            }
        }
        if case_study_params:
            defaults.update(case_study_params)
        self.params = defaults

        self.blocks_data = None
        self.uncertainty_factors = None  # dict with 'combined_matrix' etc.
        self.solutions = {}
        self.results = {}

    # ------------------------------------------------------------------ #
    # Data generation + uncertainty
    # ------------------------------------------------------------------ #
    def generate_synthetic_data(self):
        """Create synthetic blocks with scenario grades and simple precedence."""
        np.random.seed(42)
        blocks = []
        for b in range(self.params['num_blocks']):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            z = np.random.uniform(0, 300)

            # rock_type: 0=Diorite, 1=Silicified
            rock_type = np.random.choice([0, 1], p=[0.6, 0.4])

            # scenario grades (g/t)
            base_grade = np.random.lognormal(mean=-1.0, sigma=0.8)
            grades = {}
            for s in range(self.params['num_scenarios']):
                variation = np.random.normal(0, 0.3)
                grades[s] = max(0.1, base_grade * (1 + variation))

            # toy spatial precedence
            predecessors = []
            for other_b in range(b):
                other = blocks[other_b]
                if (other['z'] > z and abs(other['x'] - x) < 50 and abs(other['y'] - y) < 50):
                    predecessors.append(other_b)

            blocks.append({
                'block_id': b,
                'x': x, 'y': y, 'z': z,
                'mass': self.params['block_weight'],
                'rock_type': rock_type,
                'grades': grades,
                'predecessors': predecessors
            })

        self.blocks_data = pd.DataFrame(blocks)
        print(f"Generated {len(self.blocks_data)} blocks with scenario grades.")

    def calculate_dynamic_uncertainty_factors(self):
        """σ_{s,t} = γ_s × φ_t  (scenario weight × time modifier)."""
        S = self.params['num_scenarios']
        T = self.params['num_periods']

        scenario_weights = np.random.uniform(0.8, 1.2, S)  # γ_s
        time_decay = np.array([1.0, 1.05, 1.10, 1.15, 1.20, 1.25])[:T]  # φ_t
        combined = np.outer(scenario_weights, time_decay)                # S × T

        self.uncertainty_factors = {
            'scenario_weights': scenario_weights,
            'time_decay': time_decay,
            'combined_matrix': combined
        }
        return combined

    # ------------------------------------------------------------------ #
    # Integrated two-stage stochastic (risk-neutral)
    # ------------------------------------------------------------------ #
    def solve_two_stage_stochastic(self):
        """Deterministic-equivalent MILP with scenarios (risk-neutral)."""
        print("Solving integrated two-stage stochastic model...")

        P = self.params
        B = range(P['num_blocks'])
        T = range(P['num_periods'])
        S = range(P['num_scenarios'])
        MODES = list(P['operational_modes'].keys())

        if self.uncertainty_factors is None:
            # default to no extra uncertainty scaling if not provided
            unc = np.ones((P['num_scenarios'], P['num_periods']))
        else:
            unc = self.uncertainty_factors['combined_matrix']

        # discounted mining cost per (b,t)
        disc_mine_cost = {(b, t): P['mining_cost'] * self.blocks_data.loc[b, 'mass']
                          * (1 + P['discount_rate']) ** (-t)
                          for b in B for t in T}

        # discounted net value per ton for (b,o,s,t), including uncertainty factor
        net_value_ton = {}
        for b in B:
            grades = self.blocks_data.loc[b, 'grades']  # dict by scenario
            for s in S:
                grade = float(grades[s])                 # g/t
                oz_per_ton = grade / 31.1035
                for o in MODES:
                    mode = P['operational_modes'][o]
                    revenue_per_ton = oz_per_ton * P['gold_price'] * mode['recovery_rate']
                    net = revenue_per_ton - mode['processing_cost']  # $/ton
                    for t in T:
                        net_value_ton[(b, o, s, t)] = (
                            net * (1 + P['discount_rate']) ** (-t) * float(unc[s, t])
                        )

        mdl = Model(name="MinePlanning_TwoStage_Stochastic")

        # Stage-1 binaries
        x = {(b, t): mdl.binary_var(name=f"x_{b}_{t}") for b in B for t in T}

        # Stage-2 recourse variables (indexed by scenario)
        m = {(b, o, s, t): mdl.continuous_var(lb=0, name=f"m_{b}_{o}_{s}_{t}")
             for b in B for o in MODES for s in S for t in T}

        # ----- Constraints -----

        # Mine at most once
        for b in B:
            mdl.add_constraint(mdl.sum(x[b, t] for t in T) <= 1, ctname=f"mine_once_{b}")

        # Precedence: if b at t, all predecessors mined in 0..t
        for b in B:
            preds = self.blocks_data.loc[b, 'predecessors']
            if preds:
                for t in T:
                    for pidx in preds:
                        if 0 <= pidx < P['num_blocks']:
                            mdl.add_constraint(mdl.sum(x[pidx, tau] for tau in range(t + 1)) >= x[b, t],
                                               ctname=f"prec_b{b}_p{pidx}_t{t}")

        # Mining capacity per period
        for t in T:
            mdl.add_constraint(
                mdl.sum(x[b, t] * self.blocks_data.loc[b, 'mass'] for b in B) <= P['mining_capacity'],
                ctname=f"cap_mine_t{t}"
            )

        # Linking per (s,t): can only process in the period it is mined
        for b in B:
            mb = self.blocks_data.loc[b, 'mass']
            for t in T:
                for s in S:
                    mdl.add_constraint(
                        mdl.sum(m[b, o, s, t] for o in MODES) <= mb * x[b, t],
                        ctname=f"link_b{b}_s{s}_t{t}"
                    )

        # Processing time per (s,t)
        for s in S:
            for t in T:
                mdl.add_constraint(
                    mdl.sum(m[b, o, s, t] / P['operational_modes'][o]['processing_rate']
                            for b in B for o in MODES)
                    <= P['processing_availability'],
                    ctname=f"proc_time_s{s}_t{t}"
                )

        # Blending equality (diorite fraction) per (o,s,t)
        for s in S:
            for t in T:
                for o in MODES:
                    frac = P['operational_modes'][o]['diorite_fraction']
                    diorite_mass = mdl.sum(
                        m[b, o, s, t] for b in B if self.blocks_data.loc[b, 'rock_type'] == 0
                    )
                    total_mass_mode = mdl.sum(m[b, o, s, t] for b in B)
                    # If total_mass_mode == 0, equality reduces to 0 == 0 (OK).
                    mdl.add_constraint(diorite_mass == frac * total_mass_mode,
                                       ctname=f"blend_o{o}_s{s}_t{t}")

        # ----- Objective: expected (avg over scenarios) processing value - discounted mining cost -----
        expected_processing_value = (1.0 / P['num_scenarios']) * mdl.sum(
            net_value_ton[(b, o, s, t)] * m[b, o, s, t] for b in B for o in MODES for s in S for t in T
        )
        total_mining_cost = mdl.sum(disc_mine_cost[(b, t)] * x[b, t] for b in B for t in T)
        mdl.maximize(expected_processing_value - total_mining_cost)

        mdl.parameters.timelimit = 1800
        sol = mdl.solve(log_output=True)

        if not sol:
            print("Stochastic model failed to solve.")
            return None

        print(f"Solved. Objective value: {sol.objective_value:,.2f}")

        # ----- Extract schedules and plans -----
        mining_schedule = {t: [] for t in T}
        for b in B:
            for t in T:
                if x[b, t].solution_value > 0.5:
                    mining_schedule[t].append(b)

        # Processing plan by (s,t)
        processing_plan = {t: {s: {} for s in S} for t in T}
        for t in T:
            for s in S:
                for b in B:
                    for o in MODES:
                        val = m[b, o, s, t].solution_value
                        if val > 1e-6:
                            processing_plan[t][s].setdefault(b, {})[o] = val

        # Per (s,t) value and per-t mining cost (for analysis/plots)
        period_processing_value = {
            (s, t): sum(net_value_ton[(b, o, s, t)] * m[b, o, s, t].solution_value
                        for b in B for o in MODES)
            for s in S for t in T
        }
        period_mining_cost = {
            t: sum(disc_mine_cost[(b, t)] * x[b, t].solution_value for b in B) for t in T
        }

        self.solutions['two_stage_stoch'] = {
            'model': mdl,
            'solution': sol,
            'x': x,                # binaries
            'm': m,                # recourse
            'mining_schedule': mining_schedule,
            'processing_plan': processing_plan,
            'period_processing_value': period_processing_value,
            'period_mining_cost': period_mining_cost
        }
        return mining_schedule

    # ------------------------------------------------------------------ #
    # Results & plots (compatible with your existing visualizations)
    # ------------------------------------------------------------------ #
    def analyze_results(self):
        print("Analyzing results...")
        P = self.params
        T = range(P['num_periods'])

        if 'two_stage_stoch' not in self.solutions:
            print("No stochastic solution available.")
            return

        soln = self.solutions['two_stage_stoch']
        schedule = soln['mining_schedule']
        x = soln['x']

        # Period stats (mass & utilization)
        period_stats = {}
        for t in T:
            mined_blocks = schedule.get(t, [])
            total_mass = sum(self.blocks_data.loc[b, 'mass'] for b in mined_blocks)
            diorite_mass = sum(self.blocks_data.loc[b, 'mass'] for b in mined_blocks
                               if self.blocks_data.loc[b, 'rock_type'] == 0)
            silicified_mass = total_mass - diorite_mass
            period_stats[t] = {
                'blocks_mined': len(mined_blocks),
                'total_mass': total_mass,
                'diorite_mass': diorite_mass,
                'silicified_mass': silicified_mass,
                'capacity_utilization': (total_mass / P['mining_capacity']) if P['mining_capacity'] > 0 else 0
            }

        # Scenario NPVs per period (processing value - mining cost at t)
        S = range(P['num_scenarios'])
        scen_npvs_by_t = {t: [] for t in T}
        for t in T:
            mc_t = soln['period_mining_cost'][t]
            for s in S:
                val_proc = soln['period_processing_value'][(s, t)]
                scen_npvs_by_t[t].append(val_proc - mc_t)

        cumulative_npv = {
            t: {'scenarios': scen_npvs_by_t[t],
                'mean': float(np.mean(scen_npvs_by_t[t])) if scen_npvs_by_t[t] else 0.0,
                'std': float(np.std(scen_npvs_by_t[t])) if scen_npvs_by_t[t] else 0.0}
            for t in T
        }

        self.results['period_analysis'] = period_stats
        self.results['cumulative_npv'] = cumulative_npv
        return period_stats, cumulative_npv

    # ------------------------------------------------------------------ #
    # (Your plotting function is unchanged; it will use results above.)
    # ------------------------------------------------------------------ #
    def create_visualizations(self):
        print("Creating visualizations...")
        if not self.results:
            print("No results available for visualization")
            return

        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))

        if 'period_analysis' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            periods = list(self.results['period_analysis'].keys())
            diorite_masses = [self.results['period_analysis'][t]['diorite_mass'] / 1000 for t in periods]
            silicified_masses = [self.results['period_analysis'][t]['silicified_mass'] / 1000 for t in periods]
            width = 0.6
            ax1.bar(periods, diorite_masses, width, label='Diorite Porphyry',
                    color='gold', alpha=0.8)
            ax1.bar(periods, silicified_masses, width, bottom=diorite_masses,
                    label='Silicified Breccia', color='purple', alpha=0.8)
            ax1.set_xlabel('Period'); ax1.set_ylabel('Mass (thousands tons)')
            ax1.set_title('Rock Production by Type'); ax1.legend(); ax1.grid(True, alpha=0.3)

        if 'period_analysis' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            utilizations = [self.results['period_analysis'][t]['capacity_utilization'] * 100
                            for t in periods]
            bars = ax2.bar(periods, utilizations, color='steelblue', alpha=0.7)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Capacity Limit')
            ax2.set_xlabel('Period'); ax2.set_ylabel('Capacity Utilization (%)')
            ax2.set_title('Mining Capacity Utilization'); ax2.legend(); ax2.grid(True, alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                         ha='center', va='bottom')

        if 'cumulative_npv' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            scenario_means = [self.results['cumulative_npv'][t]['mean'] / 1e6
                              for t in periods if t in self.results['cumulative_npv']]
            scenario_stds = [self.results['cumulative_npv'][t]['std'] / 1e6
                             for t in periods if t in self.results['cumulative_npv']]
            if scenario_means:
                ax3.errorbar(periods[:len(scenario_means)], scenario_means,
                             yerr=scenario_stds, marker='o', capsize=5,
                             color='darkgreen', linewidth=2)
                ax3.set_xlabel('Period'); ax3.set_ylabel('NPV (Million $)')
                ax3.set_title('Expected NPV by Period'); ax3.grid(True, alpha=0.3)

        # Uncertainty heatmap (σ_{s,t})
        if self.uncertainty_factors is not None:
            ax4 = plt.subplot(3, 3, 4)
            uncertainty_matrix = self.uncertainty_factors['combined_matrix']
            im = ax4.imshow(uncertainty_matrix, cmap='RdYlBu_r', aspect='auto')
            ax4.set_xlabel('Time Period'); ax4.set_ylabel('Scenario')
            ax4.set_title('Dynamic Uncertainty Factors σ_{s,t}')
            cbar = plt.colorbar(im, ax=ax4); cbar.set_label('Uncertainty Factor')
            for i in range(uncertainty_matrix.shape[0]):
                for j in range(uncertainty_matrix.shape[1]):
                    ax4.text(j, i, f'{uncertainty_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        # Grade distribution by rock type
        ax7 = plt.subplot(3, 3, 7)
        diorite_grades, silicified_grades = [], []
        for _, block in self.blocks_data.iterrows():
            avg_grade = np.mean(list(block['grades'].values()))
            (diorite_grades if block['rock_type'] == 0 else silicified_grades).append(avg_grade)
        ax7.hist(diorite_grades, bins=30, alpha=0.6, label='Diorite Porphyry',
                 color='gold', density=True)
        ax7.hist(silicified_grades, bins=30, alpha=0.6, label='Silicified Breccia',
                 color='purple', density=True)
        ax7.set_xlabel('Gold Grade (g/t)'); ax7.set_ylabel('Density')
        ax7.set_title('Grade Distribution by Rock Type'); ax7.legend(); ax7.grid(True, alpha=0.3)

        # Simple sensitivity (unchanged)
        ax9 = plt.subplot(3, 3, 9)
        gold_prices = np.linspace(1000, 1400, 11)
        base_npv = 1500  # placeholder
        npv_sensitivity = [base_npv * (p / self.params['gold_price']) for p in gold_prices]
        ax9.plot(gold_prices, npv_sensitivity, marker='o', linewidth=2, color='green')
        ax9.axhline(y=base_npv, color='red', linestyle='--', alpha=0.7, label=f'Base NPV: ${base_npv}M')
        ax9.axvline(x=self.params['gold_price'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Base Price: ${self.params["gold_price"]}/oz')
        ax9.set_xlabel('Gold Price ($/oz)'); ax9.set_ylabel('NPV (Million $)')
        ax9.set_title('Sensitivity to Gold Price'); ax9.legend(); ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('mine_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "="*60)
        print("MINE OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        if 'cumulative_npv' in self.results:
            print("\nMining Schedule Summary:")
            for t in periods:
                stats = self.results['period_analysis'][t]
                print(f"  Period {t+1}: {stats['blocks_mined']:,} blocks, "
                      f"{stats['total_mass']/1e6:.2f}M tons "
                      f"({stats['capacity_utilization']*100:.1f}% capacity)")
            print("\nPer-period expected NPV (scenario mean):")
            for t in periods:
                mean = self.results['cumulative_npv'][t]['mean'] / 1e6
                std = self.results['cumulative_npv'][t]['std'] / 1e6
                print(f"  Period {t+1}: {mean:.2f} ± {std:.2f} M$")

# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def run_mine_optimization():
    print("Starting Mine Optimization (two-stage stochastic, risk-neutral)")
    print("=" * 60)

    dss = MineOptimizationDSS()
    dss.generate_synthetic_data()
    dss.calculate_dynamic_uncertainty_factors()   # re-introduced uncertainty

    mining_schedule = dss.solve_two_stage_stochastic()
    if mining_schedule:
        dss.analyze_results()
        dss.create_visualizations()
        return dss
    else:
        print("Optimization failed")
        return None


if __name__ == "__main__":
    dss_system = run_mine_optimization()
    if dss_system:
        print("\nAnalysis completed successfully!")
        print("Results saved to 'mine_optimization_results.png'")
    else:
        print("Analysis failed!")
