import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docplex.mp.model import Model
from docplex.mp.progress import TextProgressListener
import random
import warnings
warnings.filterwarnings('ignore')

class MineOptimizationDSS:
    def __init__(self, case_study_params=None):
        """
        Initialize the Mine Optimization Decision Support System
        """
        # Default parameters based on the case study
        if case_study_params is None:
            self.params = {
                'num_blocks': 50000,  # Reduced for computational efficiency
                'num_periods': 6,
                'num_scenarios': 10,
                'block_weight': 15375,  # tons
                'block_dimensions': (20, 20, 15),  # meters
                'block_density': 2.56,  # ton/m³
                'discount_rate': 0.08,
                'gold_price': 1190,  # $/oz
                'mining_cost': 20.5,  # $/ton
                'mining_capacity': 6500000,  # ton/period
                'processing_availability': 8075,  # hours/period
                'operational_modes': {
                    'A': {
                        'processing_rate': 250,  # ton/hr
                        'processing_cost': 21.4,  # $/ton
                        'recovery_rate': 0.83,
                        'diorite_fraction': 0.65,
                        'silicified_fraction': 0.35
                    },
                    'B': {
                        'processing_rate': 200,  # ton/hr
                        'processing_cost': 24.9,  # $/ton
                        'recovery_rate': 0.83,
                        'diorite_fraction': 0.45,
                        'silicified_fraction': 0.55
                    }
                }
            }
        else:
            self.params = case_study_params
        
        # Initialize data structures
        self.blocks_data = None
        self.uncertainty_factors = None
        self.solutions = {}
        self.results = {}
        
    def generate_synthetic_data(self):
        """
        Generate synthetic block data based on case study parameters
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate block coordinates and properties
        blocks = []
        for b in range(self.params['num_blocks']):
            # Generate spatial coordinates
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            z = np.random.uniform(0, 300)
            
            # Generate rock type (0: Diorite, 1: Silicified)
            rock_type = np.random.choice([0, 1], p=[0.6, 0.4])
            
            # Generate grade scenarios with spatial correlation
            base_grade = np.random.lognormal(mean=-1.0, sigma=0.8)
            grades = {}
            for s in range(self.params['num_scenarios']):
                # Add scenario-specific variation
                variation = np.random.normal(0, 0.3)
                grades[s] = max(0.1, base_grade * (1 + variation))
            
            # Calculate precedence relationships (simplified)
            predecessors = []
            for other_b in range(b):
                other_block = blocks[other_b] if blocks else None
                if other_block and other_block['z'] > z and \
                   abs(other_block['x'] - x) < 50 and abs(other_block['y'] - y) < 50:
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
        print(f"Generated {len(self.blocks_data)} blocks with geological scenarios")
        
    def calculate_dynamic_uncertainty_factors(self):
        """
        Calculate dynamic uncertainty factors σ_{s,t} = γ_s × φ_t
        """
        # Scenario-specific uncertainty weights (γ_s)
        scenario_weights = np.random.uniform(0.8, 1.2, self.params['num_scenarios'])
        
        # Time-decay modifiers (φ_t) - uncertainty grows with future periods
        time_decay = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 1.25])[:self.params['num_periods']]
        
        # Calculate combined uncertainty factors
        uncertainty_matrix = np.outer(scenario_weights, time_decay)
        
        self.uncertainty_factors = {
            'scenario_weights': scenario_weights,
            'time_decay': time_decay,
            'combined_matrix': uncertainty_matrix
        }
        
        return uncertainty_matrix
    
    def calculate_block_values(self):
        """
        Calculate block values considering uncertainty factors
        """
        values = {}
        
        for _, block in self.blocks_data.iterrows():
            b = block['block_id']
            values[b] = {}
            
            for s in range(self.params['num_scenarios']):
                values[b][s] = {}
                for t in range(self.params['num_periods']):
                    # Base revenue calculation
                    grade = block['grades'][s]
                    recovery_rate = 0.83  # Average recovery rate
                    
                    # Convert grade to revenue ($/ton)
                    # Assuming grade is in g/t, convert to oz/ton
                    oz_per_ton = grade / 31.1035  # grams to ounces
                    revenue_per_ton = oz_per_ton * self.params['gold_price'] * recovery_rate
                    
                    # Apply discount factor
                    discount_factor = (1 + self.params['discount_rate']) ** (-t)
                    
                    # Apply dynamic uncertainty factor
                    uncertainty_factor = self.uncertainty_factors['combined_matrix'][s, t]
                    
                    # Calculate expected net present value
                    processing_cost = 23.0  # Average processing cost
                    net_value_per_ton = (revenue_per_ton - processing_cost) * discount_factor
                    
                    # Apply uncertainty factor
                    adjusted_value = uncertainty_factor * net_value_per_ton
                    
                    values[b][s][t] = adjusted_value * block['mass']
        
        return values
    
    def solve_first_stage(self):
        """
        Solve the first stage optimization problem
        """
        print("Solving First Stage Optimization...")
        
        # Create CPLEX model
        model = Model(name="MinePlanning_Stage1")
        
        # Decision variables: x[b,t] = 1 if block b is mined in period t
        x = {}
        for b in range(self.params['num_blocks']):
            for t in range(self.params['num_periods']):
                x[b, t] = model.binary_var(name=f"x_{b}_{t}")
        
        # Calculate block values
        block_values = self.calculate_block_values()
        
        # Objective function: maximize NPV
        mining_costs = {}
        expected_revenues = {}
        
        for b in range(self.params['num_blocks']):
            for t in range(self.params['num_periods']):
                # Mining cost (discounted)
                discount_factor = (1 + self.params['discount_rate']) ** (-t)
                mining_costs[b, t] = self.params['mining_cost'] * \
                                   self.blocks_data.iloc[b]['mass'] * discount_factor
                
                # Expected revenue across scenarios
                expected_revenue = sum(block_values[b][s][t] 
                                     for s in range(self.params['num_scenarios'])) / \
                                 self.params['num_scenarios']
                expected_revenues[b, t] = expected_revenue
        
        # Objective: maximize expected NPV
        objective = model.sum(
            (expected_revenues[b, t] - mining_costs[b, t]) * x[b, t]
            for b in range(self.params['num_blocks'])
            for t in range(self.params['num_periods'])
        )
        model.maximize(objective)
        
        # Constraints
        
        # 1. Each block can be mined at most once
        for b in range(self.params['num_blocks']):
            model.add_constraint(
                model.sum(x[b, t] for t in range(self.params['num_periods'])) <= 1,
                ctname=f"mine_once_{b}"
            )
        
        # 2. Precedence constraints
        for b in range(self.params['num_blocks']):
            predecessors = self.blocks_data.iloc[b]['predecessors']
            if predecessors:
                for t in range(self.params['num_periods']):
                    # If block b is mined in period t, all predecessors must be mined in periods 0 to t
                    for pred in predecessors:
                        if pred < self.params['num_blocks']:  # Validate predecessor index
                            model.add_constraint(
                                model.sum(x[pred, tau] for tau in range(t + 1)) >= x[b, t],
                                ctname=f"precedence_{b}_{pred}_{t}"
                            )
        
        # 3. Mining capacity constraints
        for t in range(self.params['num_periods']):
            total_mass = model.sum(
                x[b, t] * self.blocks_data.iloc[b]['mass']
                for b in range(self.params['num_blocks'])
            )
            model.add_constraint(
                total_mass <= self.params['mining_capacity'],
                ctname=f"capacity_{t}"
            )
        
        # Solve the model
        model.parameters.timelimit = 1800  # 30 minutes time limit
        solution = model.solve(log_output=True)
        
        if solution:
            print(f"First stage solved successfully. Objective value: {solution.objective_value:,.2f}")
            
            # Extract solution
            mining_schedule = {}
            for t in range(self.params['num_periods']):
                mining_schedule[t] = []
                for b in range(self.params['num_blocks']):
                    if x[b, t].solution_value > 0.5:
                        mining_schedule[t].append(b)
            
            self.solutions['first_stage'] = {
                'model': model,
                'solution': solution,
                'mining_schedule': mining_schedule,
                'variables': x
            }
            
            return mining_schedule
        else:
            print("First stage optimization failed")
            return None
    
    def solve_second_stage(self, mining_schedule):
        """
        Solve the second stage optimization for each period and scenario
        """
        print("Solving Second Stage Optimization...")
        
        second_stage_results = {}
        
        for t in range(self.params['num_periods']):
            second_stage_results[t] = {}
            blocks_in_period = mining_schedule.get(t, [])
            
            if not blocks_in_period:
                continue
                
            for s in range(self.params['num_scenarios']):
                # Create model for this period and scenario
                model = Model(name=f"MinePlanning_Stage2_t{t}_s{s}")
                
                # Decision variables: m[b,o] = mass of block b processed in mode o
                m = {}
                operational_modes = ['A', 'B']
                
                for b in blocks_in_period:
                    for o in operational_modes:
                        m[b, o] = model.continuous_var(
                            lb=0,
                            name=f"m_{b}_{o}"
                        )
                
                # Objective: maximize processing value
                processing_values = {}
                for b in blocks_in_period:
                    block_data = self.blocks_data.iloc[b]
                    grade = block_data['grades'][s]
                    
                    for o in operational_modes:
                        mode_params = self.params['operational_modes'][o]
                        
                        # Calculate value per ton for this mode
                        oz_per_ton = grade / 31.1035
                        revenue_per_ton = oz_per_ton * self.params['gold_price'] * \
                                        mode_params['recovery_rate']
                        net_value_per_ton = revenue_per_ton - mode_params['processing_cost']
                        
                        processing_values[b, o] = net_value_per_ton
                
                objective = model.sum(
                    processing_values[b, o] * m[b, o]
                    for b in blocks_in_period
                    for o in operational_modes
                )
                model.maximize(objective)
                
                # Constraints
                
                # 1. Mass allocation constraint: sum of masses <= block mass
                for b in blocks_in_period:
                    block_mass = self.blocks_data.iloc[b]['mass']
                    model.add_constraint(
                        model.sum(m[b, o] for o in operational_modes) <= block_mass,
                        ctname=f"mass_limit_{b}"
                    )
                
                # 2. Processing time constraint
                total_processing_time = model.sum(
                    m[b, o] / self.params['operational_modes'][o]['processing_rate']
                    for b in blocks_in_period
                    for o in operational_modes
                )
                model.add_constraint(
                    total_processing_time <= self.params['processing_availability'],
                    ctname="processing_time"
                )
                
                # 3. Rock type blending constraints
                for o in operational_modes:
                    mode_params = self.params['operational_modes'][o]
                    
                    # Diorite constraint
                    diorite_mass = model.sum(
                        m[b, o] for b in blocks_in_period
                        if self.blocks_data.iloc[b]['rock_type'] == 0
                    )
                    total_mass_mode = model.sum(m[b, o] for b in blocks_in_period)
                    
                    if total_mass_mode.ub > 0:
                        model.add_constraint(
                            diorite_mass == mode_params['diorite_fraction'] * total_mass_mode,
                            ctname=f"diorite_blend_{o}"
                        )
                
                # Solve
                solution = model.solve(log_output=False)
                
                if solution:
                    second_stage_results[t][s] = {
                        'objective_value': solution.objective_value,
                        'processing_plan': {}
                    }
                    
                    for b in blocks_in_period:
                        second_stage_results[t][s]['processing_plan'][b] = {}
                        for o in operational_modes:
                            if m[b, o].solution_value > 0.001:
                                second_stage_results[t][s]['processing_plan'][b][o] = \
                                    m[b, o].solution_value
                else:
                    print(f"Second stage failed for period {t}, scenario {s}")
                    second_stage_results[t][s] = {'objective_value': 0, 'processing_plan': {}}
        
        self.solutions['second_stage'] = second_stage_results
        return second_stage_results
    
    def monte_carlo_simulation(self, num_simulations=1000):
        """
        Perform Monte Carlo simulation for risk analysis
        """
        print(f"Running Monte Carlo simulation with {num_simulations} iterations...")
        
        mc_results = []
        
        for sim in range(num_simulations):
            # Generate random parameters
            gold_price_variation = np.random.normal(1.0, 0.1)
            cost_variation = np.random.normal(1.0, 0.05)
            
            # Adjust parameters
            adjusted_gold_price = self.params['gold_price'] * gold_price_variation
            adjusted_mining_cost = self.params['mining_cost'] * cost_variation
            
            # Calculate NPV for this simulation
            total_npv = 0
            
            if 'first_stage' in self.solutions:
                mining_schedule = self.solutions['first_stage']['mining_schedule']
                
                for t in range(self.params['num_periods']):
                    blocks_in_period = mining_schedule.get(t, [])
                    
                    for b in blocks_in_period:
                        block_data = self.blocks_data.iloc[b]
                        
                        # Random scenario selection
                        s = np.random.randint(0, self.params['num_scenarios'])
                        grade = block_data['grades'][s]
                        
                        # Calculate value
                        oz_per_ton = grade / 31.1035
                        revenue = oz_per_ton * adjusted_gold_price * 0.83 * block_data['mass']
                        cost = adjusted_mining_cost * block_data['mass']
                        
                        # Apply discount
                        discount_factor = (1 + self.params['discount_rate']) ** (-t)
                        npv_contribution = (revenue - cost) * discount_factor
                        
                        total_npv += npv_contribution
            
            mc_results.append(total_npv)
        
        self.results['monte_carlo'] = {
            'simulations': mc_results,
            'mean': np.mean(mc_results),
            'std': np.std(mc_results),
            'percentiles': {
                'P10': np.percentile(mc_results, 10),
                'P50': np.percentile(mc_results, 50),
                'P90': np.percentile(mc_results, 90)
            }
        }
        
        return mc_results
    
    def analyze_results(self):
        """
        Analyze optimization results and calculate key metrics
        """
        print("Analyzing results...")
        
        if 'first_stage' not in self.solutions:
            print("No first stage solution available")
            return
        
        # Extract mining schedule
        mining_schedule = self.solutions['first_stage']['mining_schedule']
        
        # Calculate period-wise statistics
        period_stats = {}
        cumulative_npv = {}
        
        for t in range(self.params['num_periods']):
            blocks_in_period = mining_schedule.get(t, [])
            
            total_mass = sum(self.blocks_data.iloc[b]['mass'] for b in blocks_in_period)
            rock_types = [self.blocks_data.iloc[b]['rock_type'] for b in blocks_in_period]
            diorite_mass = sum(self.blocks_data.iloc[b]['mass'] for b in blocks_in_period 
                             if self.blocks_data.iloc[b]['rock_type'] == 0)
            silicified_mass = total_mass - diorite_mass
            
            period_stats[t] = {
                'blocks_mined': len(blocks_in_period),
                'total_mass': total_mass,
                'diorite_mass': diorite_mass,
                'silicified_mass': silicified_mass,
                'capacity_utilization': total_mass / self.params['mining_capacity']
            }
            
            # Calculate NPV for each scenario
            scenario_npvs = []
            if 'second_stage' in self.solutions and t in self.solutions['second_stage']:
                for s in range(self.params['num_scenarios']):
                    if s in self.solutions['second_stage'][t]:
                        scenario_npvs.append(self.solutions['second_stage'][t][s]['objective_value'])
            
            cumulative_npv[t] = {
                'scenarios': scenario_npvs,
                'mean': np.mean(scenario_npvs) if scenario_npvs else 0,
                'std': np.std(scenario_npvs) if scenario_npvs else 0
            }
        
        self.results['period_analysis'] = period_stats
        self.results['cumulative_npv'] = cumulative_npv
        
        return period_stats, cumulative_npv
    
    def create_visualizations(self):
        """
        Create comprehensive visualization plots
        """
        print("Creating visualizations...")
        
        if not self.results:
            print("No results available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Mining Schedule by Period
        if 'period_analysis' in self.results:
            ax1 = plt.subplot(3, 3, 1)
            periods = list(self.results['period_analysis'].keys())
            diorite_masses = [self.results['period_analysis'][t]['diorite_mass']/1000 
                            for t in periods]
            silicified_masses = [self.results['period_analysis'][t]['silicified_mass']/1000 
                               for t in periods]
            
            width = 0.6
            ax1.bar(periods, diorite_masses, width, label='Diorite Porphyry', 
                   color='gold', alpha=0.8)
            ax1.bar(periods, silicified_masses, width, bottom=diorite_masses, 
                   label='Silicified Breccia', color='purple', alpha=0.8)
            
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Mass (thousands tons)')
            ax1.set_title('Rock Production by Type')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Capacity Utilization
        if 'period_analysis' in self.results:
            ax2 = plt.subplot(3, 3, 2)
            utilizations = [self.results['period_analysis'][t]['capacity_utilization'] * 100 
                          for t in periods]
            
            bars = ax2.bar(periods, utilizations, color='steelblue', alpha=0.7)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Capacity Limit')
            
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Capacity Utilization (%)')
            ax2.set_title('Mining Capacity Utilization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. NPV by Scenario (if available)
        if 'cumulative_npv' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            scenario_means = [self.results['cumulative_npv'][t]['mean']/1e6 
                            for t in periods if t in self.results['cumulative_npv']]
            scenario_stds = [self.results['cumulative_npv'][t]['std']/1e6 
                           for t in periods if t in self.results['cumulative_npv']]
            
            if scenario_means:
                ax3.errorbar(periods[:len(scenario_means)], scenario_means, 
                           yerr=scenario_stds, marker='o', capsize=5, 
                           color='darkgreen', linewidth=2)
                ax3.set_xlabel('Period')
                ax3.set_ylabel('NPV (Million $)')
                ax3.set_title('Expected NPV by Period')
                ax3.grid(True, alpha=0.3)
        
        # 4. Uncertainty Factors Heatmap
        if hasattr(self, 'uncertainty_factors'):
            ax4 = plt.subplot(3, 3, 4)
            uncertainty_matrix = self.uncertainty_factors['combined_matrix']
            
            im = ax4.imshow(uncertainty_matrix, cmap='RdYlBu_r', aspect='auto')
            ax4.set_xlabel('Time Period')
            ax4.set_ylabel('Scenario')
            ax4.set_title('Dynamic Uncertainty Factors σ_{s,t}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Uncertainty Factor')
            
            # Add value annotations
            for i in range(uncertainty_matrix.shape[0]):
                for j in range(uncertainty_matrix.shape[1]):
                    text = ax4.text(j, i, f'{uncertainty_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # 5. Monte Carlo Results Distribution (if available)
        if 'monte_carlo' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            mc_results = self.results['monte_carlo']['simulations']
            
            ax5.hist(np.array(mc_results)/1e6, bins=50, alpha=0.7, color='lightblue', 
                    edgecolor='black')
            
            # Add percentile lines
            percentiles = self.results['monte_carlo']['percentiles']
            for label, value in percentiles.items():
                ax5.axvline(value/1e6, color='red', linestyle='--', alpha=0.7, 
                           label=f'{label}: ${value/1e6:.1f}M')
            
            ax5.set_xlabel('NPV (Million $)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Monte Carlo NPV Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Risk Profile Curves
        if 'monte_carlo' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            mc_results = self.results['monte_carlo']['simulations']
            
            # Calculate cumulative probability
            sorted_results = np.sort(mc_results)
            probabilities = np.arange(1, len(sorted_results) + 1) / len(sorted_results) * 100
            
            ax6.plot(sorted_results/1e6, probabilities, linewidth=2, color='darkblue')
            ax6.set_xlabel('NPV (Million $)')
            ax6.set_ylabel('Cumulative Probability (%)')
            ax6.set_title('Risk Profile Curve')
            ax6.grid(True, alpha=0.3)
            
            # Highlight key percentiles
            for p in [10, 50, 90]:
                value = np.percentile(mc_results, p)
                ax6.axvline(value/1e6, color='red', linestyle=':', alpha=0.7)
                ax6.text(value/1e6, p, f'P{p}', rotation=90, ha='right')
        
        # 7. Grade Distribution by Rock Type
        ax7 = plt.subplot(3, 3, 7)
        
        # Extract grades for visualization
        diorite_grades = []
        silicified_grades = []
        
        for _, block in self.blocks_data.iterrows():
            avg_grade = np.mean(list(block['grades'].values()))
            if block['rock_type'] == 0:
                diorite_grades.append(avg_grade)
            else:
                silicified_grades.append(avg_grade)
        
        ax7.hist(diorite_grades, bins=30, alpha=0.6, label='Diorite Porphyry', 
                color='gold', density=True)
        ax7.hist(silicified_grades, bins=30, alpha=0.6, label='Silicified Breccia', 
                color='purple', density=True)
        
        ax7.set_xlabel('Gold Grade (g/t)')
        ax7.set_ylabel('Density')
        ax7.set_title('Grade Distribution by Rock Type')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Economic Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Create summary text
        summary_text = "Economic Summary\n\n"
        
        if 'monte_carlo' in self.results:
            mc_stats = self.results['monte_carlo']
            summary_text += f"Monte Carlo Results:\n"
            summary_text += f"Mean NPV: ${mc_stats['mean']/1e6:.1f} Million\n"
            summary_text += f"Std Dev: ${mc_stats['std']/1e6:.1f} Million\n"
            summary_text += f"P10: ${mc_stats['percentiles']['P10']/1e6:.1f} Million\n"
            summary_text += f"P50: ${mc_stats['percentiles']['P50']/1e6:.1f} Million\n"
            summary_text += f"P90: ${mc_stats['percentiles']['P90']/1e6:.1f} Million\n\n"
        
        if 'period_analysis' in self.results:
            total_blocks = sum(self.results['period_analysis'][t]['blocks_mined'] 
                             for t in self.results['period_analysis'])
            total_mass = sum(self.results['period_analysis'][t]['total_mass'] 
                           for t in self.results['period_analysis'])
            
            summary_text += f"Mining Summary:\n"
            summary_text += f"Total Blocks Mined: {total_blocks:,}\n"
            summary_text += f"Total Mass: {total_mass/1e6:.2f} Million tons\n"
            summary_text += f"Mining Periods: {self.params['num_periods']}\n"
            summary_text += f"Scenarios Analyzed: {self.params['num_scenarios']}"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # 9. Sensitivity Analysis (placeholder)
        ax9 = plt.subplot(3, 3, 9)
        
        # Simple sensitivity analysis for gold price
        gold_prices = np.linspace(1000, 1400, 11)
        npv_sensitivity = []
        
        base_npv = 1500  # Million $ (placeholder)
        for price in gold_prices:
            price_factor = price / self.params['gold_price']
            npv_sensitivity.append(base_npv * price_factor)
        
        ax9.plot(gold_prices, npv_sensitivity, marker='o', linewidth=2, color='green')
        ax9.axhline(y=base_npv, color='red', linestyle='--', alpha=0.7, 
                   label=f'Base NPV: ${base_npv}M')
        ax9.axvline(x=self.params['gold_price'], color='blue', linestyle='--', 
                   alpha=0.7, label=f'Base Price: ${self.params["gold_price"]}/oz')
        
        ax9.set_xlabel('Gold Price ($/oz)')
        ax9.set_ylabel('NPV (Million $)')
        ax9.set_title('Sensitivity to Gold Price')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mine_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MINE OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        if 'monte_carlo' in self.results:
            mc_stats = self.results['monte_carlo']
            print(f"\nMonte Carlo Analysis ({len(mc_stats['simulations'])} simulations):")
            print(f"  Mean NPV: ${mc_stats['mean']/1e6:.2f} Million")
            print(f"  Standard Deviation: ${mc_stats['std']/1e6:.2f} Million")
            print(f"  P10 (Optimistic): ${mc_stats['percentiles']['P10']/1e6:.2f} Million")
            print(f"  P50 (Expected): ${mc_stats['percentiles']['P50']/1e6:.2f} Million")
            print(f"  P90 (Conservative): ${mc_stats['percentiles']['P90']/1e6:.2f} Million")
            
            # Calculate Value at Risk
            var_95 = np.percentile(mc_stats['simulations'], 5)
            print(f"  VaR (95%): ${var_95/1e6:.2f} Million")
        
        if 'period_analysis' in self.results:
            print(f"\nMining Schedule Summary:")
            for t in range(self.params['num_periods']):
                if t in self.results['period_analysis']:
                    stats = self.results['period_analysis'][t]
                    print(f"  Period {t+1}: {stats['blocks_mined']:,} blocks, "
                          f"{stats['total_mass']/1e6:.2f}M tons "
                          f"({stats['capacity_utilization']*100:.1f}% capacity)")

# Main execution function
def run_mine_optimization():
    """
    Main function to run the complete mine optimization analysis
    """
    print("Starting Mine Optimization with Dynamic Uncertainty Analysis")
    print("=" * 60)
    
    # Initialize the DSS
    dss = MineOptimizationDSS()
    
    # Generate synthetic data
    dss.generate_synthetic_data()
    
    # Calculate uncertainty factors
    dss.calculate_dynamic_uncertainty_factors()
    
    # Solve optimization
    mining_schedule = dss.solve_first_stage()
    
    if mining_schedule:
        second_stage_results = dss.solve_second_stage(mining_schedule)
        
        # Perform Monte Carlo simulation
        dss.monte_carlo_simulation(num_simulations=1000)
        
        # Analyze results
        dss.analyze_results()
        
        # Create visualizations
        dss.create_visualizations()
        
        return dss
    else:
        print("Optimization failed")
        return None

# Example usage
if __name__ == "__main__":
    # Run the complete analysis
    dss_system = run_mine_optimization()
    
    if dss_system:
        print("\nAnalysis completed successfully!")
        print("Results saved to 'mine_optimization_results.png'")
    else:
        print("Analysis failed!")