"""
Enhanced Dantzig-Wolfe Decomposition with VAE Integration
Implements Algorithm 1 from manuscript with complete mining sequences
"""

import numpy as np
import torch
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional
import pulp
from dataclasses import dataclass

@dataclass
class EnhancedColumnGenerationConfig:
    """Configuration for Algorithm 1"""
    max_iterations: int = 100
    reduced_cost_threshold: float = -1e-4
    scenario_samples: int = 50  # Dynamic VAE scenarios
    spatial_decay_factor: float = 0.95
    temporal_decay_factor: float = 0.92
    geological_weight: float = 0.15
    convergence_tolerance: float = 1e-4

class EnhancedDantzigWolfe:
    """
    Algorithm 1: Enhanced Dantzig-Wolfe with VAE-conditioned column generation
    Treats complete mining sequences as columns
    """
    
    def __init__(self, vae_model, rl_agents, config=None):
        self.vae_model = vae_model
        self.rl_agents = rl_agents
        self.config = config or EnhancedColumnGenerationConfig()
        self.columns = []
        self.dual_values = {}
        self.iteration = 0
        self.spatial_correlation_cache = {}
    
    def compute_enhanced_spatial_uncertainty(self, blocks, scenario, period):
        """
        Compute σ_enhanced(s,t) = f_spatial × φ_temporal × ψ_geological
        As per Equation 8 in manuscript
        """
        # Spatial component using Moran's I
        moran_i = self._calculate_morans_i(blocks, scenario)
        local_variance = np.var(blocks[:, :3])
        f_spatial = (1 - moran_i) + local_variance
        
        # Temporal component
        phi_temporal = self.config.temporal_decay_factor ** period
        
        # Geological component
        psi_geological = self._calculate_geological_features(scenario)
        
        # Combined enhanced uncertainty
        sigma_enhanced = f_spatial * phi_temporal * psi_geological
        
        return sigma_enhanced
    
    def _calculate_morans_i(self, blocks, scenario):
        """Calculate Moran's I spatial autocorrelation (Section 4.5)"""
        grades = scenario.get('grades', blocks[:, 3])
        n = len(grades)
        
        # Spatial weights matrix
        coords = blocks[:, :3]
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        weights = np.where(distances > 0, 1.0 / (distances + 1e-6), 0)
        np.fill_diagonal(weights, 0)
        
        # Moran's I calculation
        mean_grade = np.mean(grades)
        deviations = grades - mean_grade
        
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(weights) * np.sum(deviations**2) / n
        
        moran_i = (n / np.sum(weights)) * (numerator / (denominator + 1e-10))
        return moran_i
    
    def _calculate_geological_features(self, scenario):
        """Calculate ψ_geological (Equation 10)"""
        alteration = scenario.get('alteration', 0.5)
        structure = scenario.get('structure', 0.3)
        intrusion_dist = scenario.get('intrusion_distance', 1.0)
        
        # Weights from RL Resource Agent
        weights = self.rl_agents['resource'].get_feature_weights()
        
        psi = (weights[0] * alteration + 
               weights[1] * structure + 
               weights[2] * (1.0 / (1.0 + intrusion_dist)))
        
        return max(0.1, psi)
    
    def generate_vae_scenarios(self, num_scenarios, dual_values):
        """
        Generate scenarios using VAE conditioned on dual values
        Phase 1 of Algorithm 1
        """
        scenarios = []
        
        with torch.no_grad():
            dual_tensor = torch.tensor(list(dual_values.values())[:10], dtype=torch.float32)
            
            # Generate n scenarios (50-200+)
            generated = self.vae_model.generate_scenarios(num_scenarios, dual_tensor)
            
            for i in range(num_scenarios):
                scenario = self._process_generated_scenario(generated[i])
                if self._validate_geological_constraints(scenario):
                    scenarios.append(scenario)
        
        return scenarios
    
    def solve_master_problem(self):
        """
        Phase 2 of Algorithm 1: Solve master problem
        Select optimal combination of mining sequences
        """
        if not self.columns:
            return None, {}
        
        prob = pulp.LpProblem("Master_Mining_Schedule", pulp.LpMaximize)
        
        # Decision variables: selection of columns
        lambda_vars = [
            pulp.LpVariable(f"lambda_{i}", lowBound=0, upBound=1)
            for i in range(len(self.columns))
        ]
        
        # Objective: maximize total NPV
        prob += pulp.lpSum([
            col['npv'] * lambda_vars[i] 
            for i, col in enumerate(self.columns)
        ])
        
        # Convexity constraint
        prob += pulp.lpSum(lambda_vars) == 1
        
        # Capacity constraints for each period
        for period in range(6):
            capacity_usage = pulp.lpSum([
                col['capacity_usage'][period] * lambda_vars[i]
                for i, col in enumerate(self.columns)
            ])
            prob += capacity_usage <= 6.5e6
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract dual values
        dual_values = {
            constraint.name: constraint.pi 
            for constraint in prob.constraints.values()
        }
        
        solution = np.array([var.value() for var in lambda_vars])
        
        return solution, dual_values
    
    def solve_pricing_subproblem(self, equipment_type, scenarios):
        """
        VAE-conditioned pricing subproblem
        Generate complete mining sequences robust to dynamic scenarios
        """
        # RL agent selects strategy
        strategy = self.rl_agents['schedule'].select_strategy(
            self.dual_values, equipment_type
        )
        
        best_sequence = None
        best_reduced_cost = float('inf')
        
        for scenario in scenarios:
            sequence = self._generate_complete_mining_sequence(
                equipment_type, scenario, strategy
            )
            
            reduced_cost = self._calculate_reduced_cost_with_uncertainty(
                sequence, scenario
            )
            
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_sequence = sequence
        
        if best_reduced_cost < self.config.reduced_cost_threshold:
            return best_sequence
        
        return None
    
    def _generate_complete_mining_sequence(self, equipment_type, scenario, strategy):
        """Generate complete extraction path spanning multiple periods"""
        sequence = {
            'equipment': equipment_type,
            'blocks': [],
            'periods': [],
            'modes': [],
            'npv': 0,
            'capacity_usage': np.zeros(6),
            'spatial_correlation': 0
        }
        
        # Strategy-based extraction ordering
        if strategy == 'highgrade':
            extraction_order = np.argsort(scenario['grades'])[::-1]
        elif strategy == 'risk_balanced':
            extraction_order = self._risk_balanced_sequencing(scenario)
        else:  # spatial
            extraction_order = self._spatial_clustering_sequencing(scenario)
        
        # Assign blocks to periods with constraints
        current_period = 0
        period_capacity = 0
        
        for block_idx in extraction_order:
            if current_period >= 6:
                break
            
            block_mass = 15375  # tons per block
            
            if period_capacity + block_mass > 6.5e6:
                current_period += 1
                period_capacity = 0
                if current_period >= 6:
                    break
            
            # Add to sequence
            sequence['blocks'].append(block_idx)
            sequence['periods'].append(current_period)
            
            # Select operational mode based on rock type
            mode = self._select_operational_mode(block_idx, scenario)
            sequence['modes'].append(mode)
            
            # Calculate NPV with enhanced uncertainty
            uncertainty = self.compute_enhanced_spatial_uncertainty(
                scenario['blocks'], scenario, current_period
            )
            
            grade = scenario['grades'][block_idx]
            block_npv = self._calculate_block_npv(
                grade, block_mass, mode, current_period, uncertainty
            )
            sequence['npv'] += block_npv
            
            # Update capacity
            sequence['capacity_usage'][current_period] += block_mass
            period_capacity += block_mass
        
        # Calculate spatial correlation for quality assessment
        sequence['spatial_correlation'] = self._calculate_sequence_continuity(
            sequence['blocks'], scenario
        )
        
        return sequence
    
    def _calculate_block_npv(self, grade, mass, mode, period, uncertainty):
        """Calculate NPV for a block with enhanced uncertainty"""
        # Mode-specific parameters
        if mode == 0:  # Mode A
            recovery = 0.83
            proc_cost = 21.4
        else:  # Mode B
            recovery = 0.83
            proc_cost = 24.9
        
        # Revenue and costs
        revenue = grade * mass * 1190 * recovery * uncertainty
        mining_cost = mass * 20.5
        processing_cost = mass * proc_cost
        
        # Discount
        discount = 1 / (1.08 ** period)
        
        return (revenue - mining_cost - processing_cost) * discount
    
    def optimize(self, initial_blocks, equipment_types):
        """
        Main optimization loop implementing Algorithm 1
        """
        # Initialize with greedy columns
        self._initialize_columns(initial_blocks, equipment_types)
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Phase 2: Solve master problem
            solution, self.dual_values = self.solve_master_problem()
            
            if solution is None:
                print("Master problem infeasible")
                break
            
            # VAE scenario generation (50-200 scenarios)
            scenarios = self.generate_vae_scenarios(
                self.config.scenario_samples, 
                self.dual_values
            )
            
            # Update spatial correlations
            for scenario in scenarios:
                scenario['spatial_correlation'] = self._calculate_morans_i(
                    initial_blocks, scenario
                )
            
            # RL-guided subproblem solving
            new_columns_added = False
            
            for equipment in equipment_types:
                self.rl_agents['schedule'].update_state(self.dual_values)
                
                new_column = self.solve_pricing_subproblem(equipment, scenarios)
                
                if new_column is not None:
                    quality = self._assess_column_quality(new_column, scenarios)
                    
                    if quality > 0.5:
                        self.columns.append(new_column)
                        new_columns_added = True
                        print(f"Iteration {iteration}: Added column NPV = ${new_column['npv']/1e6:.2f}M")
            
            # Phase 3: Adaptive column pool management
            self._manage_column_pool()
            
            # RL Learning Update
            reward = self._calculate_rl_reward(solution, scenarios)
            for agent in self.rl_agents.values():
                agent.update_policy(reward)
            
            if not new_columns_added:
                print("Convergence achieved")
                break
            
            # Adaptive parameter adjustment
            self.rl_agents['parameter'].adjust_parameters(iteration)
        
        # Final solution
        final_solution, _ = self.solve_master_problem()
        return self._construct_final_schedule(final_solution)
    
    def _validate_geological_constraints(self, scenario):
        """Validate geological realism"""
        continuity_loss = self._calculate_continuity_loss(scenario)
        grade_tonnage_valid = self._check_grade_tonnage_curve(scenario)
        
        return continuity_loss < 0.1 and grade_tonnage_valid
    
    def _calculate_continuity_loss(self, scenario):
        """Assess spatial continuity"""
        if 'blocks' not in scenario:
            return 0.0
        
        grades = scenario.get('grades', [])
        if len(grades) < 2:
            return 0.0
        
        # Simple continuity metric
        grade_diff = np.diff(grades)
        return np.mean(np.abs(grade_diff)) / (np.mean(grades) + 1e-10)
    
    def _check_grade_tonnage_curve(self, scenario):
        """Validate grade-tonnage relationship"""
        grades = scenario.get('grades', [])
        if len(grades) == 0:
            return True
        
        # Higher grades should be less common
        high_grade_ratio = np.sum(np.array(grades) > 5.0) / len(grades)
        return high_grade_ratio < 0.1
    
    def _process_generated_scenario(self, generated_data):
        """Convert VAE output to scenario dictionary"""
        scenario = {
            'blocks': [],
            'grades': generated_data[:, 0] if len(generated_data.shape) > 1 else generated_data,
            'rock_types': (generated_data[:, 1] > 0.5).astype(int) if len(generated_data.shape) > 1 else np.zeros_like(generated_data),
            'alteration': float(np.mean(generated_data[:, 2])) if len(generated_data.shape) > 1 and generated_data.shape[1] > 2 else 0.5,
            'structure': float(np.mean(generated_data[:, 3])) if len(generated_data.shape) > 1 and generated_data.shape[1] > 3 else 0.3,
            'intrusion_distance': float(np.mean(generated_data[:, 4])) if len(generated_data.shape) > 1 and generated_data.shape[1] > 4 else 1.0,
            'uncertainty': {}
        }
        
        # Create block dictionaries
        for i in range(len(scenario['grades'])):
            block = {
                'index': i,
                'grade': float(scenario['grades'][i]),
                'mass': 15375.0,
                'rock_type': int(scenario['rock_types'][i]) if i < len(scenario['rock_types']) else 0
            }
            scenario['blocks'].append(block)
        
        return scenario
    
    def _risk_balanced_sequencing(self, scenario):
        """Risk-balanced extraction considering uncertainty"""
        grades = scenario['grades']
        uncertainties = np.random.uniform(0.8, 1.2, len(grades))
        risk_adjusted = grades / uncertainties
        return np.argsort(risk_adjusted)[::-1]
    
    def _spatial_clustering_sequencing(self, scenario):
        """Spatial clustering for minimized equipment movement"""
        # Simplified spatial ordering
        n_blocks = len(scenario['grades'])
        return np.arange(n_blocks)
    
    def _select_operational_mode(self, block_idx, scenario):
        """Select mode based on rock type"""
        if block_idx < len(scenario['rock_types']):
            rock_type = scenario['rock_types'][block_idx]
            return 0 if rock_type == 0 else 1
        return 0
    
    def _calculate_sequence_continuity(self, blocks, scenario):
        """Calculate spatial continuity score"""
        if len(blocks) < 2:
            return 1.0
        
        # Simple continuity metric
        grade_changes = []
        for i in range(len(blocks) - 1):
            if blocks[i] < len(scenario['grades']) and blocks[i+1] < len(scenario['grades']):
                grade_diff = abs(scenario['grades'][blocks[i]] - scenario['grades'][blocks[i+1]])
                grade_changes.append(grade_diff)
        
        if grade_changes:
            avg_change = np.mean(grade_changes)
            continuity = 1.0 / (1.0 + avg_change)
            return continuity
        return 1.0
    
    def _calculate_reduced_cost_with_uncertainty(self, sequence, scenario):
        """Calculate reduced cost with spatial uncertainty"""
        if not sequence:
            return float('inf')
        
        # Direct cost
        direct_cost = len(sequence['blocks']) * 15375 * 20.5  # Simplified
        
        # Revenue with uncertainty
        revenue = sequence['npv']
        
        # Dual contribution
        dual_contribution = 0
        if self.dual_values:
            for period in range(6):
                key = f'capacity_{period}'
                if key in self.dual_values:
                    dual_contribution += (
                        self.dual_values[key] * 
                        sequence['capacity_usage'][period]
                    )
        
        return direct_cost - revenue - dual_contribution
    
    def _assess_column_quality(self, column, scenarios):
        """Assess column quality with NPV and geological consistency"""
        if not column:
            return 0.0
        
        npv_score = column['npv'] / 1e9  # Normalize
        continuity_score = column.get('spatial_correlation', 0.5)
        
        # Combined quality
        quality = 0.7 * npv_score + 0.3 * continuity_score
        return min(1.0, max(0.0, quality))
    
    def _manage_column_pool(self):
        """Dynamic column pool management"""
        if len(self.columns) > 1000:
            # Keep best columns based on quality
            qualities = [
                self._assess_column_quality(col, [])
                for col in self.columns
            ]
            
            threshold = np.percentile(qualities, 20)
            self.columns = [
                col for col, q in zip(self.columns, qualities) 
                if q > threshold
            ]
    
    def _calculate_rl_reward(self, solution, scenarios):
        """Calculate reward for RL agents (Equation 3)"""
        if solution is None:
            return 0.0
        
        # NPV improvement
        current_npv = sum([
            self.columns[i]['npv'] * solution[i] 
            for i in range(len(solution)) if solution[i] > 0
        ])
        npv_improvement = current_npv - getattr(self, 'previous_npv', 0)
        self.previous_npv = current_npv
        
        # Constraint satisfaction
        constraint_satisfaction = 1.0
        
        # Efficiency
        efficiency = 1.0 / (1.0 + self.iteration * 0.01)
        
        # Risk
        npvs = [s.get('expected_npv', current_npv) for s in scenarios]
        risk_penalty = np.std(npvs) / (np.mean(npvs) + 1e-10) if npvs else 0
        
        # Weighted reward
        reward = (0.4 * npv_improvement/1e6 + 
                 0.3 * constraint_satisfaction + 
                 0.2 * efficiency - 
                 0.1 * risk_penalty)
        
        return reward
    
    def _initialize_columns(self, blocks, equipment_types):
        """Initialize with greedy columns"""
        for equipment in equipment_types:
            column = {
                'equipment': equipment,
                'blocks': list(range(min(1000, len(blocks)))),
                'periods': [i // 200 for i in range(min(1000, len(blocks)))],
                'modes': [0] * min(1000, len(blocks)),
                'npv': 1e8,  # Placeholder
                'capacity_usage': np.array([200 * 15375] * 6),
                'spatial_correlation': 0.5
            }
            self.columns.append(column)
    
    def _construct_final_schedule(self, solution):
        """Construct final mining schedule"""
        if solution is None:
            return {'blocks': [], 'periods': [], 'modes': [], 'total_npv': 0}
        
        schedule = {
            'blocks': [],
            'periods': [],
            'modes': [],
            'total_npv': 0
        }
        
        for i, weight in enumerate(solution):
            if weight > 0.01:
                column = self.columns[i]
                fraction = int(weight * len(column['blocks']))
                
                schedule['blocks'].extend(column['blocks'][:fraction])
                schedule['periods'].extend(column['periods'][:fraction])
                schedule['modes'].extend(column['modes'][:fraction])
                schedule['total_npv'] += column['npv'] * weight
        
        return schedule