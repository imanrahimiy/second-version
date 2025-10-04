"""
Enhanced Dantzig-Wolfe Decomposition with VAE-conditioned column generation
Based on Algorithm 1 from the revised manuscript
Treats complete mining sequences as columns rather than individual block assignments
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pulp

@dataclass
class ColumnGenerationConfig:
    """Configuration for enhanced column generation"""
    max_iterations: int = 100
    reduced_cost_threshold: float = -1e-4
    scenario_samples: int = 50  # Dynamic VAE scenario generation
    spatial_decay_factor: float = 0.95
    temporal_decay_factor: float = 0.92
    geological_weight: float = 0.15

class EnhancedDantzigWolfe:
    """
    Enhanced Dantzig-Wolfe decomposition with VAE-conditioned column generation
    Implements complete mining sequence optimization from Algorithm 1
    """
    
    def __init__(self, vae_model, rl_agents, config: ColumnGenerationConfig):
        self.vae_model = vae_model
        self.rl_agents = rl_agents  # Multi-agent RL framework
        self.config = config
        self.columns = []  # Store complete mining sequences
        self.dual_values = {}
        self.iteration = 0
        
    def compute_enhanced_spatial_uncertainty(self, blocks: np.ndarray, 
                                            scenario: Dict, period: int) -> float:
        """
        Compute enhanced spatial uncertainty using Moran's I and geological features
        σ_enhanced(s,t) = f_spatial × φ_temporal × ψ_geological
        """
        # Spatial component using Moran's I
        moran_i = self._calculate_morans_i(blocks, scenario)
        local_variance = np.var(blocks[:, :3])  # Spatial coordinates variance
        f_spatial = (1 - moran_i) + local_variance
        
        # Temporal component
        phi_temporal = self.config.temporal_decay_factor ** period
        
        # Geological component (weighted features)
        psi_geological = self._calculate_geological_features(scenario)
        
        # Combined enhanced uncertainty
        sigma_enhanced = f_spatial * phi_temporal * psi_geological
        
        return sigma_enhanced
    
    def _calculate_morans_i(self, blocks: np.ndarray, scenario: Dict) -> float:
        """
        Calculate Moran's I spatial autocorrelation statistic
        """
        grades = scenario.get('grades', blocks[:, 3])
        n = len(grades)
        
        # Spatial weights matrix (inverse distance)
        coords = blocks[:, :3]
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        weights = np.where(distances > 0, 1.0 / distances, 0)
        np.fill_diagonal(weights, 0)
        
        # Calculate Moran's I
        mean_grade = np.mean(grades)
        deviations = grades - mean_grade
        
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(weights) * np.sum(deviations**2) / n
        
        moran_i = (n / np.sum(weights)) * (numerator / denominator) if denominator > 0 else 0
        
        return moran_i
    
    def _calculate_geological_features(self, scenario: Dict) -> float:
        """
        Calculate geological feature integration
        ψ_geological = w₁·alteration + w₂·structure + w₃·distance_to_intrusion
        """
        alteration_intensity = scenario.get('alteration', 0.5)
        structural_density = scenario.get('structure', 0.3)
        distance_to_intrusion = scenario.get('intrusion_distance', 1.0)
        
        # Learned weights from RL agent
        weights = self.rl_agents['resource'].get_feature_weights()
        
        psi = (weights[0] * alteration_intensity + 
               weights[1] * structural_density + 
               weights[2] * (1.0 / (1.0 + distance_to_intrusion)))
        
        return max(0.1, psi)  # Ensure non-zero
    
    def generate_vae_scenarios(self, num_scenarios: int, dual_values: Dict) -> List[Dict]:
        """
        Generate geological scenarios using VAE conditioned on dual values
        """
        scenarios = []
        
        with torch.no_grad():
            # Sample from VAE latent space conditioned on economic signals
            dual_tensor = torch.tensor(list(dual_values.values()), dtype=torch.float32)
            
            for _ in range(num_scenarios):
                # Sample latent vector
                z = torch.randn(1, self.vae_model.latent_dim)
                
                # Condition on dual values (economic signals)
                z_conditioned = self.vae_model.condition_on_duals(z, dual_tensor)
                
                # Decode to geological scenario
                scenario = self.vae_model.decode(z_conditioned)
                
                # Validate geological constraints
                if self._validate_geological_constraints(scenario):
                    scenarios.append(self._tensor_to_scenario(scenario))
        
        return scenarios
    
    def _validate_geological_constraints(self, scenario: torch.Tensor) -> bool:
        """
        Validate geological realism constraints
        """
        # Check spatial continuity
        continuity_loss = self.vae_model.calculate_continuity_loss(scenario)
        
        # Check grade-tonnage relationships
        grade_tonnage_valid = self._check_grade_tonnage_curve(scenario)
        
        return continuity_loss < 0.1 and grade_tonnage_valid
    
    def _check_grade_tonnage_curve(self, scenario: torch.Tensor) -> bool:
        """Validate grade-tonnage relationships"""
        grades = scenario[:, 3].numpy() if isinstance(scenario, torch.Tensor) else scenario[:, 3]
        tonnages = scenario[:, 4].numpy() if isinstance(scenario, torch.Tensor) else scenario[:, 4]
        
        # Simple validation: higher grades should have lower tonnages (generally)
        correlation = np.corrcoef(grades, tonnages)[0, 1]
        return correlation < 0.3  # Weak or negative correlation expected
    
    def solve_master_problem(self) -> Tuple[np.ndarray, Dict]:
        """
        Solve master problem to select optimal combination of mining sequences
        """
        if not self.columns:
            return None, {}
        
        # Create LP for master problem
        prob = pulp.LpProblem("Master_Mining_Schedule", pulp.LpMaximize)
        
        # Decision variables: selection of columns (mining sequences)
        lambda_vars = []
        for i, col in enumerate(self.columns):
            var = pulp.LpVariable(f"lambda_{i}", lowBound=0, upBound=1)
            lambda_vars.append(var)
        
        # Objective: maximize total NPV
        prob += pulp.lpSum([col['npv'] * lambda_vars[i] 
                           for i, col in enumerate(self.columns)])
        
        # Convexity constraint: sum of lambdas = 1
        prob += pulp.lpSum(lambda_vars) == 1
        
        # Capacity constraints for each period
        for period in range(6):  # Assuming 6 periods
            capacity_usage = pulp.lpSum([
                col['capacity_usage'][period] * lambda_vars[i]
                for i, col in enumerate(self.columns)
            ])
            prob += capacity_usage <= 6.5e6  # Capacity limit
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract dual values
        dual_values = {}
        for constraint in prob.constraints.values():
            dual_values[constraint.name] = constraint.pi
        
        # Get solution
        solution = np.array([var.value() for var in lambda_vars])
        
        return solution, dual_values
    
    def solve_pricing_subproblem(self, equipment_type: str, 
                                scenarios: List[Dict]) -> Optional[Dict]:
        """
        Solve pricing subproblem to generate new column (complete mining sequence)
        VAE-conditioned with enhanced spatial uncertainty
        """
        # RL agent selects strategy based on dual values
        strategy = self.rl_agents['schedule'].select_strategy(
            self.dual_values, equipment_type
        )
        
        best_sequence = None
        best_reduced_cost = float('inf')
        
        for scenario in scenarios:
            # Generate mining sequence for this equipment type
            sequence = self._generate_mining_sequence(
                equipment_type, scenario, strategy
            )
            
            # Calculate reduced cost with enhanced uncertainty
            reduced_cost = self._calculate_reduced_cost(sequence, scenario)
            
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_sequence = sequence
        
        # Only return if negative reduced cost (beneficial)
        if best_reduced_cost < self.config.reduced_cost_threshold:
            return best_sequence
        
        return None
    
    def _generate_mining_sequence(self, equipment_type: str, 
                                 scenario: Dict, strategy: str) -> Dict:
        """
        Generate complete mining sequence spanning multiple periods
        """
        sequence = {
            'equipment': equipment_type,
            'blocks': [],
            'periods': [],
            'modes': [],
            'npv': 0,
            'capacity_usage': np.zeros(6)
        }
        
        # Use strategy to determine extraction order
        if strategy == 'highgrade':
            extraction_order = self._highgrade_sequencing(scenario)
        elif strategy == 'risk_balanced':
            extraction_order = self._risk_balanced_sequencing(scenario)
        else:
            extraction_order = self._spatial_clustering_sequencing(scenario)
        
        # Assign blocks to periods respecting constraints
        current_period = 0
        period_capacity = 0
        
        for block_idx in extraction_order:
            block = scenario['blocks'][block_idx]
            
            # Check if need to move to next period
            if period_capacity + block['mass'] > 6.5e6:
                current_period += 1
                period_capacity = 0
                if current_period >= 6:
                    break
            
            # Add to sequence
            sequence['blocks'].append(block_idx)
            sequence['periods'].append(current_period)
            sequence['modes'].append(self._select_operational_mode(block, scenario))
            
            # Update NPV with enhanced uncertainty
            uncertainty = self.compute_enhanced_spatial_uncertainty(
                scenario['blocks'], scenario, current_period
            )
            block_npv = block['value'] * uncertainty / (1.08 ** current_period)
            sequence['npv'] += block_npv
            
            # Update capacity usage
            sequence['capacity_usage'][current_period] += block['mass']
            period_capacity += block['mass']
        
        return sequence
    
    def _calculate_reduced_cost(self, sequence: Dict, scenario: Dict) -> float:
        """
        Calculate reduced cost for column with enhanced uncertainty
        """
        # Direct cost
        direct_cost = sum([
            scenario['blocks'][b]['mining_cost'] / (1.08 ** p)
            for b, p in zip(sequence['blocks'], sequence['periods'])
        ])
        
        # Revenue with uncertainty
        revenue = sequence['npv']
        
        # Dual prices contribution
        dual_contribution = 0
        if self.dual_values:
            for period in range(6):
                if f'capacity_{period}' in self.dual_values:
                    dual_contribution += (
                        self.dual_values[f'capacity_{period}'] * 
                        sequence['capacity_usage'][period]
                    )
        
        reduced_cost = direct_cost - revenue - dual_contribution
        
        return reduced_cost
    
    def optimize(self, initial_blocks: np.ndarray, 
                equipment_types: List[str]) -> Dict:
        """
        Main optimization loop implementing Algorithm 1 from manuscript
        """
        # Initialize with greedy heuristic columns
        self._initialize_columns(initial_blocks, equipment_types)
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Phase 2: Solve master problem
            solution, self.dual_values = self.solve_master_problem()
            
            if solution is None:
                print("Master problem infeasible")
                break
            
            # VAE scenario generation conditioned on dual values
            scenarios = self.generate_vae_scenarios(
                self.config.scenario_samples, 
                self.dual_values
            )
            
            # Spatial uncertainty update using Moran's I
            for scenario in scenarios:
                scenario['spatial_correlation'] = self._calculate_morans_i(
                    initial_blocks, scenario
                )
            
            # RL-guided subproblem solving for each equipment type
            new_columns_added = False
            
            for equipment in equipment_types:
                # Agent selects subproblem strategy
                self.rl_agents['schedule'].update_state(self.dual_values)
                
                # Solve pricing subproblem with VAE scenarios
                new_column = self.solve_pricing_subproblem(equipment, scenarios)
                
                if new_column is not None:
                    # Validate geological consistency
                    quality = self._assess_column_quality(new_column, scenarios)
                    
                    if quality > 0.5:  # Quality threshold
                        self.columns.append(new_column)
                        new_columns_added = True
                        print(f"Iteration {iteration}: Added column with NPV = "
                              f"${new_column['npv']/1e6:.2f}M")
            
            # Phase 3: Adaptive column pool management
            self._manage_column_pool()
            
            # RL Learning Update
            reward = self._calculate_rl_reward(solution, scenarios)
            for agent_name, agent in self.rl_agents.items():
                agent.update_policy(reward)
            
            if not new_columns_added:
                print(f"No improving columns found. Optimization complete.")
                break
            
            # Adjust parameters based on convergence
            self.rl_agents['parameter'].adjust_parameters(iteration)
        
        # Final solution
        final_solution, _ = self.solve_master_problem()
        return self._construct_final_schedule(final_solution)
    
    def _assess_column_quality(self, column: Dict, scenarios: List[Dict]) -> float:
        """
        Assess column quality combining NPV and geological consistency
        """
        npv_score = column['npv'] / 1e9  # Normalize to billions
        
        # Check geological continuity
        blocks = column['blocks']
        continuity_score = self._calculate_sequence_continuity(blocks, scenarios[0])
        
        # Combined quality metric
        quality = 0.7 * npv_score + 0.3 * continuity_score
        
        return quality
    
    def _manage_column_pool(self):
        """
        Dynamic column pool management - remove outdated, keep high-quality
        """
        if len(self.columns) > 1000:  # Maximum pool size
            # Calculate quality scores for all columns
            qualities = [
                self._assess_column_quality(col, [{'blocks': np.random.randn(100, 5)}])
                for col in self.columns
            ]
            
            # Keep top 80%
            threshold = np.percentile(qualities, 20)
            self.columns = [
                col for col, q in zip(self.columns, qualities) 
                if q > threshold
            ]
    
    def _calculate_rl_reward(self, solution: np.ndarray, 
                            scenarios: List[Dict]) -> float:
        """
        Calculate reinforcement learning reward
        R(t) = α·NPV_improvement + β·Constraint_satisfaction + γ·Efficiency - δ·Risk
        """
        # NPV improvement
        current_npv = sum([
            self.columns[i]['npv'] * solution[i] 
            for i in range(len(solution)) if solution[i] > 0
        ])
        npv_improvement = current_npv - getattr(self, 'previous_npv', 0)
        self.previous_npv = current_npv
        
        # Constraint satisfaction (0 to 1)
        constraint_satisfaction = 1.0  # Assuming master problem enforces
        
        # Computational efficiency (based on iteration count)
        efficiency = 1.0 / (1.0 + self.iteration * 0.01)
        
        # Risk penalty (variance across scenarios)
        npvs = [s.get('expected_npv', current_npv) for s in scenarios]
        risk_penalty = np.std(npvs) / np.mean(npvs) if np.mean(npvs) > 0 else 0
        
        # Weighted reward
        reward = (0.4 * npv_improvement/1e6 + 
                 0.3 * constraint_satisfaction + 
                 0.2 * efficiency - 
                 0.1 * risk_penalty)
        
        return reward
    
    def _initialize_columns(self, blocks: np.ndarray, equipment_types: List[str]):
        """Initialize with greedy heuristic columns"""
        for equipment in equipment_types:
            # Create initial column using value density ranking
            initial_column = self._create_greedy_column(blocks, equipment)
            self.columns.append(initial_column)
    
    def _create_greedy_column(self, blocks: np.ndarray, equipment: str) -> Dict:
        """Create initial column using greedy heuristic"""
        # Implement greedy value density approach
        value_density = blocks[:, 3] / blocks[:, 4]  # grade/mass
        sorted_indices = np.argsort(value_density)[::-1]
        
        column = {
            'equipment': equipment,
            'blocks': sorted_indices[:1000].tolist(),  # Top 1000 blocks
            'periods': [i // 200 for i in range(1000)],  # Distribute across periods
            'modes': [0] * 1000,  # Default mode
            'npv': np.sum(blocks[sorted_indices[:1000], 3] * 1000),  # Simplified NPV
            'capacity_usage': np.array([200 * blocks[0, 4]] * 5 + [0])
        }
        
        return column
    
    def _select_operational_mode(self, block: Dict, scenario: Dict) -> int:
        """Select operational mode based on rock type and recovery optimization"""
        rock_type = block.get('rock_type', 0)
        if rock_type == 0:  # Diorite Porphyry
            return 0  # Mode A
        else:  # Silicified Breccia
            return 1  # Mode B
    
    def _highgrade_sequencing(self, scenario: Dict) -> List[int]:
        """High-grade first extraction strategy"""
        grades = [b['grade'] for b in scenario['blocks']]
        return np.argsort(grades)[::-1].tolist()
    
    def _risk_balanced_sequencing(self, scenario: Dict) -> List[int]:
        """Risk-balanced extraction considering uncertainty"""
        # Balance grade with uncertainty
        scores = []
        for i, block in enumerate(scenario['blocks']):
            uncertainty = scenario.get('uncertainty', {}).get(i, 1.0)
            score = block['grade'] / uncertainty
            scores.append(score)
        return np.argsort(scores)[::-1].tolist()
    
    def _spatial_clustering_sequencing(self, scenario: Dict) -> List[int]:
        """Spatial clustering to minimize equipment movement"""
        from sklearn.cluster import DBSCAN
        
        coords = np.array([b['coords'] for b in scenario['blocks']])
        clustering = DBSCAN(eps=50, min_samples=5).fit(coords)
        
        # Extract blocks cluster by cluster
        sequence = []
        for cluster_id in range(max(clustering.labels_) + 1):
            cluster_blocks = np.where(clustering.labels_ == cluster_id)[0]
            sequence.extend(cluster_blocks.tolist())
        
        return sequence
    
    def _calculate_sequence_continuity(self, blocks: List[int], scenario: Dict) -> float:
        """Calculate spatial continuity score for mining sequence"""
        if len(blocks) < 2:
            return 1.0
        
        total_distance = 0
        for i in range(len(blocks) - 1):
            coord1 = scenario['blocks'][blocks[i]]['coords']
            coord2 = scenario['blocks'][blocks[i+1]]['coords']
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            total_distance += distance
        
        # Normalize (lower distance = higher continuity)
        avg_distance = total_distance / (len(blocks) - 1)
        continuity = 1.0 / (1.0 + avg_distance / 100)  # Normalize by 100m
        
        return continuity
    
    def _construct_final_schedule(self, solution: np.ndarray) -> Dict:
        """Construct final mining schedule from selected columns"""
        schedule = {
            'blocks': [],
            'periods': [],
            'modes': [],
            'total_npv': 0
        }
        
        # Combine selected columns weighted by solution
        for i, weight in enumerate(solution):
            if weight > 0.01:  # Threshold for selection
                column = self.columns[i]
                fraction = int(weight * len(column['blocks']))
                
                schedule['blocks'].extend(column['blocks'][:fraction])
                schedule['periods'].extend(column['periods'][:fraction])
                schedule['modes'].extend(column['modes'][:fraction])
                schedule['total_npv'] += column['npv'] * weight
        
        return schedule
    
    def _tensor_to_scenario(self, tensor: torch.Tensor) -> Dict:
        """Convert tensor output from VAE to scenario dictionary"""
        scenario = {
            'blocks': [],
            'grades': tensor[:, 3].numpy(),
            'uncertainty': {},
            'alteration': float(torch.mean(tensor[:, 5])),
            'structure': float(torch.mean(tensor[:, 6])),
            'intrusion_distance': float(torch.mean(tensor[:, 7]))
        }
        
        for i in range(len(tensor)):
            block = {
                'coords': tensor[i, :3].numpy(),
                'grade': float(tensor[i, 3]),
                'mass': float(tensor[i, 4]),
                'value': float(tensor[i, 3]) * 1190 * 0.83,  # grade * price * recovery
                'mining_cost': 20.5 * float(tensor[i, 4]),
                'rock_type': int(tensor[i, 8]) if tensor.shape[1] > 8 else 0
            }
            scenario['blocks'].append(block)
        
        return scenario