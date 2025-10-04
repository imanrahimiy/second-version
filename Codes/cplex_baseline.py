"""
CPLEX Baseline Comparison for Mining Optimization
Implements MILP formulation for benchmarking against the AI-enhanced approach
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
try:
    import cplex
    from cplex.exceptions import CplexError
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False
    print("CPLEX not available. Using PuLP as fallback.")
    import pulp

@dataclass
class CPLEXConfig:
    """Configuration for CPLEX solver"""
    time_limit: int = 3600  # 1 hour
    mip_gap: float = 0.01  # 1% optimality gap
    threads: int = 8
    emphasis: str = 'balanced'  # 'balanced', 'feasibility', 'optimality'
    node_limit: int = 10000

class CPLEXMiningOptimizer:
    """
    CPLEX-based mining optimization for baseline comparison
    Implements standard MILP formulation without AI enhancements
    """
    
    def __init__(self, config: CPLEXConfig):
        self.config = config
        self.problem = None
        self.solution = None
        self.solve_time = 0
        
    def create_milp_model(self, blocks: np.ndarray, 
                         scenarios: List[Dict],
                         periods: int = 6) -> None:
        """
        Create MILP model for open-pit mining scheduling
        Standard formulation without VAE or RL enhancements
        """
        if CPLEX_AVAILABLE:
            self._create_cplex_model(blocks, scenarios, periods)
        else:
            self._create_pulp_model(blocks, scenarios, periods)
    
    def _create_cplex_model(self, blocks: np.ndarray, 
                           scenarios: List[Dict],
                           periods: int) -> None:
        """
        Create CPLEX model for mining optimization
        """
        self.problem = cplex.Cplex()
        self.problem.objective.set_sense(self.problem.objective.sense.maximize)
        
        num_blocks = len(blocks)
        num_scenarios = len(scenarios)
        
        # Decision variables: x[b,t] = 1 if block b is mined in period t
        var_names = []
        obj_coeffs = []
        
        for b in range(num_blocks):
            for t in range(periods):
                var_name = f"x_{b}_{t}"
                var_names.append(var_name)
                
                # Calculate NPV coefficient (average across scenarios)
                npv = 0
                for s, scenario in enumerate(scenarios):
                    grade = scenario['grades'][b] if 'grades' in scenario else blocks[b, 3]
                    mass = blocks[b, 4]
                    revenue = grade * mass * 1190 * 0.83  # grade * mass * price * recovery
                    cost = mass * 20.5
                    discounted = (revenue - cost) / (1.08 ** t)
                    npv += discounted / num_scenarios
                
                obj_coeffs.append(npv)
        
        # Add variables to model
        self.problem.variables.add(
            names=var_names,
            obj=obj_coeffs,
            types=[self.problem.variables.type.binary] * len(var_names)
        )
        
        # Constraints
        self._add_assignment_constraints(num_blocks, periods)
        self._add_precedence_constraints(blocks, num_blocks, periods)
        self._add_capacity_constraints(blocks, num_blocks, periods)
        
        # Set CPLEX parameters
        self.problem.parameters.timelimit.set(self.config.time_limit)
        self.problem.parameters.mip.tolerances.mipgap.set(self.config.mip_gap)
        self.problem.parameters.threads.set(self.config.threads)
        
        if self.config.emphasis == 'feasibility':
            self.problem.parameters.emphasis.mip.set(1)
        elif self.config.emphasis == 'optimality':
            self.problem.parameters.emphasis.mip.set(2)
    
    def _add_assignment_constraints(self, num_blocks: int, periods: int) -> None:
        """
        Each block can be mined at most once
        """
        for b in range(num_blocks):
            constraint_vars = [f"x_{b}_{t}" for t in range(periods)]
            constraint_coeffs = [1.0] * periods
            
            self.problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(constraint_vars, constraint_coeffs)],
                senses=['L'],
                rhs=[1.0],
                names=[f"assignment_{b}"]
            )
    
    def _add_precedence_constraints(self, blocks: np.ndarray, 
                                   num_blocks: int, periods: int) -> None:
        """
        Block precedence constraints based on spatial relationships
        """
        for b in range(num_blocks):
            # Find predecessor blocks (simplified: blocks above current)
            predecessors = self._find_predecessors(b, blocks)
            
            for pred in predecessors:
                for t in range(periods):
                    # If block b is mined in period t, predecessor must be mined by t
                    constraint_vars = [f"x_{b}_{t}"]
                    constraint_coeffs = [1.0]
                    
                    for tau in range(t + 1):
                        constraint_vars.append(f"x_{pred}_{tau}")
                        constraint_coeffs.append(-1.0)
                    
                    self.problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(constraint_vars, constraint_coeffs)],
                        senses=['L'],
                        rhs=[0.0],
                        names=[f"precedence_{b}_{pred}_{t}"]
                    )
    
    def _add_capacity_constraints(self, blocks: np.ndarray, 
                                 num_blocks: int, periods: int) -> None:
        """
        Mining capacity constraints per period
        """
        capacity_limit = 6.5e6  # tons per period
        
        for t in range(periods):
            constraint_vars = []
            constraint_coeffs = []
            
            for b in range(num_blocks):
                constraint_vars.append(f"x_{b}_{t}")
                constraint_coeffs.append(blocks[b, 4])  # mass
            
            self.problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(constraint_vars, constraint_coeffs)],
                senses=['L'],
                rhs=[capacity_limit],
                names=[f"capacity_{t}"]
            )
    
    def _create_pulp_model(self, blocks: np.ndarray, 
                          scenarios: List[Dict],
                          periods: int) -> None:
        """
        Create PuLP model as fallback when CPLEX not available
        """
        self.problem = pulp.LpProblem("Mining_Schedule", pulp.LpMaximize)
        
        num_blocks = len(blocks)
        num_scenarios = len(scenarios)
        
        # Decision variables
        x = {}
        for b in range(num_blocks):
            for t in range(periods):
                x[b, t] = pulp.LpVariable(f"x_{b}_{t}", cat='Binary')
        
        # Objective function
        obj = 0
        for b in range(num_blocks):
            for t in range(periods):
                npv = 0
                for scenario in scenarios:
                    grade = scenario.get('grades', [blocks[b, 3]])[b] if b < len(scenario.get('grades', [])) else blocks[b, 3]
                    mass = blocks[b, 4]
                    revenue = grade * mass * 1190 * 0.83
                    cost = mass * 20.5
                    discounted = (revenue - cost) / (1.08 ** t)
                    npv += discounted / num_scenarios
                
                obj += npv * x[b, t]
        
        self.problem += obj
        
        # Assignment constraints
        for b in range(num_blocks):
            self.problem += pulp.lpSum(x[b, t] for t in range(periods)) <= 1
        
        # Precedence constraints
        for b in range(num_blocks):
            predecessors = self._find_predecessors(b, blocks)
            for pred in predecessors:
                for t in range(periods):
                    self.problem += x[b, t] <= pulp.lpSum(x[pred, tau] for tau in range(t + 1))
        
        # Capacity constraints
        capacity_limit = 6.5e6
        for t in range(periods):
            self.problem += pulp.lpSum(blocks[b, 4] * x[b, t] for b in range(num_blocks)) <= capacity_limit
    
    def solve(self) -> Dict:
        """
        Solve the MILP model and return solution
        """
        start_time = time.time()
        
        if CPLEX_AVAILABLE:
            try:
                self.problem.solve()
                self.solve_time = time.time() - start_time
                
                # Extract solution
                solution_values = self.problem.solution.get_values()
                objective_value = self.problem.solution.get_objective_value()
                gap = self.problem.solution.MIP.get_mip_relative_gap()
                
                self.solution = {
                    'objective': objective_value,
                    'gap': gap,
                    'solve_time': self.solve_time,
                    'status': self.problem.solution.get_status_string(),
                    'values': solution_values
                }
                
            except CplexError as e:
                print(f"CPLEX error: {e}")
                self.solution = {
                    'objective': 0,
                    'gap': 1.0,
                    'solve_time': self.solve_time,
                    'status': 'error',
                    'values': []
                }
        else:
            # Solve with PuLP
            self.problem.solve(pulp.PULP_CBC_CMD(timeLimit=self.config.time_limit))
            self.solve_time = time.time() - start_time
            
            self.solution = {
                'objective': pulp.value(self.problem.objective),
                'gap': 0.01,  # PuLP doesn't provide gap directly
                'solve_time': self.solve_time,
                'status': pulp.LpStatus[self.problem.status],
                'values': {v.name: v.varValue for v in self.problem.variables()}
            }
        
        return self.solution
    
    def compare_with_ai_solution(self, ai_solution: Dict) -> Dict:
        """
        Compare CPLEX solution with AI-enhanced solution
        """
        if self.solution is None:
            return {'error': 'No CPLEX solution available'}
        
        comparison = {
            'cplex_npv': self.solution['objective'],
            'ai_npv': ai_solution.get('total_npv', 0),
            'npv_improvement': ai_solution.get('total_npv', 0) - self.solution['objective'],
            'cplex_time': self.solution['solve_time'],
            'ai_time': ai_solution.get('solve_time', 0),
            'speedup': self.solution['solve_time'] / ai_solution.get('solve_time', 1),
            'cplex_gap': self.solution['gap'],
            'ai_gap': ai_solution.get('optimality_gap', 0)
        }
        
        # Calculate improvement percentage
        if self.solution['objective'] > 0:
            comparison['improvement_pct'] = (
                comparison['npv_improvement'] / self.solution['objective'] * 100
            )
        else:
            comparison['improvement_pct'] = 0
        
        return comparison
    
    def _find_predecessors(self, block_idx: int, blocks: np.ndarray) -> List[int]:
        """
        Find predecessor blocks (simplified spatial relationship)
        """
        predecessors = []
        current_pos = blocks[block_idx, :3]
        
        for i, other_block in enumerate(blocks):
            if i == block_idx:
                continue
            
            other_pos = other_block[:3]
            
            # Simple rule: blocks above current block are predecessors
            if (abs(other_pos[0] - current_pos[0]) < 30 and  # x proximity
                abs(other_pos[1] - current_pos[1]) < 30 and  # y proximity
                other_pos[2] > current_pos[2]):  # z above
                predecessors.append(i)
        
        return predecessors
    
    def run_scaling_experiments(self, problem_sizes: List[int]) -> Dict:
        """
        Run experiments with different problem sizes for benchmarking
        """
        results = {}
        
        for size in problem_sizes:
            print(f"Running CPLEX on {size} blocks...")
            
            # Generate synthetic data
            blocks = np.random.randn(size, 5)
            blocks[:, 3] = np.abs(blocks[:, 3]) * 2  # Grades
            blocks[:, 4] = np.abs(blocks[:, 4]) * 15000 + 10000  # Mass
            
            scenarios = [{'grades': np.abs(np.random.randn(size) * 2)} 
                        for _ in range(10)]
            
            # Create and solve model
            self.create_milp_model(blocks, scenarios)
            solution = self.solve()
            
            results[size] = {
                'objective': solution['objective'],
                'time': solution['solve_time'],
                'gap': solution['gap'],
                'status': solution['status']
            }
            
            # Check if reached time limit
            if solution['solve_time'] >= self.config.time_limit * 0.95:
                print(f"  Time limit reached at {size} blocks")
                break
        
        return results
    
    def export_solution_to_schedule(self) -> Dict:
        """
        Convert CPLEX solution to mining schedule format
        """
        if self.solution is None or self.solution['status'] == 'error':
            return {}
        
        schedule = {
            'blocks': [],
            'periods': [],
            'npv': self.solution['objective'],
            'solve_time': self.solution['solve_time'],
            'optimality_gap': self.solution['gap']
        }
        
        if CPLEX_AVAILABLE:
            # Parse CPLEX solution
            for i, val in enumerate(self.solution['values']):
                if val > 0.5:  # Binary variable is 1
                    # Extract block and period from variable name
                    parts = self.problem.variables.get_names(i).split('_')
                    if len(parts) == 3 and parts[0] == 'x':
                        block = int(parts[1])
                        period = int(parts[2])
                        schedule['blocks'].append(block)
                        schedule['periods'].append(period)
        else:
            # Parse PuLP solution
            for var_name, val in self.solution['values'].items():
                if val and val > 0.5:
                    parts = var_name.split('_')
                    if len(parts) == 3 and parts[0] == 'x':
                        block = int(parts[1])
                        period = int(parts[2])
                        schedule['blocks'].append(block)
                        schedule['periods'].append(period)
        
        return schedule


class BenchmarkRunner:
    """
    Run comprehensive benchmarking experiments
    """
    
    def __init__(self):
        self.cplex_config = CPLEXConfig()
        self.results = {}
    
    def run_comparison(self, ai_optimizer, test_cases: List[Dict]) -> Dict:
        """
        Run comparison between CPLEX and AI-enhanced approach
        """
        comparison_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\nRunning test case {i+1}/{len(test_cases)}")
            print(f"  Blocks: {test_case['num_blocks']}, "
                  f"Scenarios: {test_case['num_scenarios']}")
            
            # Prepare data
            blocks = test_case['blocks']
            scenarios = test_case['scenarios']
            
            # Run CPLEX
            print("  Running CPLEX...")
            cplex_opt = CPLEXMiningOptimizer(self.cplex_config)
            cplex_opt.create_milp_model(blocks, scenarios)
            cplex_solution = cplex_opt.solve()
            
            # Run AI-enhanced optimizer
            print("  Running AI-enhanced optimizer...")
            ai_solution = ai_optimizer.optimize(blocks, scenarios)
            
            # Compare results
            comparison = cplex_opt.compare_with_ai_solution(ai_solution)
            comparison['test_case'] = i
            comparison['num_blocks'] = test_case['num_blocks']
            comparison['num_scenarios'] = test_case['num_scenarios']
            
            comparison_results.append(comparison)
            
            # Print summary
            print(f"  CPLEX NPV: ${comparison['cplex_npv']/1e6:.2f}M")
            print(f"  AI NPV: ${comparison['ai_npv']/1e6:.2f}M")
            print(f"  Improvement: {comparison['improvement_pct']:.1f}%")
            print(f"  Speedup: {comparison['speedup']:.1f}x")
        
        return {
            'comparisons': comparison_results,
            'summary': self._summarize_results(comparison_results)
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """
        Summarize benchmark results
        """
        if not results:
            return {}
        
        npv_improvements = [r['improvement_pct'] for r in results]
        speedups = [r['speedup'] for r in results]
        
        summary = {
            'avg_npv_improvement': np.mean(npv_improvements),
            'std_npv_improvement': np.std(npv_improvements),
            'avg_speedup': np.mean(speedups),
            'std_speedup': np.std(speedups),
            'max_speedup': np.max(speedups),
            'min_speedup': np.min(speedups),
            'cplex_timeouts': sum(1 for r in results if r['cplex_time'] >= 3590),
            'total_tests': len(results)
        }
        
        return summary