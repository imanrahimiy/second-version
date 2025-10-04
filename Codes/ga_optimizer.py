"""
Genetic Algorithm component for population-based exploration
Integrates with LNS+SA for hybrid optimization as described in manuscript
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm"""
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    tournament_size: int = 3
    diversity_threshold: float = 0.7
    parallel_evaluation: bool = True
    num_threads: int = 8

class MiningChromosome:
    """
    Represents a mining schedule chromosome for GA
    Encodes block extraction sequence and operational modes
    """
    
    def __init__(self, num_blocks: int, num_periods: int = 6):
        self.num_blocks = num_blocks
        self.num_periods = num_periods
        self.genes = {}  # {block_id: (period, mode)}
        self.fitness = None
        self.feasibility_score = 1.0
        
    def random_initialize(self, precedence_graph: Dict):
        """
        Initialize chromosome with random valid schedule
        """
        # Topological sort for valid initialization
        sorted_blocks = self._topological_sort(precedence_graph)
        
        for block in sorted_blocks:
            # Random period assignment respecting precedence
            min_period = self._get_min_feasible_period(block, precedence_graph)
            if min_period < self.num_periods:
                period = random.randint(min_period, self.num_periods - 1)
                mode = random.choice([0, 1])  # Binary mode selection
                self.genes[block] = (period, mode)
            else:
                self.genes[block] = (-1, 0)  # Not scheduled
    
    def _topological_sort(self, precedence: Dict) -> List[int]:
        """
        Topological sort for precedence-feasible ordering
        """
        in_degree = {}
        for block in precedence:
            if block not in in_degree:
                in_degree[block] = 0
            for pred in precedence[block]:
                in_degree[pred] = in_degree.get(pred, 0) + 1
        
        queue = [node for node in in_degree if in_degree[node] == 0]
        sorted_order = []
        
        while queue:
            node = queue.pop(0)
            sorted_order.append(node)
            
            if node in precedence:
                for neighbor in precedence[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return sorted_order
    
    def _get_min_feasible_period(self, block: int, precedence: Dict) -> int:
        """
        Get minimum feasible period for block considering precedence
        """
        if block not in precedence or not precedence[block]:
            return 0
        
        max_pred_period = -1
        for pred in precedence[block]:
            if pred in self.genes:
                pred_period = self.genes[pred][0]
                max_pred_period = max(max_pred_period, pred_period)
        
        return max(0, max_pred_period)
    
    def crossover(self, other: 'MiningChromosome', 
                 crossover_rate: float) -> Tuple['MiningChromosome', 'MiningChromosome']:
        """
        Geology-aware crossover preserving spatial correlations
        """
        if random.random() > crossover_rate:
            return self.copy(), other.copy()
        
        child1 = MiningChromosome(self.num_blocks, self.num_periods)
        child2 = MiningChromosome(self.num_blocks, self.num_periods)
        
        # Two-point crossover with spatial clustering
        blocks = list(self.genes.keys())
        if len(blocks) < 3:
            return self.copy(), other.copy()
        
        # Sort blocks spatially (simplified: by block ID)
        blocks.sort()
        
        # Select crossover points
        point1 = random.randint(1, len(blocks) // 2)
        point2 = random.randint(len(blocks) // 2 + 1, len(blocks) - 1)
        
        # Create offspring
        for i, block in enumerate(blocks):
            if i < point1 or i >= point2:
                if block in self.genes:
                    child1.genes[block] = self.genes[block]
                if block in other.genes:
                    child2.genes[block] = other.genes[block]
            else:
                if block in other.genes:
                    child1.genes[block] = other.genes[block]
                if block in self.genes:
                    child2.genes[block] = self.genes[block]
        
        return child1, child2
    
    def mutate(self, mutation_rate: float, precedence: Dict):
        """
        Adaptive mutation preserving feasibility
        """
        for block in list(self.genes.keys()):
            if random.random() < mutation_rate:
                # Period mutation
                if random.random() < 0.7:
                    min_period = self._get_min_feasible_period(block, precedence)
                    if min_period < self.num_periods:
                        new_period = random.randint(min_period, self.num_periods - 1)
                        old_mode = self.genes[block][1]
                        self.genes[block] = (new_period, old_mode)
                
                # Mode mutation
                if random.random() < 0.3:
                    old_period = self.genes[block][0]
                    new_mode = 1 - self.genes[block][1]  # Flip mode
                    self.genes[block] = (old_period, new_mode)
    
    def copy(self) -> 'MiningChromosome':
        """Create deep copy of chromosome"""
        new_chromosome = MiningChromosome(self.num_blocks, self.num_periods)
        new_chromosome.genes = self.genes.copy()
        new_chromosome.fitness = self.fitness
        new_chromosome.feasibility_score = self.feasibility_score
        return new_chromosome


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for mining optimization
    Implements population-based exploration with geological constraints
    """
    
    def __init__(self, config: GAConfig, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.population = []
        self.best_solution = None
        self.generation = 0
        self.diversity_history = []
        
    def initialize_population(self, blocks: np.ndarray, precedence: Dict):
        """
        Initialize diverse population with different strategies
        """
        self.population = []
        
        # Strategy 1: Random initialization (25%)
        for _ in range(self.config.population_size // 4):
            chromosome = MiningChromosome(len(blocks))
            chromosome.random_initialize(precedence)
            self.population.append(chromosome)
        
        # Strategy 2: High-grade first (25%)
        for _ in range(self.config.population_size // 4):
            chromosome = self._create_highgrade_chromosome(blocks, precedence)
            self.population.append(chromosome)
        
        # Strategy 3: Spatial clustering (25%)
        for _ in range(self.config.population_size // 4):
            chromosome = self._create_spatial_chromosome(blocks, precedence)
            self.population.append(chromosome)
        
        # Strategy 4: Risk-balanced (25%)
        remaining = self.config.population_size - len(self.population)
        for _ in range(remaining):
            chromosome = self._create_balanced_chromosome(blocks, precedence)
            self.population.append(chromosome)
    
    def _create_highgrade_chromosome(self, blocks: np.ndarray, 
                                    precedence: Dict) -> MiningChromosome:
        """Create chromosome prioritizing high-grade blocks"""
        chromosome = MiningChromosome(len(blocks))
        
        # Sort blocks by grade
        grades = blocks[:, 3]
        sorted_indices = np.argsort(grades)[::-1]
        
        period = 0
        period_capacity = 0
        capacity_limit = 6.5e6
        
        for idx in sorted_indices:
            if period >= chromosome.num_periods:
                break
            
            # Check precedence feasibility
            min_period = chromosome._get_min_feasible_period(idx, precedence)
            if min_period > period:
                period = min_period
                period_capacity = 0
            
            if period < chromosome.num_periods:
                # Check capacity
                block_mass = blocks[idx, 4]
                if period_capacity + block_mass > capacity_limit:
                    period += 1
                    period_capacity = 0
                
                if period < chromosome.num_periods:
                    mode = 0 if blocks[idx, 5] < 0.5 else 1  # Rock type based
                    chromosome.genes[idx] = (period, mode)
                    period_capacity += block_mass
        
        return chromosome
    
    def _create_spatial_chromosome(self, blocks: np.ndarray, 
                                  precedence: Dict) -> MiningChromosome:
        """Create chromosome using spatial clustering"""
        from sklearn.cluster import KMeans
        
        chromosome = MiningChromosome(len(blocks))
        
        # Cluster blocks spatially
        coords = blocks[:, :3]
        n_clusters = min(20, len(blocks) // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        # Process clusters sequentially
        for cluster_id in range(n_clusters):
            cluster_blocks = np.where(labels == cluster_id)[0]
            period = cluster_id % chromosome.num_periods
            
            for block_idx in cluster_blocks:
                min_period = chromosome._get_min_feasible_period(block_idx, precedence)
                actual_period = max(period, min_period)
                
                if actual_period < chromosome.num_periods:
                    mode = 0 if blocks[block_idx, 5] < 0.5 else 1
                    chromosome.genes[block_idx] = (actual_period, mode)
        
        return chromosome
    
    def _create_balanced_chromosome(self, blocks: np.ndarray, 
                                   precedence: Dict) -> MiningChromosome:
        """Create risk-balanced chromosome"""
        chromosome = MiningChromosome(len(blocks))
        
        # Calculate risk-adjusted value for each block
        values = blocks[:, 3] * blocks[:, 4]  # grade * mass
        uncertainties = np.random.uniform(0.8, 1.2, len(blocks))
        risk_adjusted = values / uncertainties
        
        sorted_indices = np.argsort(risk_adjusted)[::-1]
        
        # Distribute across periods
        for i, idx in enumerate(sorted_indices):
            period = (i * chromosome.num_periods) // len(sorted_indices)
            min_period = chromosome._get_min_feasible_period(idx, precedence)
            actual_period = max(period, min_period)
            
            if actual_period < chromosome.num_periods:
                mode = 0 if blocks[idx, 5] < 0.5 else 1
                chromosome.genes[idx] = (actual_period, mode)
        
        return chromosome
    
    def evaluate_population(self, blocks: np.ndarray, scenarios: List[Dict]):
        """
        Parallel evaluation of population fitness
        """
        if self.config.parallel_evaluation and len(self.population) > 10:
            self._parallel_evaluation(blocks, scenarios)
        else:
            self._sequential_evaluation(blocks, scenarios)
    
    def _parallel_evaluation(self, blocks: np.ndarray, scenarios: List[Dict]):
        """Parallel fitness evaluation using thread pool"""
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = []
            for chromosome in self.population:
                future = executor.submit(self._evaluate_chromosome, 
                                       chromosome, blocks, scenarios)
                futures.append(future)
            
            for i, future in enumerate(futures):
                fitness, feasibility = future.result()
                self.population[i].fitness = fitness
                self.population[i].feasibility_score = feasibility
    
    def _sequential_evaluation(self, blocks: np.ndarray, scenarios: List[Dict]):
        """Sequential fitness evaluation"""
        for chromosome in self.population:
            fitness, feasibility = self._evaluate_chromosome(chromosome, blocks, scenarios)
            chromosome.fitness = fitness
            chromosome.feasibility_score = feasibility
    
    def _evaluate_chromosome(self, chromosome: MiningChromosome, 
                            blocks: np.ndarray, 
                            scenarios: List[Dict]) -> Tuple[float, float]:
        """
        Evaluate chromosome fitness considering NPV and constraints
        """
        total_npv = 0
        violations = 0
        
        # Calculate NPV across scenarios
        for scenario in scenarios:
            scenario_npv = 0
            for block_idx, (period, mode) in chromosome.genes.items():
                if period >= 0:  # Block is scheduled
                    grade = scenario.get('grades', [blocks[block_idx, 3]])[block_idx] \
                           if block_idx < len(scenario.get('grades', [])) else blocks[block_idx, 3]
                    mass = blocks[block_idx, 4]
                    
                    # Mode-specific recovery
                    recovery = 0.83 if mode == 0 else 0.85
                    revenue = grade * mass * 1190 * recovery
                    
                    # Mode-specific processing cost
                    proc_cost = 21.4 if mode == 0 else 24.9
                    cost = mass * (20.5 + proc_cost)
                    
                    # Discount factor
                    discounted = (revenue - cost) / (1.08 ** period)
                    scenario_npv += discounted
            
            total_npv += scenario_npv / len(scenarios)
        
        # Check capacity constraints
        capacity_per_period = np.zeros(6)
        for block_idx, (period, _) in chromosome.genes.items():
            if 0 <= period < 6:
                capacity_per_period[period] += blocks[block_idx, 4]
        
        for period_capacity in capacity_per_period:
            if period_capacity > 6.5e6:
                violations += (period_capacity - 6.5e6) / 6.5e6
        
        # Feasibility score
        feasibility = 1.0 / (1.0 + violations)
        
        # Combined fitness
        fitness = total_npv * feasibility
        
        return fitness, feasibility
    
    def tournament_selection(self) -> MiningChromosome:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, self.config.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness if x.fitness else -float('inf'))
        return winner.copy()
    
    def calculate_diversity(self) -> float:
        """Calculate population diversity metric"""
        if len(self.population) < 2:
            return 1.0
        
        # Compare gene differences
        total_diff = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, min(i + 10, len(self.population))):
                chr1 = self.population[i]
                chr2 = self.population[j]
                
                common_blocks = set(chr1.genes.keys()) & set(chr2.genes.keys())
                if common_blocks:
                    diff_count = sum(1 for b in common_blocks 
                                   if chr1.genes[b] != chr2.genes[b])
                    total_diff += diff_count / len(common_blocks)
                    comparisons += 1
        
        diversity = total_diff / comparisons if comparisons > 0 else 0
        return diversity
    
    def evolve(self, blocks: np.ndarray, precedence: Dict, 
              scenarios: List[Dict], generations: int = None) -> MiningChromosome:
        """
        Main evolution loop
        """
        if generations is None:
            generations = self.config.generations
        
        # Initialize population if empty
        if not self.population:
            self.initialize_population(blocks, precedence)
        
        # Evaluate initial population
        self.evaluate_population(blocks, scenarios)
        
        for gen in range(generations):
            self.generation = gen
            
            # Track diversity
            diversity = self.calculate_diversity()
            self.diversity_history.append(diversity)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solutions
            sorted_pop = sorted(self.population, 
                              key=lambda x: x.fitness if x.fitness else -float('inf'), 
                              reverse=True)
            for i in range(self.config.elite_size):
                new_population.append(sorted_pop[i].copy())
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Parent selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child1, child2 = parent1.crossover(parent2, self.config.crossover_rate)
                
                # Mutation
                child1.mutate(self.config.mutation_rate, precedence)
                child2.mutate(self.config.mutation_rate, precedence)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            self.population = new_population[:self.config.population_size]
            
            # Evaluate new population
            self.evaluate_population(blocks, scenarios)
            
            # Update best solution
            current_best = max(self.population, 
                             key=lambda x: x.fitness if x.fitness else -float('inf'))
            
            if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
                self.best_solution = current_best.copy()
            
            # Adaptive parameter adjustment based on diversity
            if diversity < self.config.diversity_threshold:
                self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.1)
            else:
                self.config.mutation_rate = max(0.05, self.config.mutation_rate * 0.95)
            
            # Logging
            if gen % 10 == 0:
                avg_fitness = np.mean([c.fitness for c in self.population if c.fitness])
                print(f"Generation {gen}: Best = ${self.best_solution.fitness/1e6:.2f}M, "
                      f"Avg = ${avg_fitness/1e6:.2f}M, Diversity = {diversity:.3f}")
        
        return self.best_solution
    
    def convert_to_schedule(self, chromosome: MiningChromosome, blocks: np.ndarray) -> Dict:
        """Convert chromosome to mining schedule format"""
        schedule = {
            'blocks': [],
            'periods': [],
            'modes': [],
            'npv': chromosome.fitness,
            'feasibility': chromosome.feasibility_score
        }
        
        for block_idx, (period, mode) in chromosome.genes.items():
            if period >= 0:  # Scheduled block
                schedule['blocks'].append(block_idx)
                schedule['periods'].append(period)
                schedule['modes'].append(mode)
        
        return schedule