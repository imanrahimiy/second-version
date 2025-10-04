"""
Enhanced Large Neighborhood Search with GA+SA hybrid and epsilon-constraint handling
Based on Algorithm 2 from the revised manuscript
Implements the hybrid GA+LNS+SA metaheuristic with adaptive constraint relaxation
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class LNSConfig:
    """Configuration for enhanced LNS with GA integration"""
    destruction_min: float = 0.15
    destruction_max: float = 0.30
    population_size: int = 50
    max_generations: int = 10
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    epsilon_max: float = 0.1
    epsilon_threshold: float = 0.01
    neighborhoods: int = 5

class HybridGALNSSA:
    """
    Hybrid Genetic Algorithm + Large Neighborhood Search + Simulated Annealing
    with epsilon-constraint handling for mining optimization
    """
    
    def __init__(self, config: LNSConfig, gpu_evaluator, vae_model):
        self.config = config
        self.gpu_evaluator = gpu_evaluator
        self.vae_model = vae_model
        self.temperature = 300.0
        self.cooling_rate = 0.95
        self.current_epsilon = config.epsilon_max
        
    def create_neighborhoods(self, blocks: np.ndarray) -> List[List[int]]:
        """
        Partition solution space into neighborhoods based on geological similarity
        Uses VAE latent space for intelligent clustering
        """
        with torch.no_grad():
            # Encode blocks using VAE to get latent representations
            block_features = torch.tensor(blocks, dtype=torch.float32)
            latent_repr = self.vae_model.encode(block_features)[0]
            
            # Cluster using K-means in latent space
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.neighborhoods, random_state=42)
            neighborhood_labels = kmeans.fit_predict(latent_repr.numpy())
            
        neighborhoods = []
        for i in range(self.config.neighborhoods):
            neighborhood = np.where(neighborhood_labels == i)[0].tolist()
            neighborhoods.append(neighborhood)
            
        return neighborhoods
    
    def geological_crossover(self, parent1: Dict, parent2: Dict, 
                            neighborhood: List[int]) -> Dict:
        """
        Geology-aware crossover that preserves spatial correlations
        """
        offspring = parent1.copy()
        
        # Only crossover within the current neighborhood
        for block_idx in neighborhood:
            if random.random() < self.config.crossover_rate:
                # Inherit from parent2 while maintaining precedence
                if self._check_precedence_feasible(parent2, block_idx):
                    offspring['schedule'][block_idx] = parent2['schedule'][block_idx]
                    offspring['mode'][block_idx] = parent2['mode'][block_idx]
        
        return offspring
    
    def spatial_mutation(self, solution: Dict, neighborhood: List[int]) -> Dict:
        """
        Mutation that preserves geological continuity and precedence constraints
        """
        mutated = solution.copy()
        
        for block_idx in neighborhood:
            if random.random() < self.config.mutation_rate:
                # Find feasible periods considering precedence
                feasible_periods = self._get_feasible_periods(mutated, block_idx)
                if feasible_periods:
                    mutated['schedule'][block_idx] = random.choice(feasible_periods)
                    
                # Randomly change operational mode
                if random.random() < 0.5:
                    mutated['mode'][block_idx] = 1 - mutated['mode'][block_idx]
        
        return mutated
    
    def evaluate_fitness_epsilon(self, solution: Dict, epsilon: float) -> float:
        """
        Fitness evaluation with epsilon-relaxed constraints
        """
        npv = self.gpu_evaluator.evaluate_npv(solution)
        violations = self._calculate_violations(solution)
        
        # Apply epsilon relaxation
        relaxed_violation = max(0, violations - epsilon)
        penalty = 1e6 * relaxed_violation
        
        return npv - penalty
    
    def tournament_selection(self, population: List[Dict], 
                            fitness_scores: List[float]) -> Dict:
        """
        Tournament selection for parent selection
        """
        tournament_indices = random.sample(range(len(population)), 
                                         self.config.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx]
    
    def lns_repair(self, solution: Dict, destroyed_blocks: List[int]) -> Dict:
        """
        GPU-accelerated repair mechanism for infeasible solutions
        """
        repaired = solution.copy()
        
        # Use GPU to evaluate all possible repairs in parallel
        repair_candidates = self.gpu_evaluator.generate_repair_candidates(
            repaired, destroyed_blocks
        )
        
        # Select best repair based on NPV and feasibility
        best_repair = self.gpu_evaluator.evaluate_repairs_parallel(
            repair_candidates, self.current_epsilon
        )
        
        return best_repair
    
    def sa_accept(self, current_fitness: float, new_fitness: float) -> bool:
        """
        Simulated annealing acceptance criterion
        """
        if new_fitness > current_fitness:
            return True
        
        delta = new_fitness - current_fitness
        probability = np.exp(delta / self.temperature)
        return random.random() < probability
    
    def optimize(self, initial_solution: Dict, max_iterations: int) -> Dict:
        """
        Main optimization loop implementing Algorithm 2 from manuscript
        """
        # Initialize population
        population = [initial_solution]
        for _ in range(self.config.population_size - 1):
            perturbed = self._perturb_solution(initial_solution)
            population.append(perturbed)
        
        # Create neighborhoods based on geological similarity
        neighborhoods = self.create_neighborhoods(initial_solution['blocks'])
        
        best_solution = initial_solution
        best_fitness = self.evaluate_fitness_epsilon(best_solution, 0)
        
        for iteration in range(max_iterations):
            # Update epsilon constraint (linearly decrease)
            self.current_epsilon = self.config.epsilon_max * (1 - iteration/max_iterations)
            
            for neighborhood in neighborhoods:
                # Population-based exploration in neighborhood
                new_population = []
                
                for _ in range(self.config.max_generations):
                    # Select parents using tournament selection
                    fitness_scores = [
                        self.evaluate_fitness_epsilon(sol, self.current_epsilon)
                        for sol in population
                    ]
                    
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                    
                    # Generate offspring with geology-aware crossover
                    offspring = self.geological_crossover(parent1, parent2, neighborhood)
                    
                    # Apply spatial mutation
                    offspring = self.spatial_mutation(offspring, neighborhood)
                    
                    new_population.append(offspring)
                
                # LNS repair for infeasible solutions
                for i, solution in enumerate(new_population):
                    violations = self._calculate_violations(solution)
                    if violations > self.current_epsilon:
                        # Apply destruction phase
                        destroyed = self._destroy_solution(solution, neighborhood)
                        # GPU-accelerated repair
                        new_population[i] = self.lns_repair(solution, destroyed)
                
                # SA-based neighborhood transition
                best_in_neighborhood = max(new_population, 
                    key=lambda x: self.evaluate_fitness_epsilon(x, self.current_epsilon))
                
                fitness = self.evaluate_fitness_epsilon(best_in_neighborhood, 
                                                       self.current_epsilon)
                
                if self.sa_accept(best_fitness, fitness):
                    best_solution = best_in_neighborhood
                    best_fitness = fitness
                    
                # Update population (elitist selection)
                population = self._elitist_selection(population + new_population)
            
            # Remove highly infeasible solutions if epsilon is small
            if self.current_epsilon < self.config.epsilon_threshold:
                population = [sol for sol in population 
                            if self._calculate_violations(sol) <= self.current_epsilon]
            
            # Cool down temperature
            self.temperature *= self.config.cooling_rate
            
            print(f"Iteration {iteration}: Best NPV = ${best_fitness/1e6:.2f}M, "
                  f"Epsilon = {self.current_epsilon:.4f}, Temp = {self.temperature:.2f}")
        
        # Final feasibility check with epsilon = 0
        if self._calculate_violations(best_solution) > 0:
            print("Applying final repair with strict constraints...")
            destroyed = self._destroy_solution(best_solution, range(len(best_solution['blocks'])))
            best_solution = self.lns_repair(best_solution, destroyed)
        
        return best_solution
    
    def _check_precedence_feasible(self, solution: Dict, block_idx: int) -> bool:
        """Check if block assignment respects precedence constraints"""
        precedence = solution.get('precedence', {})
        if block_idx in precedence:
            for pred in precedence[block_idx]:
                if solution['schedule'][pred] > solution['schedule'][block_idx]:
                    return False
        return True
    
    def _get_feasible_periods(self, solution: Dict, block_idx: int) -> List[int]:
        """Get list of feasible periods for a block considering precedence"""
        precedence = solution.get('precedence', {})
        min_period = 0
        
        if block_idx in precedence:
            predecessor_periods = [solution['schedule'][pred] 
                                 for pred in precedence[block_idx]]
            if predecessor_periods:
                min_period = max(predecessor_periods)
        
        max_period = solution.get('num_periods', 6)
        return list(range(min_period, max_period))
    
    def _calculate_violations(self, solution: Dict) -> float:
        """Calculate total constraint violations"""
        violations = 0.0
        
        # Check precedence violations
        for block_idx, period in enumerate(solution['schedule']):
            if not self._check_precedence_feasible(solution, block_idx):
                violations += 1.0
        
        # Check capacity violations
        capacity_per_period = solution.get('capacity', [6.5e6] * 6)
        mass_per_period = np.zeros(len(capacity_per_period))
        
        for block_idx, period in enumerate(solution['schedule']):
            if period >= 0:  # -1 means not scheduled
                mass_per_period[period] += solution['blocks'][block_idx]['mass']
        
        for period, (mass, capacity) in enumerate(zip(mass_per_period, capacity_per_period)):
            if mass > capacity:
                violations += (mass - capacity) / capacity
        
        return violations
    
    def _perturb_solution(self, solution: Dict) -> Dict:
        """Create perturbed version of solution for population initialization"""
        perturbed = solution.copy()
        num_blocks = len(solution['blocks'])
        
        # Randomly change 10-20% of block assignments
        num_changes = random.randint(int(0.1 * num_blocks), int(0.2 * num_blocks))
        blocks_to_change = random.sample(range(num_blocks), num_changes)
        
        for block_idx in blocks_to_change:
            feasible_periods = self._get_feasible_periods(perturbed, block_idx)
            if feasible_periods:
                perturbed['schedule'][block_idx] = random.choice(feasible_periods)
        
        return perturbed
    
    def _destroy_solution(self, solution: Dict, neighborhood: List[int]) -> List[int]:
        """Destroy part of solution in neighborhood"""
        destruction_size = random.uniform(self.config.destruction_min, 
                                        self.config.destruction_max)
        num_to_destroy = int(len(neighborhood) * destruction_size)
        return random.sample(neighborhood, num_to_destroy)
    
    def _elitist_selection(self, population: List[Dict]) -> List[Dict]:
        """Select best solutions to maintain population size"""
        fitness_scores = [
            self.evaluate_fitness_epsilon(sol, self.current_epsilon)
            for sol in population
        ]
        
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[i] for i in sorted_indices[:self.config.population_size]]