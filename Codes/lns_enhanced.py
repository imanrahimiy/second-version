"""
Algorithm 2: Hybrid GA+LNS+SA with Epsilon-Constraint Handling
From manuscript Section 4.5 and Algorithm 2
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch

@dataclass
class HybridConfig:
    """Configuration for Algorithm 2"""
    # GA parameters
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    
    # LNS parameters
    destruction_min: float = 0.15
    destruction_max: float = 0.30
    
    # SA parameters
    initial_temperature: float = 300.0
    cooling_rate: float = 0.95
    iterations_per_temp: int = 3
    
    # Epsilon-constraint
    epsilon_max: float = 0.1
    epsilon_threshold: float = 0.01
    
    # Neighborhoods
    n_neighborhoods: int = 5

class HybridGALNSSA:
    """
    Algorithm 2: Hybrid metaheuristic with epsilon-constraint handling
    Integrates GA population diversity, LNS repair, and SA acceptance
    """
    
    def __init__(self, config, gpu_evaluator, vae_model):
        self.config = config
        self.gpu_evaluator = gpu_evaluator
        self.vae_model = vae_model
        self.temperature = config.initial_temperature
        self.current_epsilon = config.epsilon_max
        self.iteration = 0
    
    def create_neighborhoods(self, blocks):
        """
        Partition solution space based on geological similarity
        Uses VAE latent space for intelligent clustering
        """
        with torch.no_grad():
            block_features = torch.tensor(blocks, dtype=torch.float32)
            
            # Encode blocks to latent space
            mu, _ = self.vae_model.encode(block_features)
            latent_repr = mu.numpy()
            
            # K-means clustering in latent space
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.n_neighborhoods, random_state=42)
            labels = kmeans.fit_predict(latent_repr)
        
        neighborhoods = []
        for i in range(self.config.n_neighborhoods):
            neighborhood = np.where(labels == i)[0].tolist()
            neighborhoods.append(neighborhood)
        
        return neighborhoods
    
    def geological_crossover(self, parent1, parent2, neighborhood):
        """
        Geology-aware crossover preserving spatial correlations
        """
        offspring = parent1.copy()
        
        for block_idx in neighborhood:
            if random.random() < self.config.crossover_rate:
                # Check precedence feasibility
                if self._check_precedence_feasible(parent2, block_idx):
                    offspring['schedule'][block_idx] = parent2['schedule'][block_idx]
                    offspring['mode'][block_idx] = parent2['mode'][block_idx]
        
        return offspring
    
    def spatial_mutation(self, solution, neighborhood):
        """
        Mutation preserving geological continuity
        """
        mutated = solution.copy()
        
        for block_idx in neighborhood:
            if random.random() < self.config.mutation_rate:
                feasible_periods = self._get_feasible_periods(mutated, block_idx)
                if feasible_periods:
                    mutated['schedule'][block_idx] = random.choice(feasible_periods)
                
                # Mode mutation
                if random.random() < 0.5:
                    mutated['mode'][block_idx] = 1 - mutated['mode'][block_idx]
        
        return mutated
    
    def evaluate_fitness_epsilon(self, solution, epsilon):
        """
        Fitness evaluation with epsilon-relaxed constraints
        """
        npv = self.gpu_evaluator.evaluate_npv(solution)
        violations = self._calculate_violations(solution)
        
        # Apply epsilon relaxation
        relaxed_violation = max(0, violations - epsilon)
        penalty = 1e6 * relaxed_violation
        
        return npv - penalty
    
    def tournament_selection(self, population, fitness_scores):
        """Tournament selection for GA"""
        tournament_indices = random.sample(
            range(len(population)), 
            min(self.config.tournament_size, len(population))
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def lns_destroy(self, solution, neighborhood):
        """
        Destroy phase: remove blocks from neighborhood
        """
        destruction_size = random.uniform(
            self.config.destruction_min,
            self.config.destruction_max
        )
        num_to_destroy = int(len(neighborhood) * destruction_size)
        destroyed_blocks = random.sample(neighborhood, min(num_to_destroy, len(neighborhood)))
        
        for block_idx in destroyed_blocks:
            solution['schedule'][block_idx] = -1  # Mark as unassigned
        
        return solution, destroyed_blocks
    
    def lns_repair(self, solution, destroyed_blocks):
        """
        GPU-accelerated repair with VAE-enhanced selection
        """
        # Use GPU evaluator for parallel repair
        repaired = self.gpu_evaluator.gpu_accelerated_repair(
            solution, destroyed_blocks, []  # Empty scenarios for now
        )
        return repaired
    
    def sa_accept(self, current_fitness, new_fitness):
        """Simulated annealing acceptance criterion"""
        if new_fitness > current_fitness:
            return True
        
        delta = new_fitness - current_fitness
        probability = np.exp(delta / self.temperature)
        return random.random() < probability
    
    def optimize(self, initial_solution, max_iterations):
        """
        Main optimization loop (Algorithm 2)
        """
        # Step 1: Initialize population
        population = [initial_solution]
        for _ in range(self.config.population_size - 1):
            perturbed = self._perturb_solution(initial_solution)
            population.append(perturbed)
        
        # Step 2: Create neighborhoods
        neighborhoods = self.create_neighborhoods(initial_solution['blocks'])
        
        best_solution = initial_solution
        best_fitness = self.evaluate_fitness_epsilon(best_solution, 0)
        
        # Main loop
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # Step 3: Update epsilon
            self.current_epsilon = self.config.epsilon_max * (1 - iteration/max_iterations)
            
            # Step 4: Process each neighborhood
            for neighborhood in neighborhoods:
                new_population = []
                
                # Step 5-10: GA exploration
                for _ in range(self.config.population_size):
                    # Calculate fitness
                    fitness_scores = [
                        self.evaluate_fitness_epsilon(sol, self.current_epsilon)
                        for sol in population
                    ]
                    
                    # Select parents
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    offspring = self.geological_crossover(parent1, parent2, neighborhood)
                    
                    # Mutation
                    offspring = self.spatial_mutation(offspring, neighborhood)
                    
                    new_population.append(offspring)
                
                # Step 11-15: LNS repair for infeasible solutions
                for i, solution in enumerate(new_population):
                    violations = self._calculate_violations(solution)
                    if violations > self.current_epsilon:
                        # Destroy
                        destroyed_solution, destroyed_blocks = self.lns_destroy(
                            solution, neighborhood
                        )
                        # Repair
                        new_population[i] = self.lns_repair(
                            destroyed_solution, destroyed_blocks
                        )
                
                # Step 16-18: SA-based neighborhood transition
                best_in_neighborhood = max(
                    new_population,
                    key=lambda x: self.evaluate_fitness_epsilon(x, self.current_epsilon)
                )
                
                fitness = self.evaluate_fitness_epsilon(
                    best_in_neighborhood, self.current_epsilon
                )
                
                if self.sa_accept(best_fitness, fitness):
                    best_solution = best_in_neighborhood
                    best_fitness = fitness
                
                # Update population
                population = self._elitist_selection(population + new_population)
            
            # Step 19: Remove infeasible if epsilon is small
            if self.current_epsilon < self.config.epsilon_threshold:
                population = [
                    sol for sol in population
                    if self._calculate_violations(sol) <= self.current_epsilon
                ]
                if not population:
                    population = [best_solution]
            
            # Step 20: Cool temperature
            self.temperature *= self.config.cooling_rate
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best NPV = ${best_fitness/1e6:.2f}M, "
                      f"ε = {self.current_epsilon:.4f}, T = {self.temperature:.2f}")
        
        # Step 21-22: Final repair with epsilon = 0
        if self._calculate_violations(best_solution) > 0:
            print("Applying final repair...")
            all_blocks = list(range(len(best_solution['blocks'])))
            destroyed_solution, destroyed_blocks = self.lns_destroy(
                best_solution, all_blocks
            )
            best_solution = self.lns_repair(destroyed_solution, destroyed_blocks)
        
        return best_solution
    
    def _check_precedence_feasible(self, solution, block_idx):
        """Check precedence constraints"""
        precedence = solution.get('precedence', {})
        if block_idx in precedence:
            for pred in precedence[block_idx]:
                if solution['schedule'][pred] > solution['schedule'][block_idx]:
                    return False
        return True
    
    def _get_feasible_periods(self, solution, block_idx):
        """Get feasible periods considering precedence"""
        precedence = solution.get('precedence', {})
        min_period = 0
        
        if block_idx in precedence:
            pred_periods = [
                solution['schedule'][pred] 
                for pred in precedence[block_idx]
                if pred < len(solution['schedule'])
            ]
            if pred_periods:
                min_period = max(pred_periods)
        
        return list(range(min_period, 6))
    
    def _calculate_violations(self, solution):
        """Calculate constraint violations"""
        violations = 0.0
        
        # Precedence violations
        for block_idx in range(len(solution['schedule'])):
            if not self._check_precedence_feasible(solution, block_idx):
                violations += 1.0
        
        # Capacity violations
        capacity_per_period = np.zeros(6)
        for block_idx, period in enumerate(solution['schedule']):
            if 0 <= period < 6:
                capacity_per_period[period] += 15375  # Block mass
        
        for period_mass in capacity_per_period:
            if period_mass > 6.5e6:
                violations += (period_mass - 6.5e6) / 6.5e6
        
        return violations
    
    def _perturb_solution(self, solution):
        """Create perturbed solution for population"""
        perturbed = solution.copy()
        num_blocks = len(solution['blocks'])
        
        num_changes = random.randint(
            int(0.1 * num_blocks), 
            int(0.2 * num_blocks)
        )
        blocks_to_change = random.sample(range(num_blocks), min(num_changes, num_blocks))
        
        for block_idx in blocks_to_change:
            feasible_periods = self._get_feasible_periods(perturbed, block_idx)
            if feasible_periods:
                perturbed['schedule'][block_idx] = random.choice(feasible_periods)
        
        return perturbed
    
    def _elitist_selection(self, population):
        """Select best solutions for next generation"""
        fitness_scores = [
            self.evaluate_fitness_epsilon(sol, self.current_epsilon)
            for sol in population
        ]
        
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[i] for i in sorted_indices[:self.config.population_size]]