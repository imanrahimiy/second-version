# data_generator.py
"""
Synthetic mining data generation with geological scenarios
Based on Quelopana & Navarra (2024) methodology
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
import pickle
import os
from config import MiningParameters, DATA_DIR

class MiningDataGenerator:
    """
    Generate synthetic mining dataset with:
    - 3D block model
    - Spatially correlated gold grades
    - Rock types (Diorite Porphyry, Silicified Intrusive Breccia)
    - Geological uncertainty scenarios
    - Precedence relationships
    - Economic block values
    """
    
    def __init__(self, params: MiningParameters, n_blocks: int = None):
        self.params = params
        self.n_blocks = n_blocks or params.n_blocks
        self.grid_size = int(np.ceil(np.cbrt(self.n_blocks)))
        
    def generate_complete_dataset(self) -> Dict:
        """Generate complete synthetic mining dataset"""
        print("=" * 80)
        print("GENERATING SYNTHETIC MINING DATASET")
        print("=" * 80)
        print(f"Blocks: {self.n_blocks}")
        print(f"Periods: {self.params.n_periods}")
        print(f"Scenarios: {self.params.n_scenarios}")
        print("-" * 80)
        
        # 1. Generate 3D spatial coordinates
        print("Generating 3D block coordinates...")
        coordinates = self._generate_3d_coordinates()
        
        # 2. Generate geological features
        print("Generating geological features...")
        geological_features = self._generate_geological_features(coordinates)
        
        # 3. Generate grades and rock types with spatial correlation
        print("Generating grades and rock types...")
        grades, rock_types = self._generate_spatially_correlated_grades(
            coordinates, geological_features
        )
        
        # 4. Generate precedence relationships
        print("Generating precedence constraints...")
        precedence = self._generate_precedence_constraints(coordinates)
        
        # 5. Generate geological uncertainty scenarios
        print("Generating stochastic scenarios...")
        scenarios = self._generate_stochastic_scenarios(grades, rock_types)
        
        # 6. Calculate economic block values
        print("Calculating block values...")
        block_values = self._calculate_block_values(scenarios)
        
        # 7. Generate discounted mining costs
        print("Generating mining costs...")
        mining_costs = self._generate_discounted_costs()
        
        dataset = {
            'n_blocks': self.n_blocks,
            'coordinates': coordinates,
            'geological_features': geological_features,
            'base_grades': grades,
            'base_rock_types': rock_types,
            'precedence': precedence,
            'scenarios': scenarios,
            'block_values': block_values,
            'mining_costs': mining_costs,
            'params': self.params
        }
        
        self._print_statistics(dataset)
        
        print("=" * 80)
        print(f"Dataset generation complete: {self.n_blocks} blocks")
        print("=" * 80)
        
        return dataset
    
    def _generate_3d_coordinates(self) -> np.ndarray:
        """Generate 3D block coordinates in mine space"""
        x = np.arange(self.grid_size) * self.params.block_dimensions[0]
        y = np.arange(self.grid_size) * self.params.block_dimensions[1]
        z = np.arange(self.grid_size) * self.params.block_dimensions[2]
        
        xx, yy, zz = np.meshgrid(x, y, z)
        coordinates = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Ensure exact block count
        coordinates = coordinates[:self.n_blocks]
        self.n_blocks = len(coordinates)
        
        return coordinates
    
    def _generate_geological_features(self, coordinates: np.ndarray) -> Dict:
        """
        Generate geological features:
        - Alteration intensity (proximity to intrusion)
        - Structural density (fault zones)
        - Distance to intrusion center
        """
        n = len(coordinates)
        
        # Simulate porphyry intrusion center
        intrusion_center = np.array([
            coordinates[:, 0].mean(),
            coordinates[:, 1].mean(),
            coordinates[:, 2].min() + 100  # 100m from surface
        ])
        
        # Calculate distance to intrusion (normalized)
        distances = np.linalg.norm(coordinates - intrusion_center, axis=1)
        distance_to_intrusion = distances / distances.max()
        
        # Alteration intensity (higher near intrusion, exponential decay)
        alteration_intensity = np.exp(-distance_to_intrusion * 2)
        alteration_intensity += np.random.normal(0, 0.1, n)
        alteration_intensity = np.clip(alteration_intensity, 0, 1)
        
        # Structural density (random with beta distribution)
        structural_density = np.random.beta(2, 5, n)
        
        # Apply spatial correlation to structural density
        structural_density = self._apply_spatial_smoothing(
            coordinates, structural_density, radius=50
        )
        
        return {
            'alteration_intensity': alteration_intensity,
            'structural_density': structural_density,
            'distance_to_intrusion': distance_to_intrusion
        }
    
    def _generate_spatially_correlated_grades(
        self, 
        coordinates: np.ndarray,
        features: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate gold grades with spatial correlation
        Rock types: 0=Diorite Porphyry, 1=Silicified Intrusive Breccia
        """
        n = len(coordinates)
        
        # Rock type probability based on alteration intensity
        # Higher alteration -> more likely to be silicified
        rock_type_prob = 0.3 + 0.4 * features['alteration_intensity']
        rock_types = (np.random.random(n) < rock_type_prob).astype(int)
        
        # Base grade correlated with alteration
        base_grade = 0.02 + 0.08 * features['alteration_intensity']
        
        # Add lognormal variability (gold grades are lognormally distributed)
        grade_noise = np.random.lognormal(0, 0.5, n)
        grades = base_grade * grade_noise
        
        # Silicified rocks tend to have higher grades
        grades[rock_types == 1] *= 1.3
        
        # Apply spatial smoothing to create realistic spatial correlation
        grades = self._apply_spatial_smoothing(coordinates, grades, radius=40)
        
        # Clip to reasonable range
        grades = np.clip(grades, 0.001, 1.0)  # oz/ton
        
        return grades, rock_types
    
    def _apply_spatial_smoothing(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        radius: float = 40
    ) -> np.ndarray:
        """Apply spatial smoothing filter using Gaussian weights"""
        smoothed = values.copy()
        
        for i in range(len(coordinates)):
            # Find neighbors within radius
            distances = np.linalg.norm(
                coordinates - coordinates[i], axis=1
            )
            neighbors = np.where(distances < radius)[0]
            
            if len(neighbors) > 1:
                # Gaussian weights
                weights = np.exp(-distances[neighbors] / (radius/3))
                weights /= weights.sum()
                smoothed[i] = np.sum(values[neighbors] * weights)
        
        return smoothed
    
    def _generate_precedence_constraints(
        self,
        coordinates: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Generate precedence relationships:
        Blocks directly above must be mined before blocks below (mining sequence)
        """
        precedence = {}
        
        for i in range(len(coordinates)):
            predecessors = []
            
            # Find all blocks directly above (higher z, same x,y)
            for j in range(len(coordinates)):
                if (coordinates[j, 0] == coordinates[i, 0] and 
                    coordinates[j, 1] == coordinates[i, 1] and
                    coordinates[j, 2] > coordinates[i, 2]):
                    predecessors.append(j)
            
            precedence[i] = predecessors
        
        return precedence
    
    def _generate_stochastic_scenarios(
        self,
        base_grades: np.ndarray,
        base_rock_types: np.ndarray
    ) -> List[Dict]:
        """
        Generate geological uncertainty scenarios
        Represents uncertainty in grade estimation and rock type classification
        """
        scenarios = []
        
        for s in range(self.params.n_scenarios):
            # Add uncertainty to grades (±15% coefficient of variation)
            uncertainty_factor = np.random.normal(1.0, 0.15, len(base_grades))
            scenario_grades = base_grades * uncertainty_factor
            scenario_grades = np.clip(scenario_grades, 0.001, None)
            
            # Small probability of rock type misclassification (5%)
            scenario_rocks = base_rock_types.copy()
            flip_mask = np.random.random(len(base_rock_types)) < 0.05
            scenario_rocks[flip_mask] = 1 - scenario_rocks[flip_mask]
            
            scenarios.append({
                'scenario_id': s,
                'grades': scenario_grades,
                'rock_types': scenario_rocks,
                'probability': 1.0 / self.params.n_scenarios  # Equal probability
            })
        
        return scenarios
    
    def _calculate_block_values(self, scenarios: List[Dict]) -> np.ndarray:
        """
        Calculate economic block values for each scenario and operational mode
        Shape: (n_blocks, n_scenarios, 2_modes)
        Mode 0: Mode A (Diorite-focused)
        Mode 1: Mode B (Silicified-focused)
        """
        n_blocks = self.n_blocks
        n_scenarios = len(scenarios)
        values = np.zeros((n_blocks, n_scenarios, 2))
        
        for s, scenario in enumerate(scenarios):
            grades = scenario['grades']
            
            # Mode A value (Diorite Porphyry focused)
            revenue_a = (
                grades * 
                self.params.gold_price * 
                self.params.mode_a_recovery
            )
            cost_a = self.params.mode_a_cost
            values[:, s, 0] = (revenue_a - cost_a) * self.params.block_weight
            
            # Mode B value (Silicified Intrusive Breccia focused)
            revenue_b = (
                grades * 
                self.params.gold_price * 
                self.params.mode_b_recovery
            )
            cost_b = self.params.mode_b_cost
            values[:, s, 1] = (revenue_b - cost_b) * self.params.block_weight
        
        return values
    
    def _generate_discounted_costs(self) -> np.ndarray:
        """
        Generate discounted mining costs for each period
        Shape: (n_blocks, n_periods)
        """
        costs = np.zeros((self.n_blocks, self.params.n_periods))
        
        for t in range(self.params.n_periods):
            discount_factor = 1 / ((1 + self.params.discount_rate) ** t)
            costs[:, t] = (
                self.params.mining_cost * 
                self.params.block_weight * 
                discount_factor
            )
        
        return costs
    
    def _print_statistics(self, dataset: Dict):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print("-" * 80)
        
        grades = dataset['base_grades']
        rocks = dataset['base_rock_types']
        
        print(f"Grade Statistics:")
        print(f"  Mean: {grades.mean():.4f} oz/ton")
        print(f"  Std:  {grades.std():.4f} oz/ton")
        print(f"  Min:  {grades.min():.4f} oz/ton")
        print(f"  Max:  {grades.max():.4f} oz/ton")
        
        print(f"\nRock Type Distribution:")
        print(f"  Diorite Porphyry: {(rocks==0).sum()} blocks ({(rocks==0).mean()*100:.1f}%)")
        print(f"  Silicified Breccia: {(rocks==1).sum()} blocks ({(rocks==1).mean()*100:.1f}%)")
        
        print(f"\nPrecedence Constraints:")
        total_precedences = sum(len(preds) for preds in dataset['precedence'].values())
        avg_precedences = total_precedences / self.n_blocks
        print(f"  Total: {total_precedences}")
        print(f"  Average per block: {avg_precedences:.2f}")
        
        values = dataset['block_values']
        print(f"\nBlock Values:")
        print(f"  Mode A mean: ${values[:,:,0].mean()/1e3:.2f}K")
        print(f"  Mode B mean: ${values[:,:,1].mean()/1e3:.2f}K")
    
    def save_dataset(self, dataset: Dict, filename: str = 'mining_data.pkl'):
        """Save dataset to file"""
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nDataset saved to {filepath}")
    
    @staticmethod
    def load_dataset(filename: str = 'mining_data.pkl') -> Dict:
        """Load dataset from file"""
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Dataset loaded from {filepath}")
        return dataset

# ============================================================================
# MAIN: Generate and save dataset
# ============================================================================

if __name__ == '__main__':
    from config import DEFAULT_MINING_PARAMS
    
    # Generate dataset
    generator = MiningDataGenerator(DEFAULT_MINING_PARAMS)
    dataset = generator.generate_complete_dataset()
    
    # Save to file
    generator.save_dataset(dataset)
    
    print("\nData generation complete!")
    print("Use MiningDataGenerator.load_dataset() to load in other modules")