# config.py
"""
Configuration and parameters for Mining Optimization System
Based on Quelopana & Navarra (2024)
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class MiningParameters:
    """Mining parameters from Table 1 of Quelopana & Navarra (2024)"""
    
    # Problem dimensions
    n_blocks: int = 13392  # Reduced for computational feasibility
    n_periods: int = 6
    n_scenarios: int = 10
    
    # Block properties
    block_weight: float = 15375  # tons
    block_dimensions: Tuple[int, int, int] = (20, 20, 15)  # meters (x, y, z)
    block_density: float = 2.56  # ton/m³
    
    # Economic parameters
    discount_rate: float = 0.08  # 8% annual discount rate
    gold_price: float = 1190  # $/oz
    mining_cost: float = 20.5  # $/ton
    
    # Capacity constraints
    mining_capacity: float = 6.5e6  # ton/period
    processing_availability: float = 8075  # hours/period
    
    # Operational Mode A (Diorite Porphyry focused)
    mode_a_rate: float = 250  # ton/hr processing rate
    mode_a_cost: float = 21.4  # $/ton processing cost
    mode_a_recovery: float = 0.83  # metal recovery rate
    mode_a_diorite_pct: float = 0.65  # preferred diorite percentage
    mode_a_silicified_pct: float = 0.35  # preferred silicified percentage
    
    # Operational Mode B (Silicified Intrusive Breccia focused)
    mode_b_rate: float = 200  # ton/hr processing rate
    mode_b_cost: float = 24.9  # $/ton processing cost
    mode_b_recovery: float = 0.83  # metal recovery rate
    mode_b_diorite_pct: float = 0.45  # preferred diorite percentage
    mode_b_silicified_pct: float = 0.55  # preferred silicified percentage

@dataclass
class VNDParameters:
    """Variable Neighborhood Descent parameters from paper Section 3"""
    
    max_iterations: int = 100
    k_max: int = 4  # Maximum neighborhood size
    shake_intensity: float = 0.15  # Percentage of blocks to shake
    local_search_iterations: int = 20
    improvement_threshold: float = 0.001  # 0.1% improvement threshold
    time_limit: int = 3600  # seconds
    
    # Neighborhood structures
    neighborhood_types: List[str] = field(default_factory=lambda: [
        'swap',      # Swap blocks between periods
        'shift',     # Shift block to different period
        'exchange',  # Exchange blocks between periods
        'insert'     # Insert block into new period
    ])

@dataclass
class GAParameters:
    """Genetic Algorithm baseline parameters"""
    
    population_size: int = 50
    max_iterations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism_count: int = 5
    time_limit: int = 3600

@dataclass
class VAEParameters:
    """VAE parameters for geological uncertainty modeling"""
    
    input_dim: int = 5  # [grade, rock_type, alteration, structural, distance]
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    beta: float = 0.5  # KL divergence weight

@dataclass
class DWParameters:
    """Dantzig-Wolfe decomposition parameters"""
    
    max_iterations: int = 100
    convergence_tolerance: float = 1e-4
    column_generation_limit: int = 50

@dataclass
class DOEParameters:
    """Design of Experiments parameters for parameter tuning"""
    
    n_replications: int = 5
    significance_level: float = 0.05
    
    # Parameter ranges for Taguchi L16 design
    parameter_ranges: Dict = field(default_factory=lambda: {
        'vnd_k_max': [2, 3, 4, 5],
        'vnd_shake': [0.10, 0.15, 0.20, 0.25],
        'vnd_local_iter': [10, 20, 30, 40],
        'ga_pop_size': [30, 50, 70, 100],
        'ga_mutation': [0.05, 0.10, 0.15, 0.20],
        'vae_latent': [16, 32, 64, 128]
    })

# ============================================================================
# Global Configuration
# ============================================================================

# Computation settings
USE_GPU = True
RANDOM_SEED = 42
NUM_THREADS = -1  # -1 means use all available cores

# Directory structure
OUTPUT_DIR = 'results'
DATA_DIR = 'data'
MODEL_DIR = 'models'
FIGURE_DIR = 'figures'

# Logging
VERBOSE = True
LOG_LEVEL = 'INFO'

# ============================================================================
# Setup Functions
# ============================================================================

def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get computing device (GPU/CPU)"""
    if torch.cuda.is_available() and USE_GPU:
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
        if USE_GPU:
            print("Warning: GPU requested but not available")
    
    return device

def setup_directories():
    """Create necessary directories"""
    import os
    dirs = [OUTPUT_DIR, DATA_DIR, MODEL_DIR, FIGURE_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

# ============================================================================
# Initialize on import
# ============================================================================

set_random_seeds(RANDOM_SEED)
setup_directories()

# Export main configuration objects
DEVICE = get_device()

# Default parameter instances
DEFAULT_MINING_PARAMS = MiningParameters()
DEFAULT_VND_PARAMS = VNDParameters()
DEFAULT_GA_PARAMS = GAParameters()
DEFAULT_VAE_PARAMS = VAEParameters()
DEFAULT_DW_PARAMS = DWParameters()
DEFAULT_DOE_PARAMS = DOEParameters()

if __name__ == '__main__':
    print("=" * 80)
    print("CONFIGURATION MODULE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Directories: {[OUTPUT_DIR, DATA_DIR, MODEL_DIR, FIGURE_DIR]}")
    print("=" * 80)