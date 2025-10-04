"""
Enhanced Configuration for AI-Enhanced Mining Optimization System
Based on Revised Manuscript (2025) - Implements all algorithm parameters
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import os

@dataclass
class MiningParameters:
    """Mining parameters from Table 1 of the revised manuscript"""
    
    # Problem dimensions - Manuscript specifies 50,000 blocks
    n_blocks: int = 50000  # As per manuscript
    n_periods: int = 6     # 6 annual periods
    n_scenarios: int = 10  # Initial static scenarios (will be expanded by VAE)
    
    # Block properties (Table 1)
    block_weight: float = 15375  # tons
    block_dimensions: Tuple[int, int, int] = (20, 20, 15)  # meters (x, y, z)
    block_density: float = 2.56  # ton/m³
    
    # Economic parameters (Table 1)
    discount_rate: float = 0.08  # 8% annual discount rate
    gold_price: float = 1190  # $/oz
    mining_cost: float = 20.5  # $/ton
    
    # Capacity constraints (Table 1)
    mining_capacity: float = 6.5e6  # ton/period
    processing_availability: float = 8075  # hours/period
    
    # Operational Mode A (Table 1)
    mode_a_rate: float = 250  # ton/hr processing rate
    mode_a_cost: float = 21.4  # $/ton processing cost
    mode_a_recovery: float = 0.83  # 83% metal recovery
    mode_a_diorite_pct: float = 0.65  # 65% Diorite
    mode_a_silicified_pct: float = 0.35  # 35% Silicified
    
    # Operational Mode B (Table 1)
    mode_b_rate: float = 200  # ton/hr processing rate
    mode_b_cost: float = 24.9  # $/ton processing cost
    mode_b_recovery: float = 0.83  # 83% metal recovery
    mode_b_diorite_pct: float = 0.45  # 45% Diorite
    mode_b_silicified_pct: float = 0.55  # 55% Silicified

@dataclass
class HybridGALNSSAParameters:
    """Parameters for Algorithm 2 - Hybrid GA+LNS+SA with ε-constraint"""
    
    # GA parameters
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    
    # LNS parameters (Table 2)
    destruction_min: float = 0.15  # δ_min
    destruction_max: float = 0.30  # δ_max
    
    # SA parameters (Table 2)
    initial_temperature: float = 300.0  # T_0
    final_temperature: float = 1.0  # T_f
    cooling_rate: float = 0.95  # α
    iterations_per_temp: int = 3  # I_T
    
    # Epsilon-constraint parameters (Algorithm 2)
    epsilon_max: float = 0.1
    epsilon_threshold: float = 0.01
    
    # Neighborhood parameters
    n_neighborhoods: int = 5

@dataclass
class VAEParameters:
    """VAE parameters for geological scenario generation (Section 4.3-4.4)"""
    
    input_dim: int = 5  # [grade, rock_type, alteration, structure, distance]
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    beta: float = 0.5  # KL divergence weight
    lambda_geo: float = 0.1  # Geological constraint weight
    
    # Dynamic scenario generation
    min_scenarios: int = 50
    max_scenarios: int = 200

@dataclass
class RLAgentParameters:
    """Multi-agent RL parameters (Section 4.4)"""
    
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 64
    hidden_size: int = 256
    
    # Reward weights (Equation 3)
    alpha: float = 0.4  # NPV improvement weight
    beta: float = 0.3   # Constraint satisfaction weight
    gamma_eff: float = 0.2  # Efficiency weight
    delta: float = 0.1  # Risk penalty weight

@dataclass
class GPUParameters:
    """GPU configuration (Table 2, Algorithm 3-4)"""
    
    threads_per_block: int = 256
    max_blocks_per_grid: int = 1024
    batch_size: int = 50  # Blocks evaluated per GPU batch
    shared_memory_size: int = 49152  # 48KB
    
    # Enables 262,144 concurrent evaluations as per manuscript
    max_concurrent_evaluations: int = 262144

@dataclass
class DantzigWolfeParameters:
    """Enhanced Dantzig-Wolfe parameters (Algorithm 1)"""
    
    max_iterations: int = 100
    convergence_tolerance: float = 1e-4
    column_generation_limit: int = 50
    scenario_samples: int = 50  # Dynamic VAE scenario generation
    
    # Enhanced spatial uncertainty parameters (Equation 2)
    spatial_decay_factor: float = 0.95
    temporal_decay_factor: float = 0.92
    geological_weight: float = 0.15

@dataclass
class SpatialUncertaintyParameters:
    """Enhanced spatial uncertainty parameters (Equations 8-10)"""
    
    max_correlation_distance: float = 100.0  # meters
    correlation_range: float = 50.0  # for exponential correlation
    moran_i_threshold: float = 0.3  # for spatial autocorrelation
    
    # Geological feature weights (ψ_geological)
    alteration_weight: float = 0.4  # w₁
    structure_weight: float = 0.3   # w₂
    intrusion_weight: float = 0.3   # w₃

# Global Configuration
USE_GPU = True
RANDOM_SEED = 42
NUM_THREADS = -1  # Use all available cores

# Directory structure
OUTPUT_DIR = 'results'
DATA_DIR = 'data'
MODEL_DIR = 'models'
FIGURE_DIR = 'figures'

# Logging
VERBOSE = True
LOG_LEVEL = 'INFO'

def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get computing device (GPU/CPU)"""
    if torch.cuda.is_available() and USE_GPU:
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    return device

def setup_directories():
    """Create necessary directories"""
    for directory in [OUTPUT_DIR, DATA_DIR, MODEL_DIR, FIGURE_DIR]:
        os.makedirs(directory, exist_ok=True)

# Initialize on import
set_random_seeds(RANDOM_SEED)
setup_directories()
DEVICE = get_device()

# Default parameter instances
DEFAULT_MINING_PARAMS = MiningParameters()
DEFAULT_HYBRID_PARAMS = HybridGALNSSAParameters()
DEFAULT_VAE_PARAMS = VAEParameters()
DEFAULT_RL_PARAMS = RLAgentParameters()
DEFAULT_GPU_PARAMS = GPUParameters()
DEFAULT_DW_PARAMS = DantzigWolfeParameters()
DEFAULT_SPATIAL_PARAMS = SpatialUncertaintyParameters()