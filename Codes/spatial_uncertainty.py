# spatial_uncertainty.py
"""
Enhanced spatial uncertainty quantification for mining blocks
Implements spatial autocorrelation and local variance analysis
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from typing import Dict, Tuple

class EnhancedSpatialUncertainty:
    """
    Quantify and model spatial uncertainty in geological attributes
    
    Methods:
    - Moran's I spatial autocorrelation
    - Local variance estimation
    - Enhanced uncertainty with temporal decay
    - Spatial weight matrix computation
    """
    
    def __init__(self, coordinates: np.ndarray, grades: np.ndarray, 
                 geological_features: Dict):
        """
        Initialize spatial uncertainty model
        
        Args:
            coordinates: Block coordinates (n_blocks, 3)
            grades: Gold grades (n_blocks,)
            geological_features: Dict with geological properties
        """
        self.coordinates = coordinates
        self.grades = grades
        self.features = geological_features
        self.weights = None
        self.morans_i = None
    
    def compute_spatial_weights(self, max_distance: float = 100) -> np.ndarray:
        """
        Compute spatial weight matrix using inverse distance weighting
        
        W[i,j] = 1/d[i,j] if d[i,j] < max_distance, else 0
        Row-normalized so sum(W[i,:]) = 1
        """
        # Compute pairwise distances
        distances = cdist(self.coordinates, self.coordinates, metric='euclidean')
        
        # Initialize weight matrix
        W = np.zeros_like(distances)
        
        # Apply inverse distance weighting
        mask = (distances < max_distance) & (distances > 0)
        W[mask] = 1.0 / (distances[mask] + 1e-6)
        
        # Row normalize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums
        
        self.weights = W
        return W
    
    def compute_morans_i(self) -> float:
        """
        Compute Moran's I spatial autocorrelation statistic
        
        I = (n / S0) * (Σ Σ w[i,j] * (x[i] - x_mean) * (x[j] - x_mean)) / Σ(x[i] - x_mean)²
        
        Returns:
            Moran's I value (-1 to +1)
            +1: Perfect positive spatial correlation
             0: Random spatial pattern
            -1: Perfect negative spatial correlation
        """
        if self.weights is None:
            self.compute_spatial_weights()
        
        n = len(self.grades)
        mean_grade = np.mean(self.grades)
        
        # Deviations from mean
        deviations = self.grades - mean_grade
        
        # Numerator: weighted cross-product of deviations
        numerator = np.sum(
            self.weights * np.outer(deviations, deviations)
        )
        
        # Denominator: sum of squared deviations
        denominator = np.sum(deviations ** 2)
        
        # Sum of weights
        S0 = np.sum(self.weights)
        
        # Moran's I
        if denominator > 0 and S0 > 0:
            I = (n / S0) * (numerator / denominator)
        else:
            I = 0
        
        self.morans_i = I
        return I
    
    def compute_local_variance(self, block_idx: int, k: int = 10) -> float:
        """
        Compute local grade variance for a specific block
        Uses k nearest neighbors
        
        Args:
            block_idx: Index of target block
            k: Number of nearest neighbors
        
        Returns:
            Local variance
        """
        # Find k nearest neighbors
        distances = np.linalg.norm(
            self.coordinates - self.coordinates[block_idx], 
            axis=1
        )
        
        # Exclude self (distance=0)
        nearest_indices = np.argsort(distances)[1:k+1]
        
        # Compute local variance
        local_grades = self.grades[nearest_indices]
        local_variance = np.var(local_grades)
        
        return local_variance
    
    def compute_local_morans_i(self, block_idx: int) -> float:
        """
        Compute Local Moran's I (LISA) for a specific block
        Measures local spatial autocorrelation
        """
        if self.weights is None:
            self.compute_spatial_weights()
        
        mean_grade = np.mean(self.grades)
        deviation_i = self.grades[block_idx] - mean_grade
        
        # Weighted sum of neighbor deviations
        neighbor_deviations = self.grades - mean_grade
        weighted_sum = np.sum(
            self.weights[block_idx, :] * neighbor_deviations
        )
        
        # Variance
        variance = np.var(self.grades)
        
        if variance > 0:
            local_I = deviation_i * weighted_sum / variance
        else:
            local_I = 0
        
        return local_I
    
    def compute_enhanced_uncertainty(
        self, 
        block_idx: int, 
        period: int,
        temporal_decay: float = 0.1,
        w1: float = 0.3,
        w2: float = 0.3,
        w3: float = 0.4
    ) -> float:
        """
        Compute enhanced uncertainty for a block in a given period
        
        σ_enhanced(s,t) = f_spatial × φ_temporal × ψ_geological
        
        Components:
        - f_spatial: Spatial component (Moran's I + local variance)
        - φ_temporal: Temporal decay (uncertainty increases over time)
        - ψ_geological: Geological features weighting
        
        Args:
            block_idx: Block index
            period: Time period
            temporal_decay: Rate of temporal uncertainty increase
            w1, w2, w3: Weights for geological features
        
        Returns:
            Enhanced uncertainty value
        """
        # Spatial component
        if self.morans_i is None:
            self.compute_morans_i()
        
        sigma_local = self.compute_local_variance(block_idx, k=10)
        f_spatial = (1 - abs(self.morans_i)) + sigma_local
        
        # Temporal component (uncertainty increases over time)
        phi_temporal = np.exp(temporal_decay * period)
        
        # Geological features component
        psi_geological = (
            w1 * self.features['alteration_intensity'][block_idx] +
            w2 * self.features['structural_density'][block_idx] +
            w3 * self.features['distance_to_intrusion'][block_idx]
        )
        
        # Combined enhanced uncertainty
        sigma_enhanced = f_spatial * phi_temporal * psi_geological
        
        return sigma_enhanced
    
    def compute_uncertainty_map(
        self,
        period: int = 0,
        temporal_decay: float = 0.1
    ) -> np.ndarray:
        """
        Compute uncertainty map for all blocks
        
        Returns:
            Array of uncertainty values (n_blocks,)
        """
        uncertainties = np.zeros(len(self.coordinates))
        
        for i in range(len(self.coordinates)):
            uncertainties[i] = self.compute_enhanced_uncertainty(
                i, period, temporal_decay
            )
        
        return uncertainties
    
    def get_high_uncertainty_blocks(
        self,
        period: int = 0,
        threshold_percentile: float = 75
    ) -> np.ndarray:
        """
        Identify blocks with high uncertainty
        
        Args:
            period: Time period
            threshold_percentile: Percentile threshold (e.g., 75 = top 25%)
        
        Returns:
            Indices of high-uncertainty blocks
        """
        uncertainties = self.compute_uncertainty_map(period)
        threshold = np.percentile(uncertainties, threshold_percentile)
        
        high_uncertainty_blocks = np.where(uncertainties >= threshold)[0]
        
        return high_uncertainty_blocks
    
    def compute_spatial_correlation_matrix(self) -> np.ndarray:
        """
        Compute spatial correlation matrix between all blocks
        Uses exponential decay with distance
        """
        distances = cdist(self.coordinates, self.coordinates, metric='euclidean')
        
        # Exponential correlation: exp(-distance/range)
        correlation_range = 50  # meters
        correlation_matrix = np.exp(-distances / correlation_range)
        
        return correlation_matrix
    
    def quantify_risk(
        self,
        block_idx: int,
        period: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Quantify risk using Value at Risk (VaR) approach
        
        Returns:
            (expected_value, var, cvar)
            - expected_value: Mean estimate
            - var: Value at Risk (percentile)
            - cvar: Conditional Value at Risk (tail average)
        """
        # Enhanced uncertainty
        sigma = self.compute_enhanced_uncertainty(block_idx, period)
        
        # Expected value (mean grade)
        expected_value = self.grades[block_idx]
        
        # Value at Risk (percentile)
        z_score = norm.ppf(1 - confidence_level)
        var = expected_value + z_score * sigma
        
        # Conditional Value at Risk (expected shortfall)
        cvar = expected_value - sigma * (
            norm.pdf(z_score) / (1 - confidence_level)
        )
        
        return expected_value, var, cvar

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def analyze_spatial_structure(coordinates: np.ndarray, 
                              grades: np.ndarray,
                              features: Dict) -> Dict:
    """
    Comprehensive spatial structure analysis
    
    Returns dictionary with:
    - Moran's I
    - Average local variance
    - Spatial range
    - Uncertainty statistics
    """
    uncertainty_model = EnhancedSpatialUncertainty(
        coordinates, grades, features
    )
    
    # Compute spatial statistics
    uncertainty_model.compute_spatial_weights(max_distance=100)
    morans_i = uncertainty_model.compute_morans_i()
    
    # Local variances
    local_variances = [
        uncertainty_model.compute_local_variance(i, k=10)
        for i in range(min(100, len(coordinates)))  # Sample for efficiency
    ]
    
    # Uncertainty map
    uncertainty_map = uncertainty_model.compute_uncertainty_map(period=0)
    
    analysis = {
        'morans_i': morans_i,
        'spatial_autocorrelation': 'Positive' if morans_i > 0 else 'Negative',
        'avg_local_variance': np.mean(local_variances),
        'uncertainty_mean': np.mean(uncertainty_map),
        'uncertainty_std': np.std(uncertainty_map),
        'uncertainty_range': (np.min(uncertainty_map), np.max(uncertainty_map))
    }
    
    return analysis

# ============================================================================
# MAIN: Test spatial uncertainty module
# ============================================================================

if __name__ == '__main__':
    from data_generator import MiningDataGenerator
    
    print("=" * 80)
    print("SPATIAL UNCERTAINTY ANALYSIS")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MiningDataGenerator.load_dataset()
    
    # Create uncertainty model
    print("Initializing spatial uncertainty model...")
    uncertainty_model = EnhancedSpatialUncertainty(
        coordinates=dataset['coordinates'],
        grades=dataset['base_grades'],
        geological_features=dataset['geological_features']
    )
    
    # Compute spatial weights
    print("\nComputing spatial weights...")
    weights = uncertainty_model.compute_spatial_weights(max_distance=100)
    print(f"Weight matrix shape: {weights.shape}")
    print(f"Average neighbors per block: {(weights > 0).sum(axis=1).mean():.2f}")
    
    # Compute Moran's I
    print("\nComputing Moran's I...")
    morans_i = uncertainty_model.compute_morans_i()
    print(f"Moran's I: {morans_i:.4f}")
    if morans_i > 0.3:
        print("Strong positive spatial autocorrelation detected")
    elif morans_i > 0:
        print("Weak positive spatial autocorrelation detected")
    else:
        print("Negative or no spatial autocorrelation")
    
    # Test enhanced uncertainty
    print("\nComputing enhanced uncertainty for sample blocks...")
    for block_idx in [0, 100, 500, 1000]:
        if block_idx < len(dataset['coordinates']):
            uncertainty = uncertainty_model.compute_enhanced_uncertainty(
                block_idx, period=0
            )
            local_var = uncertainty_model.compute_local_variance(block_idx)
            print(f"Block {block_idx:4d}: "
                  f"Uncertainty={uncertainty:.4f}, "
                  f"Local Var={local_var:.6f}")
    
    # Comprehensive analysis
    print("\n" + "-" * 80)
    print("COMPREHENSIVE SPATIAL ANALYSIS")
    print("-" * 80)
    
    analysis = analyze_spatial_structure(
        dataset['coordinates'],
        dataset['base_grades'],
        dataset['geological_features']
    )
    
    for key, value in analysis.items():
        print(f"{key:25s}: {value}")
    
    print("\n" + "=" * 80)
    print("SPATIAL UNCERTAINTY ANALYSIS COMPLETE")
    print("=" * 80)