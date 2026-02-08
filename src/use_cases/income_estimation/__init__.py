"""
Income Estimation Use Case

A comprehensive income estimation system for financial decision-making that combines:
- Deposit pattern analysis for income detection
- Graph network analysis for validation and peer comparison
- Behavioral stability modeling and risk assessment
- Calibrated predictions with uncertainty quantification
- End-to-end pipeline for production deployment

Author: Principal Data Science Decision Agent
"""

from .deposit_intelligence import (
    DepositDetector,
    DepositPattern,
    IncomeSource
)

from .graph_payment import (
    PaymentNetworkAnalyzer,
    EmployerNode,
    PeerComparison,
    NetworkValidation
)

from .stability_model import (
    IncomeStabilityScorer,
    StabilityMetrics,
    RiskAssessment,
    TrendAnalysis
)

from .calibration import (
    ConformalPredictor,
    IsotonicCalibrator,
    BayesianCalibrator,
    QuantileRegressor,
    DynamicCalibrator,
    PredictionInterval,
    QuantilePrediction,
    CalibrationMetrics
)

from .income_pipeline import (
    IncomeEstimationPipeline,
    IncomeEstimationResult,
    ValidationMetrics
)

__all__ = [
    # Deposit Intelligence
    'DepositDetector',
    'DepositPattern',
    'IncomeSource',
    
    # Graph Payment Analysis
    'PaymentNetworkAnalyzer',
    'EmployerNode',
    'PeerComparison',
    'NetworkValidation',
    
    # Stability Modeling
    'IncomeStabilityScorer',
    'StabilityMetrics',
    'RiskAssessment',
    'TrendAnalysis',
    
    # Calibration
    'ConformalPredictor',
    'IsotonicCalibrator',
    'BayesianCalibrator',
    'QuantileRegressor',
    'DynamicCalibrator',
    'PredictionInterval',
    'QuantilePrediction',
    'CalibrationMetrics',
    
    # Pipeline
    'IncomeEstimationPipeline',
    'IncomeEstimationResult',
    'ValidationMetrics',
]

__version__ = '1.0.0'
