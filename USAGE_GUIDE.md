# Usage Guide: Principal Data Science Decision Agent

> **Complete guide to using the ML decision support framework for financial services**

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Core Components](#core-components)
4. [Use Case Examples](#use-case-examples)
5. [Advanced Features](#advanced-features)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sushil-tata/claude.git
cd claude

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo to verify installation
python demo.py
```

### Your First Analysis

```python
import sys
sys.path.insert(0, 'src')

from agent.decision_agent import DecisionAgent, ProblemDefinition, UseCase

# Initialize the agent
agent = DecisionAgent()

# Define your problem
problem = ProblemDefinition(
    use_case=UseCase.FRAUD_DETECTION,
    business_objective="Detect fraudulent transactions with <0.1% false positive rate",
    data_sources=["transactions", "devices", "merchants"],
    target_variable="is_fraud",
    evaluation_metrics=["auc", "precision", "recall"]
)

# Get decision recommendations
decision = agent.analyze_problem(problem)

# Generate report
report = agent.generate_report(decision)
print(report)
```

---

## 📖 Basic Usage

### 1. Understanding the Decision Agent

The Decision Agent provides structured recommendations following an 8-part framework:

1. **Problem Understanding** - Business objectives, loss function
2. **Data Architecture** - Required datasets, schema design
3. **Feature Blueprint** - Feature engineering strategy
4. **Modeling Blueprint** - Algorithm recommendations with trade-offs
5. **Optimization Strategy** - Hyperparameter tuning approach
6. **Validation Blueprint** - Testing and stability checks
7. **Simulation & Policy** - Economic impact modeling
8. **Production Design** - Deployment and monitoring

### 2. Working with Data

```python
from data.data_loader import DataLoader
from data.data_quality import DataQualityChecker

# Load data
loader = DataLoader()
df = loader.load_csv("path/to/data.csv")

# Check data quality
quality_checker = DataQualityChecker()
quality_report = quality_checker.analyze(df)
print(quality_report.summary())
```

### 3. Feature Engineering

```python
from features.behavioral_features import BehavioralFeatureEngine
from features.temporal_features import TemporalFeatureEngine

# Create behavioral features
behavioral_engine = BehavioralFeatureEngine()
behavioral_features = behavioral_engine.create_features(
    df=transactions_df,
    windows=[7, 14, 30, 60, 90],
    metrics=['velocity', 'momentum', 'volatility']
)

# Create temporal features
temporal_engine = TemporalFeatureEngine()
temporal_features = temporal_engine.create_features(
    df=transactions_df,
    rolling_windows=[7, 30, 90],
    lag_periods=[1, 7, 14]
)
```

### 4. Model Training

```python
from models.tree_models import LightGBMModel
from agent.orchestrator import ModelOrchestrator, ModelCandidate

# Single model approach
model = LightGBMModel()
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)

# Multi-model orchestration (recommended)
orchestrator = ModelOrchestrator()

# Add candidate models
from models.tree_models import XGBoostModel, CatBoostModel

orchestrator.add_candidate(ModelCandidate(
    name="LightGBM",
    model_type="tree",
    model_instance=LightGBMModel()
))

orchestrator.add_candidate(ModelCandidate(
    name="XGBoost", 
    model_type="tree",
    model_instance=XGBoostModel()
))

# Train all candidates
orchestrator.train_all_candidates(X_train, y_train, X_val, y_val)

# Evaluate and select champion
results = orchestrator.evaluate_all_candidates(y_val)
champion, challengers = orchestrator.select_champion_challengers(
    primary_metric='auc',
    n_challengers=2
)

print(orchestrator.generate_recommendation_report())
```

---

## 🎯 Use Case Examples

### Use Case 1: Collections NBA (Next Best Action)

**Objective**: Optimize collections recovery by recommending the best action, channel, and timing for each customer.

```python
from use_cases.collections_nba import NBAPipeline
import pandas as pd

# Initialize pipeline
nba_pipeline = NBAPipeline()

# Prepare customer data
customer_data = pd.DataFrame({
    'customer_id': [1001, 1002, 1003],
    'days_past_due': [30, 60, 90],
    'outstanding_balance': [5000, 15000, 25000],
    'payment_history_score': [0.7, 0.4, 0.2],
    'previous_contacts': [2, 5, 10],
    'last_payment_days_ago': [45, 90, 180]
})

# Get recommendations
recommendations = nba_pipeline.get_recommendations(customer_data)

# Results include:
# - propensity_to_pay: Probability of repayment
# - expected_payment_amount: Predicted payment
# - recommended_treatment: Legal/Settlement/Restructuring/etc.
# - recommended_channel: Email/SMS/Call/Letter
# - optimal_timing: Best time to contact

print(recommendations[['customer_id', 'recommended_treatment', 'expected_recovery']])
```

**Advanced NBA Usage:**

```python
from use_cases.collections_nba import (
    PropensityModel,
    PaymentEstimator,
    TreatmentOptimizer,
    ChannelOptimizer
)

# Step-by-step approach for customization

# 1. Predict propensity to pay
propensity_model = PropensityModel()
propensity_model.train(historical_data, target='repaid')
propensity_scores = propensity_model.predict(customer_data)

# 2. Estimate payment amount
payment_estimator = PaymentEstimator()
payment_estimator.train(historical_data, target='payment_amount')
payment_estimates = payment_estimator.predict(customer_data)

# 3. Optimize treatment strategy
treatment_optimizer = TreatmentOptimizer()
optimal_treatments = treatment_optimizer.recommend(
    customer_data=customer_data,
    propensity_scores=propensity_scores,
    payment_estimates=payment_estimates,
    constraints={'max_legal_actions_per_day': 100}
)

# 4. Optimize channel and timing
channel_optimizer = ChannelOptimizer()
channel_recommendations = channel_optimizer.optimize(
    customers=customer_data,
    treatments=optimal_treatments,
    channel_costs={'email': 0.1, 'sms': 0.5, 'call': 5.0, 'letter': 2.0}
)
```

### Use Case 2: Fraud Detection

**Objective**: Detect fraudulent transactions in real-time using graph-based features and anomaly detection.

```python
from use_cases.fraud_detection import FraudDetectionPipeline
import pandas as pd

# Initialize pipeline
fraud_pipeline = FraudDetectionPipeline()

# Prepare transaction data
transactions = pd.DataFrame({
    'transaction_id': [1, 2, 3],
    'customer_id': [101, 102, 103],
    'merchant_id': [501, 502, 503],
    'amount': [100.0, 5000.0, 50.0],
    'device_id': ['dev1', 'dev2', 'dev3'],
    'timestamp': pd.date_range('2024-01-01', periods=3, freq='h')
})

# Score transactions
fraud_scores = fraud_pipeline.score_transactions(transactions)

# Results include:
# - fraud_probability: Supervised model score
# - anomaly_score: Unsupervised anomaly detection
# - risk_propagation_score: Network-based risk
# - in_fraud_ring: Boolean indicating fraud ring membership

high_risk = fraud_scores[fraud_scores['fraud_probability'] > 0.9]
print(f"Found {len(high_risk)} high-risk transactions")
```

**Advanced Fraud Detection:**

```python
from use_cases.fraud_detection import (
    GraphBuilder,
    SupervisedFraudDetector,
    AnomalyDetector,
    RiskPropagator
)

# 1. Build transaction graph
graph_builder = GraphBuilder()
graph = graph_builder.build_graph(
    transactions=transactions,
    include_merchants=True,
    include_devices=True,
    time_window_days=30
)

# 2. Extract graph features
graph_features = graph_builder.extract_features(graph)

# 3. Supervised fraud detection
fraud_detector = SupervisedFraudDetector()
fraud_detector.train(historical_data, target='is_fraud')
supervised_scores = fraud_detector.predict(transactions, graph_features)

# 4. Anomaly detection (unsupervised)
anomaly_detector = AnomalyDetector(method='isolation_forest')
anomaly_scores = anomaly_detector.detect(transactions, graph_features)

# 5. Risk propagation through network
risk_propagator = RiskPropagator()
propagated_risk = risk_propagator.propagate(
    graph=graph,
    initial_scores=supervised_scores,
    iterations=10
)

# Combine all signals
final_scores = (
    0.5 * supervised_scores +
    0.3 * anomaly_scores +
    0.2 * propagated_risk
)
```

### Use Case 3: Behavioral Scoring

**Objective**: Score customers based on transaction behavior patterns.

```python
from use_cases.behavioral_scoring import BehavioralScoringPipeline
import pandas as pd

# Initialize pipeline
scoring_pipeline = BehavioralScoringPipeline()

# Prepare transaction history
transaction_history = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2],
    'transaction_date': pd.date_range('2024-01-01', periods=5, freq='D'),
    'amount': [100, 200, 150, 50, 75],
    'merchant_category': ['groceries', 'gas', 'restaurant', 'groceries', 'utilities']
})

# Score customers
credit_scores = scoring_pipeline.score_customers(transaction_history)

# Results include:
# - behavioral_score: 0-1000 score
# - stability_score: Behavioral stability metric
# - confidence_interval: [lower, upper] bounds

print(credit_scores[['customer_id', 'behavioral_score', 'risk_category']])
```

### Use Case 4: Income Estimation

**Objective**: Estimate customer income from transaction patterns.

```python
from use_cases.income_estimation import IncomeEstimationPipeline
import pandas as pd

# Initialize pipeline
income_pipeline = IncomeEstimationPipeline()

# Prepare customer transactions
customer_transactions = pd.DataFrame({
    'customer_id': [1, 1, 1],
    'transaction_date': pd.date_range('2024-01-01', periods=3, freq='15D'),
    'amount': [3500, 3500, 3500],
    'transaction_type': ['deposit', 'deposit', 'deposit'],
    'description': ['Salary - ACME Corp', 'Salary - ACME Corp', 'Salary - ACME Corp']
})

# Estimate income
income_estimates = income_pipeline.estimate_income(customer_transactions)

# Results include:
# - estimated_monthly_income: Predicted income
# - confidence_interval_80: [lower, upper] 80% confidence
# - confidence_interval_90: [lower, upper] 90% confidence
# - income_stability_score: Stability metric
# - income_sources: Breakdown by source (salary, gig, investment)

print(income_estimates[['customer_id', 'estimated_monthly_income', 'income_stability_score']])
```

---

## 🔬 Advanced Features

### Simulation & Scenario Analysis

```python
from simulation.monte_carlo import MonteCarloSimulator
from simulation.policy_simulator import PolicySimulator

# Monte Carlo simulation for recovery scenarios
simulator = MonteCarloSimulator()
results = simulator.simulate_repayment(
    accounts=customer_data,
    scenarios=['baseline', 'optimistic', 'pessimistic'],
    n_simulations=10000
)

print(f"Expected recovery: ${results['expected_value']:.2f}")
print(f"95% confidence interval: ${results['ci_95'][0]:.2f} - ${results['ci_95'][1]:.2f}")

# Policy impact simulation
policy_sim = PolicySimulator()
impact = policy_sim.simulate_policy_change(
    current_policy={'max_contacts': 5},
    new_policy={'max_contacts': 3},
    population=customer_data
)

print(f"Estimated impact: {impact['delta_recovery']:.1%}")
```

### Model Validation

```python
from validation.stability_testing import StabilityTester
from validation.drift_monitor import DriftMonitor
from validation.calibration_validator import CalibrationValidator

# Stability testing (PSI, CSI)
stability_tester = StabilityTester()
stability_report = stability_tester.test_stability(
    reference_data=train_data,
    test_data=oot_data,
    features=feature_list
)

if stability_report['overall_status'] == 'RED':
    print("⚠️ Model stability issues detected!")

# Drift monitoring
drift_monitor = DriftMonitor()
drift_detected = drift_monitor.detect_drift(
    reference=train_data,
    current=production_data,
    threshold=0.05
)

# Calibration validation
calibrator = CalibrationValidator()
calibration_report = calibrator.validate(
    y_true=y_test,
    y_pred=predictions,
    n_bins=10
)

print(f"Brier Score: {calibration_report['brier_score']:.4f}")
print(f"Hosmer-Lemeshow p-value: {calibration_report['hl_pvalue']:.4f}")
```

### Production Deployment

```python
from production.deployment import ModelDeployer
from production.monitoring import ModelMonitor
from production.feature_serving import FeatureServer

# Deploy model
deployer = ModelDeployer()
deployment = deployer.deploy_model(
    model=champion_model,
    deployment_type='rest_api',
    environment='production',
    health_check=True
)

print(f"Model deployed at: {deployment.endpoint}")

# Setup monitoring
monitor = ModelMonitor()
monitor.start_monitoring(
    model_id=deployment.model_id,
    metrics=['auc', 'precision', 'recall'],
    drift_detection=True,
    alert_threshold={'auc_drop': 0.05}
)

# Feature serving for real-time inference
feature_server = FeatureServer()
feature_server.start(
    cache_type='redis',
    ttl_seconds=300
)

# Real-time prediction
features = feature_server.get_features(customer_id=12345)
prediction = deployment.predict(features)
```

---

## ⚙️ Configuration

### Agent Configuration (config/agent_config.yaml)

```yaml
agent:
  name: "Principal Data Science Decision Agent"
  operating_rules:
    - "Always Think Multi-Model"
    - "Always Check Temporal Robustness"
    - "Always Recommend Challenger Designs"
    - "Always Explain Trade-offs"
    - "Always Optimize for Business Value"

orchestration:
  max_models: 10
  parallel_execution: true
  max_workers: 4

validation:
  oot_months: 3
  min_segment_size: 1000
  psi_threshold: 0.25
```

### Model Configuration (config/model_config.yaml)

```yaml
lightgbm:
  default:
    boosting_type: 'gbdt'
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
  
  search_space:
    num_leaves: [15, 31, 63, 127]
    learning_rate: [0.01, 0.05, 0.1]
    max_depth: [5, 7, 10, -1]
```

### Feature Configuration (config/feature_config.yaml)

```yaml
behavioral:
  velocity_windows: [7, 14, 30, 60, 90]
  metrics: ['mean', 'median', 'std', 'cv']

temporal:
  rolling_windows:
    short_term: [7, 14]
    medium_term: [30, 60]
    long_term: [90, 180]
```

---

## 💡 Best Practices

### 1. Always Use Champion-Challenger Framework

```python
# Don't: Single model
model = LightGBMModel()
model.fit(X_train, y_train)

# Do: Multiple models with comparison
orchestrator = ModelOrchestrator()
orchestrator.add_candidate(ModelCandidate("LightGBM", "tree", LightGBMModel()))
orchestrator.add_candidate(ModelCandidate("XGBoost", "tree", XGBoostModel()))
orchestrator.add_candidate(ModelCandidate("CatBoost", "tree", CatBoostModel()))

orchestrator.train_all_candidates(X_train, y_train, X_val, y_val)
champion, challengers = orchestrator.select_champion_challengers()
```

### 2. Validate on Out-of-Time Data

```python
# Always split data by time, not randomly
train_data = df[df['date'] < '2024-01-01']
val_data = df[(df['date'] >= '2024-01-01') & (df['date'] < '2024-04-01')]
oot_data = df[df['date'] >= '2024-04-01']

# Validate stability
stability_tester.test_stability(reference_data=train_data, test_data=oot_data)
```

### 3. Monitor Sub-Segment Performance

```python
# Check performance across segments
for segment in ['age_<30', 'age_30-50', 'age_50+']:
    segment_data = test_data[test_data['segment'] == segment]
    segment_auc = roc_auc_score(segment_data['y_true'], segment_data['y_pred'])
    print(f"{segment}: AUC = {segment_auc:.3f}")
```

### 4. Use Feature Store for Leakage Prevention

```python
from features.feature_store import FeatureStore

feature_store = FeatureStore()
feature_store.register_feature(
    name='avg_transaction_amount_30d',
    calculation_logic=lambda df: df.rolling(30).mean(),
    leakage_check=True
)

# Automatic leakage detection
leakage_report = feature_store.check_leakage(features, target)
if leakage_report['high_risk_features']:
    print(f"⚠️ Potential leakage: {leakage_report['high_risk_features']}")
```

### 5. Document Everything for Governance

```python
from validation.governance_report import GovernanceReporter

reporter = GovernanceReporter()
model_card = reporter.generate_model_card(
    model=champion_model,
    training_data=train_data,
    performance_metrics=results,
    fairness_analysis=fairness_report,
    use_case="Collections NBA"
)

# Save for regulatory review
model_card.save('model_cards/collections_nba_v1.md')
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: Import Error

```python
# Error: No module named 'agent'
# Solution: Add src to Python path
import sys
sys.path.insert(0, 'src')
```

#### Issue: Missing Dependencies

```bash
# Error: ModuleNotFoundError: No module named 'pandas'
# Solution: Install requirements
pip install -r requirements.txt
```

#### Issue: Configuration Not Found

```python
# Error: Config file not found
# Solution: Ensure you're running from repository root
import os
print(os.getcwd())  # Should be /path/to/claude
```

#### Issue: Model Performance Issues

```python
# Low AUC or poor metrics
# 1. Check feature quality
from data.data_quality import DataQualityChecker
quality_checker.analyze(df)

# 2. Check for data leakage
from features.feature_store import FeatureStore
feature_store.check_leakage(features, target)

# 3. Try different models
orchestrator.add_candidate(ModelCandidate("Neural", "neural", TabNetModel()))

# 4. Tune hyperparameters
from models.meta_learner import BayesianOptimizer
optimizer = BayesianOptimizer()
best_params = optimizer.optimize(model, X_train, y_train)
```

#### Issue: Memory Errors with Large Datasets

```python
# Use chunking for large files
from data.data_loader import DataLoader

loader = DataLoader()
for chunk in loader.load_csv_chunks('large_file.csv', chunksize=10000):
    process(chunk)
```

---

## 📚 Additional Resources

- **README.md** - Project overview and quick start
- **docs/** - Detailed documentation for each component
- **examples/** - Working code examples
- **FINAL_STATUS.md** - Project completion status
- **demo.py** - Quick demo script

### Getting Help

1. Check the documentation in `docs/`
2. Review examples in `examples/`
3. Run `python demo.py` to verify setup
4. Run `python status_check.py` for health check
5. Open an issue on GitHub for bugs or questions

---

## 🎓 Next Steps

1. **Try the Quick Start** - Get the agent running
2. **Explore Use Cases** - Pick a use case relevant to your needs
3. **Customize Configuration** - Adjust YAML configs for your environment
4. **Build Your Pipeline** - Combine components for your specific problem
5. **Deploy to Production** - Use production modules for deployment

**Happy modeling! 🚀**
