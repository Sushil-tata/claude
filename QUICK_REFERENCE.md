# Quick Reference Guide

> **Fast lookup for common tasks with the Principal Data Science Decision Agent**

## 🚀 Quick Commands

```bash
# Install
pip install -r requirements.txt

# Run demo
python demo.py

# Check status
python status_check.py
```

## 📝 Common Code Snippets

### Initialize Agent

```python
import sys
sys.path.insert(0, 'src')
from agent.decision_agent import DecisionAgent

agent = DecisionAgent()
```

### Load Data

```python
from data.data_loader import DataLoader

loader = DataLoader()
df = loader.load_csv("data.csv")
```

### Create Features

```python
from features.behavioral_features import BehavioralFeatureEngine

engine = BehavioralFeatureEngine()
features = engine.create_features(df, windows=[7, 30, 90])
```

### Train Model

```python
from models.tree_models import LightGBMModel

model = LightGBMModel()
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)
```

### Multi-Model Comparison

```python
from agent.orchestrator import ModelOrchestrator, ModelCandidate
from models.tree_models import LightGBMModel, XGBoostModel

orchestrator = ModelOrchestrator()
orchestrator.add_candidate(ModelCandidate("LightGBM", "tree", LightGBMModel()))
orchestrator.add_candidate(ModelCandidate("XGBoost", "tree", XGBoostModel()))

orchestrator.train_all_candidates(X_train, y_train, X_val, y_val)
results = orchestrator.evaluate_all_candidates(y_val)
champion, challengers = orchestrator.select_champion_challengers()
```

## 🎯 Use Case Templates

### Collections NBA

```python
from use_cases.collections_nba import NBAPipeline

pipeline = NBAPipeline()
recommendations = pipeline.get_recommendations(customer_data)
```

### Fraud Detection

```python
from use_cases.fraud_detection import FraudDetectionPipeline

pipeline = FraudDetectionPipeline()
scores = pipeline.score_transactions(transactions)
```

### Behavioral Scoring

```python
from use_cases.behavioral_scoring import BehavioralScoringPipeline

pipeline = BehavioralScoringPipeline()
scores = pipeline.score_customers(transaction_history)
```

### Income Estimation

```python
from use_cases.income_estimation import IncomeEstimationPipeline

pipeline = IncomeEstimationPipeline()
estimates = pipeline.estimate_income(customer_transactions)
```

## 🔬 Validation & Testing

### Stability Testing

```python
from validation.stability_testing import StabilityTester

tester = StabilityTester()
report = tester.test_stability(train_data, oot_data, features)
```

### Drift Detection

```python
from validation.drift_monitor import DriftMonitor

monitor = DriftMonitor()
drift = monitor.detect_drift(reference_data, current_data)
```

### Calibration

```python
from validation.calibration_validator import CalibrationValidator

validator = CalibrationValidator()
report = validator.validate(y_true, y_pred)
```

## 🚀 Production Deployment

### Deploy Model

```python
from production.deployment import ModelDeployer

deployer = ModelDeployer()
deployment = deployer.deploy_model(model, deployment_type='rest_api')
```

### Monitor Model

```python
from production.monitoring import ModelMonitor

monitor = ModelMonitor()
monitor.start_monitoring(model_id, metrics=['auc', 'precision'])
```

### Feature Serving

```python
from production.feature_serving import FeatureServer

server = FeatureServer()
server.start(cache_type='redis')
features = server.get_features(customer_id=123)
```

## 📊 Simulation

### Monte Carlo

```python
from simulation.monte_carlo import MonteCarloSimulator

sim = MonteCarloSimulator()
results = sim.simulate_repayment(accounts, scenarios=['baseline', 'stress'])
```

### Policy Impact

```python
from simulation.policy_simulator import PolicySimulator

sim = PolicySimulator()
impact = sim.simulate_policy_change(current_policy, new_policy, population)
```

## ⚙️ Key Configuration Paths

- **Agent Config**: `config/agent_config.yaml`
- **Model Config**: `config/model_config.yaml`
- **Feature Config**: `config/feature_config.yaml`

## 🔍 Troubleshooting Quick Fixes

```python
# Import error - add src to path
import sys
sys.path.insert(0, 'src')

# Check data quality
from data.data_quality import DataQualityChecker
checker = DataQualityChecker()
report = checker.analyze(df)

# Check for leakage
from features.feature_store import FeatureStore
store = FeatureStore()
leakage = store.check_leakage(features, target)
```

## 📚 More Info

See **USAGE_GUIDE.md** for detailed explanations and examples.
