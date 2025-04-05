# Create a test script test_drift_detection.py
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from retrain import ModelRetrainer

class MockTrainModel:
    @staticmethod
    def train_model(data_path):
        """Mock train_model function that returns paths to model and test data"""
        # Extract directory from data_path
        data_dir = os.path.dirname(data_path)
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join(os.path.dirname(data_dir), "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Define paths
        model_path = os.path.join(model_dir, "model.joblib")
        test_data_path = os.path.join(data_dir, "test_data.npz")
        
        # Create a dummy model
        model = RandomForestClassifier(n_estimators=10)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        joblib.dump(model, model_path)
        
        # Create test data
        np.savez(test_data_path, X=X[:20], y=y[:20])
        
        return model_path, test_data_path

class MockEvaluateModel:
    @staticmethod
    def evaluate_model(model_path, test_data_path):
        """Mock evaluate_model function that returns metrics"""
        # Return mock metrics
        return {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.86,
            "f1": 0.85,
            "roc_auc": 0.90
        }

def test_performance_drift():
    retrainer = ModelRetrainer(performance_threshold=0.75)
    
    # Simulate good performance history
    good_metrics = [
        {"f1": 0.82, "roc_auc": 0.85},
        {"f1": 0.81, "roc_auc": 0.84},
        {"f1": 0.83, "roc_auc": 0.86}
    ]
    
    # Simulate performance drop below threshold
    bad_metrics = {"f1": 0.72, "roc_auc": 0.73}
    
    # Test 1: No drift with good metrics
    retrainer.metrics_history = [{"timestamp": "2023-01-01", "metrics": m} for m in good_metrics]
    drift_detected, details = retrainer.detect_performance_drift()
    print(f"Test 1 (good metrics): Drift detected? {drift_detected} (Should be False)")
    assert not drift_detected, "Should not detect drift with good metrics"
    
    # Test 2: Add bad metrics
    retrainer.metrics_history.append({"timestamp": "2023-01-02", "metrics": bad_metrics})
    drift_detected, details = retrainer.detect_performance_drift()
    print(f"Test 2 (below threshold): Drift detected? {drift_detected} (Should be True)")
    print("Details:", json.dumps(details, indent=2))
    assert drift_detected, "Should detect drift when metrics are below threshold"
    assert details["reason"] == "below_threshold", "Drift reason should be below_threshold"
    
    # Test 3: Significant decrease
    retrainer.metrics_history = [
        {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
        {"timestamp": "2023-01-02", "metrics": {"f1": 0.75, "roc_auc": 0.77}}
    ]
    drift_detected, details = retrainer.detect_performance_drift()
    print(f"Test 3 (significant drop): Drift detected? {drift_detected} (Should be True)")
    print("Details:", json.dumps(details, indent=2))
    assert drift_detected, "Should detect drift with significant performance drop"
    assert details["reason"] == "significant_decrease", "Drift reason should be significant_decrease"

@patch('retrain.train_model', MockTrainModel.train_model)
@patch('retrain.evaluate_model', MockEvaluateModel.evaluate_model)
def test_retraining_workflow():
    """Test the complete retraining workflow with simulated drift"""
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create required directory structure
        data_dir = tmpdir_path / "data" / "processed"
        model_dir = tmpdir_path / "model"
        monitor_dir = tmpdir_path / "monitoring"
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(monitor_dir, exist_ok=True)
        
        # Create test data files
        train_path = data_dir / "train.npz"
        test_path = data_dir / "test_data.npz"
        reference_path = data_dir / "reference_data.npz"
        model_path = model_dir / "model.joblib"
        metrics_history_path = monitor_dir / "metrics_history.json"
        
        # Create dummy train data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = np.array([f"feature_{i}" for i in range(5)])
        np.savez(train_path, X=X, y=y, feature_names=feature_names)
        
        # Create dummy test data
        np.savez(test_path, X=X[:20], y=y[:20], feature_names=feature_names)
        
        # Create dummy reference data
        np.savez(reference_path, X=X[:50], y=y[:50], feature_names=feature_names)
        
        # Initialize retrainer with customized paths
        custom_config = {
            'reference_data_path': str(reference_path),
            'model_path': str(model_path),
            'monitor_dir': str(monitor_dir),
            'history_file': str(monitor_dir / "drift_history.json"),
            'visualizations_dir': str(monitor_dir / "visualizations"),
        }
        
        retrainer = ModelRetrainer(
            model_path=str(model_path),
            test_data_path=str(test_path),
            processed_data_path=str(train_path),
            performance_threshold=0.8,
            metrics_history_path=str(metrics_history_path),
            monitoring_config=custom_config
        )
        
        # Set up the metrics history to simulate drift
        retrainer.metrics_history = [
            {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
            {"timestamp": "2023-01-02", "metrics": {"f1": 0.75, "roc_auc": 0.77}}  # Below threshold
        ]
        retrainer._save_metrics_history()
        
        print("\n=== Testing Performance Drift Scenario ===")
        result = retrainer.run_retraining_workflow()
        print("Actions:", result["actions_taken"])
        print("Conclusion:", result["conclusion"])
        
        # Verify retraining occurred
        assert "model_retraining" in result["actions_taken"], "Retraining should have occurred"
        assert result["conclusion"] == "Retraining successful", "Retraining should have succeeded"
        
        print("Test passed successfully")

@patch('retrain.train_model', MockTrainModel.train_model)
@patch('retrain.evaluate_model', MockEvaluateModel.evaluate_model)
def test_initial_training():
    """Test the initial training workflow when no model exists"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create required directory structure
        data_dir = tmpdir_path / "data" / "processed"
        os.makedirs(data_dir, exist_ok=True)
        
        # Create dummy train data
        train_path = data_dir / "train.npz"
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        np.savez(train_path, X=X, y=y)
        
        # Initialize retrainer with nonexistent model path to trigger initial training
        model_path = tmpdir_path / "model" / "model.joblib"
        test_path = data_dir / "test_data.npz"
        
        retrainer = ModelRetrainer(
            model_path=str(model_path),
            test_data_path=str(test_path),
            processed_data_path=str(train_path)
        )
        
        print("\n=== Testing Initial Training Scenario ===")
        result = retrainer.run_retraining_workflow()
        print("Actions:", result["actions_taken"])
        print("Conclusion:", result["conclusion"])
        
        # Verify initial training occurred
        assert "initial_training" in result["actions_taken"], "Initial training should have occurred"
        assert result["conclusion"] == "Initial training complete", "Initial training should have completed"
        
        print("Initial training test passed successfully")

if __name__ == "__main__":
    print("Running performance drift tests...")
    test_performance_drift()
    
    print("\nRunning initial training test...")
    test_initial_training()
    
    print("\nRunning retraining workflow test...")
    test_retraining_workflow()
    
    print("\nAll tests completed successfully!")