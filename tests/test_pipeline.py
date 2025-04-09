import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
import json
from pathlib import Path
import datetime
import joblib
from src.retrain import ModelRetrainer
from sklearn.ensemble import RandomForestClassifier
import pytest

class TestDriftDetection:
    @pytest.fixture
    def setup_temp_environment(self, tmp_path):
        """Create temporary directory with test data for isolated testing"""
        # Use environment variables for base path if available
        base_path = os.getenv('BASE_PATH', tmp_path)
        
        # Create directory structure
        model_dir = Path(base_path) / "model"
        data_dir = Path(base_path) / "data" / "processed"
        monitoring_dir = Path(base_path) / "monitoring"
        versions_dir = Path(base_path) / "versions"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup reference/baseline data
        n_samples = 1000
        n_features = 10
        np.random.seed(42)
        
        # Create reference data with clear distribution
        X_ref = np.random.normal(0, 1, (n_samples, n_features))
        y_ref = (X_ref[:, 0] + X_ref[:, 1] > 0).astype(int)
        
        # Save reference data
        np.savez(data_dir / "reference_data.npz", X=X_ref, y=y_ref, 
                 feature_names=np.array([f"feature_{i}" for i in range(n_features)]))
        
        # Train and save a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_ref, y_ref)
        model_path = model_dir / "test_model.joblib"
        joblib.dump(model, model_path)
        
        # Create test data (initially similar to reference)
        X_test = np.random.normal(0, 1, (200, n_features))
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
        np.savez(data_dir / "test_data.npz", X=X_test, y=y_test, 
                 feature_names=np.array([f"feature_{i}" for i in range(n_features)]))
        
        # Training data for retraining
        np.savez(data_dir / "train.npz", X=X_ref, y=y_ref, 
                 feature_names=np.array([f"feature_{i}" for i in range(n_features)]))
        
        # Create empty metrics history file
        with open(monitoring_dir / "metrics_history.json", "w") as f:
            json.dump([], f)
            
        # Create test config
        config = {
            'reference_data_path': str(data_dir / "reference_data.npz"),
            'model_path': str(model_path),
            'monitor_dir': str(monitoring_dir),
            'history_file': str(monitoring_dir / "drift_history.json"),
            'visualizations_dir': str(monitoring_dir / "visualizations"),
            'versions_dir': str(versions_dir),
            'drift_thresholds': {
                'data_drift_threshold': 0.1,
                'concept_drift_threshold': 0.05,
                'label_drift_threshold': 0.1,
                'significant_drift_ratio': 0.3
            }
        }
        
        return {
            'tmp_path': Path(base_path),
            'model_path': model_path,
            'test_data_path': data_dir / "test_data.npz",
            'reference_data_path': data_dir / "reference_data.npz",
            'processed_data_path': data_dir / "train.npz",
            'config': config,
            'n_features': n_features
        }
    
    def test_data_drift_detection(self, setup_temp_environment):
        """Test detection of data drift when feature distributions change"""
        env = setup_temp_environment
        
        # Initialize retrainer with test configuration
        retrainer = ModelRetrainer(
            model_path=str(env['model_path']),
            test_data_path=str(env['test_data_path']),
            processed_data_path=str(env['processed_data_path']),
            monitoring_config=env['config']
        )
        
        # Test 1: No drift with similar data distribution
        drift_results, drift_detected = retrainer.check_for_data_drift()
        assert not drift_detected, "No drift should be detected with similar distributions"
        
        # Test 2: Create data with drift in feature distributions
        n_samples = 200
        n_features = env['n_features']
        
        # Load original test data
        orig_data = np.load(env['test_data_path'])
        X_orig = orig_data['X']
        y_orig = orig_data['y']
        feature_names = orig_data['feature_names']
        
        # Create drifted data - shift mean of features 0 and 1
        X_drifted = np.random.normal(0, 1, (n_samples, n_features))
        X_drifted[:, 0] = np.random.normal(3, 1, n_samples)  # Shift mean of feature 0
        X_drifted[:, 2] = np.random.normal(-2, 1.5, n_samples)  # Shift mean and variance of feature 2
        y_drifted = y_orig  # Keep labels the same for data drift test
        
        # Save drifted data
        np.savez(env['test_data_path'], X=X_drifted, y=y_drifted, feature_names=feature_names)
        
        # Test with drifted data
        drift_results, drift_detected = retrainer.check_for_data_drift()
        
        # Check detection result
        assert drift_detected, "Data drift should be detected"
        assert "drifted_features" in drift_results, "Should report which features drifted"
        assert len(drift_results["drifted_features"]) > 0, "Should identify drifted features"
        
    def test_concept_drift_detection(self, setup_temp_environment):
        """Test detection of concept drift when relationship between features and labels changes"""
        env = setup_temp_environment
        
        # Initialize retrainer with test configuration
        retrainer = ModelRetrainer(
            model_path=str(env['model_path']),
            test_data_path=str(env['test_data_path']),
            processed_data_path=str(env['processed_data_path']),
            monitoring_config=env['config']
        )
        
        # Load original test data
        orig_data = np.load(env['test_data_path'])
        X_orig = orig_data['X']
        feature_names = orig_data['feature_names']
        
        # Create concept drift by changing the relationship between features and labels
        n_samples = X_orig.shape[0]
        y_new = (X_orig[:, 2] + X_orig[:, 3] > 0).astype(int)
        
        # Save data with concept drift
        np.savez(env['test_data_path'], X=X_orig, y=y_new, feature_names=feature_names)
        
        # Test for concept drift
        drift_results, drift_detected = retrainer.check_for_data_drift()
        
        # Check detection results
        assert drift_detected, "Concept drift should be detected"
        assert drift_results.get("concept_drift_detected", False), "Concept drift flag should be True"
        
    def test_label_drift_detection(self, setup_temp_environment):
        """Test detection of label drift when label distribution changes"""
        env = setup_temp_environment
        
        # Initialize retrainer with test configuration
        retrainer = ModelRetrainer(
            model_path=str(env['model_path']),
            test_data_path=str(env['test_data_path']),
            processed_data_path=str(env['processed_data_path']),
            monitoring_config=env['config']
        )
        
        # Load original test data
        orig_data = np.load(env['test_data_path'])
        X_orig = orig_data['X']
        feature_names = orig_data['feature_names']
        
        # Create label drift by changing the distribution of labels
        n_samples = X_orig.shape[0]
        np.random.seed(42)
        y_skewed = np.random.binomial(1, 0.95, n_samples)  # 95% are class 1
        
        # Save data with label drift
        np.savez(env['test_data_path'], X=X_orig, y=y_skewed, feature_names=feature_names)
        
        # Test for label drift
        drift_results, drift_detected = retrainer.check_for_data_drift()
        
        # Check detection results
        assert drift_detected, "Label drift should be detected"
        assert drift_results.get("label_drift_detected", False), "Label drift flag should be True"
        
    def test_retraining_with_drift(self, setup_temp_environment):
        """Test full retraining workflow in presence of drift"""
        env = setup_temp_environment
        
        # Initialize retrainer with test configuration
        retrainer = ModelRetrainer(
            model_path=str(env['model_path']),
            test_data_path=str(env['test_data_path']),
            processed_data_path=str(env['processed_data_path']),
            monitoring_config=env['config']
        )
        
        # Create drifted data
        orig_data = np.load(env['test_data_path'])
        X_orig = orig_data['X']
        feature_names = orig_data['feature_names']
        n_samples = X_orig.shape[0]
        
        # Create combined drift scenario
        X_drifted = X_orig.copy()
        X_drifted[:, 0] = np.random.normal(2, 1, n_samples)  # Shift feature 0
        y_drifted = (X_drifted[:, 0] - X_drifted[:, 1] > 0).astype(int)  # Change concept
        
        # Save drifted data to test file
        np.savez(env['test_data_path'], X=X_drifted, y=y_drifted, feature_names=feature_names)
        
        # Save similar data to training file for retraining
        n_train = 1000
        X_train = np.random.normal(0, 1, (n_train, env['n_features']))
        X_train[:, 0] = np.random.normal(2, 1, n_train)  # Match shifted feature
        y_train = (X_train[:, 0] - X_train[:, 1] > 0).astype(int)  # Match new concept
        
        np.savez(env['processed_data_path'], X=X_train, y=y_train, 
                 feature_names=np.array([f"feature_{i}" for i in range(env['n_features'])]))
        
        # Set up initial metrics for comparison
        initial_metrics = {
            "f1": 0.65,
            "roc_auc": 0.68,
            "accuracy": 0.70
        }
        retrainer.metrics_history = [{
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": initial_metrics
        }]
        
        # Run retraining workflow
        result = retrainer.run_retraining_workflow()
        
        # Check results
        assert "actions_taken" in result, "Should report actions taken"
        assert "model_retraining" in result["actions_taken"], "Should have performed retraining"
        assert result["conclusion"] == "Retraining successful", "Should conclude with successful retraining"

if __name__ == "__main__":
    pytest.main(["-v", __file__])