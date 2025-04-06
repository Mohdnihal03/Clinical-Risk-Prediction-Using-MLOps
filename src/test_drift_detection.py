import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import os
import json
from pathlib import Path
import tempfile
import shutil
import joblib
import pandas as pd
from scipy.stats import ks_2samp

# Import modules to test
from monitor import DriftDetector
from retrain import ModelRetrainer

class TestDriftDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create paths
        self.model_path = os.path.join(self.test_dir, "model.joblib")
        self.train_path = os.path.join(self.test_dir, "train.npz")
        self.test_path = os.path.join(self.test_dir, "test.npz")
        self.reference_path = os.path.join(self.test_dir, "reference.npz")
        self.metrics_history_path = os.path.join(self.test_dir, "metrics_history.json")
        self.drift_history_path = os.path.join(self.test_dir, "drift_history.json")
        self.visualizations_dir = os.path.join(self.test_dir, "visualizations")
        
        # Create directories
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Create dummy data (different distributions for drift testing)
        # Reference data
        X_ref = np.random.normal(0, 1, (100, 5))
        y_ref = np.array([0] * 50 + [1] * 50)
        feature_names = np.array(['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        
        # New data (shifted distribution)
        X_new = np.random.normal(0.5, 1.2, (100, 5))  # Shifted mean and variance
        y_new = np.zeros(100)  # Changed class distribution
        
        # Create a simple dummy model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_ref, y_ref)
        joblib.dump(model, self.model_path)
        
        # Save data files
        np.savez(self.reference_path, X=X_ref, y=y_ref, feature_names=feature_names)
        np.savez(self.test_path, X=X_new, y=y_new, feature_names=feature_names)
        np.savez(self.train_path, X=X_ref, y=y_ref, feature_names=feature_names)
        
        # Create monitoring config
        self.monitoring_config = {
            'reference_data_path': self.reference_path,
            'model_path': self.model_path,
            'monitor_dir': self.test_dir,
            'history_file': self.drift_history_path,
            'visualizations_dir': self.visualizations_dir,
            'drift_thresholds': {
                'data_drift_threshold': 0.1,
                'concept_drift_threshold': 0.05,
                'label_drift_threshold': 0.1,
                'evidently_drift_threshold': 0.2,
                'significant_drift_ratio': 0.3
            }
        }
        
        # Initialize metrics history
        self.metrics_history = [
            {
                "timestamp": "2023-01-01T00:00:00",
                "metrics": {"f1": 0.85, "roc_auc": 0.88, "accuracy": 0.86, "precision": 0.84, "recall": 0.87}
            },
            {
                "timestamp": "2023-01-02T00:00:00",
                "metrics": {"f1": 0.84, "roc_auc": 0.87, "accuracy": 0.85, "precision": 0.83, "recall": 0.86}
            }
        ]
        
        with open(self.metrics_history_path, 'w') as f:
            json.dump(self.metrics_history, f)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_data_drift_detection(self):
        """Test if data drift detection works correctly"""
        detector = DriftDetector(self.monitoring_config)
        
        # Run drift detection
        results, retraining_recommended = detector.analyze_drift(self.test_path)
        
        # Check results structure
        self.assertIn('data_drift', results)
        self.assertIn('concept_drift', results)
        self.assertIn('label_drift', results)
        self.assertIn('drift_detected', results)
        self.assertIn('retraining_recommended', results)
        
        # Verify drift is detected (should be, as we created different distributions)
        self.assertTrue(results['data_drift']['detected'], 
                       "Data drift should be detected with our test data")
        
        # Check KS test results
        self.assertIn('ks_test', results['data_drift']['methods'])
        self.assertIn('detected', results['data_drift']['methods']['ks_test'])
        
        # Check evidently results
        self.assertIn('evidently', results['data_drift']['methods'])
    
    def test_concept_drift_detection(self):
        """Test concept drift detection with a model that performs differently on new data"""
        detector = DriftDetector(self.monitoring_config)
        
        # Access the private method using name mangling
        X_new = np.load(self.test_path)['X']
        y_new = np.load(self.test_path)['y']
        
        concept_results, drift_detected = detector._detect_concept_drift(X_new, y_new)
        
        # Check results structure
        self.assertIn('reference_metrics', concept_results)
        self.assertIn('new_metrics', concept_results)
        self.assertIn('differences', concept_results)
        
        # Check all metrics exist
        for metric_set in ['reference_metrics', 'new_metrics']:
            self.assertIn('accuracy', concept_results[metric_set])
            self.assertIn('precision', concept_results[metric_set])
            self.assertIn('recall', concept_results[metric_set])
            self.assertIn('f1', concept_results[metric_set])
    
    def test_label_drift_detection(self):
        """Test label drift detection with different distributions"""
        detector = DriftDetector(self.monitoring_config)
        
        # Create perfectly balanced reference data (50/50 split)
        self.y_ref = np.array([0] * 50 + [1] * 50)
        
        # Create extreme drift case (all zeros)
        y_new_biased = np.zeros(100)
        
        label_results, drift_detected = detector._detect_label_drift(y_new_biased)
        
        # Check results
        self.assertTrue(drift_detected, "Label drift should be detected with extreme distribution")
        self.assertIn('reference_distribution', label_results)
        self.assertIn('new_distribution', label_results)
        self.assertIn('distribution_diff', label_results)
        
        # With perfect 50/50 reference and all zeros new data, difference should be exactly 0.5
        self.assertEqual(label_results['distribution_diff'], 0.5)
        
    def test_performance_drift_detection(self):
        """Test performance drift detection in ModelRetrainer"""
        # Initialize retrainer with our test config
        retrainer = ModelRetrainer(
            model_path=self.model_path,
            test_data_path=self.test_path,
            processed_data_path=self.train_path,
            metrics_history_path=self.metrics_history_path,
            performance_threshold=0.7,  # Lower threshold to test significant decrease
            monitoring_config=self.monitoring_config
        )
        
        # Test 1: No drift with good metrics
        retrainer.metrics_history = [
            {"timestamp": "2023-01-01", "metrics": {"f1": 0.82, "roc_auc": 0.85}},
            {"timestamp": "2023-01-02", "metrics": {"f1": 0.83, "roc_auc": 0.86}}
        ]
        drift_detected, details = retrainer.detect_performance_drift()
        self.assertFalse(drift_detected, "No drift should be detected with good metrics")
        
        # Test 2: Drift with metrics below threshold
        retrainer.metrics_history.append(
            {"timestamp": "2023-01-03", "metrics": {"f1": 0.65, "roc_auc": 0.67}}
        )
        drift_detected, details = retrainer.detect_performance_drift()
        self.assertTrue(drift_detected, "Drift should be detected with metrics below threshold")
        self.assertEqual(details.get("reason"), "below_threshold")
        
        # Test 3: Drift with significant decrease (but above threshold)
        retrainer.metrics_history = [
            {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
            {"timestamp": "2023-01-02", "metrics": {"f1": 0.78, "roc_auc": 0.81}}  # 0.07 decrease
        ]
        drift_detected, details = retrainer.detect_performance_drift()
        self.assertTrue(drift_detected, "Drift should be detected with significant decrease")
        self.assertEqual(details.get("reason"), "significant_decrease")
    
    @patch('retrain.train_model')
    @patch('retrain.evaluate_model')
    def test_retraining_workflow(self, mock_evaluate, mock_train):
        """Test the complete retraining workflow with mocked training functions"""
        # Setup mocks with lower performance
        mock_train.return_value = (self.model_path, self.test_path)
        mock_evaluate.return_value = {"f1": 0.75, "roc_auc": 0.77, "accuracy": 0.76, "precision": 0.74, "recall": 0.76}
        
        # Initialize retrainer with higher threshold
        retrainer = ModelRetrainer(
            model_path=self.model_path,
            test_data_path=self.test_path,
            processed_data_path=self.train_path,
            metrics_history_path=self.metrics_history_path,
            performance_threshold=0.8,
            monitoring_config=self.monitoring_config
        )
        
        # Set history with decreasing performance
        retrainer.metrics_history = [
            {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
            {"timestamp": "2023-01-02", "metrics": {"f1": 0.82, "roc_auc": 0.85}},
            {"timestamp": "2023-01-03", "metrics": {"f1": 0.75, "roc_auc": 0.77}}
        ]
        os.environ["GEMINI_API_KEY"] = "fake_key"
        result = retrainer.run_retraining_workflow(force_retrain=False)
        os.environ.pop("GEMINI_API_KEY", None)
        self.assertIn("performance_drift_detection", result["actions_taken"])
        self.assertIn("model_retraining", result["actions_taken"])
        self.assertEqual("Retraining successful", result["conclusion"])
        self.assertTrue(result["performance_drift"]["detected"])
        
    @patch('retrain.genai.GenerativeModel')
    def test_gemini_analysis(self, mock_genai_model):
        """Test Gemini analysis with mocked API"""
        # Setup mock response
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "analysis": "Model performance has declined significantly, suggesting drift.",
            "retrain_recommended": True,
            "confidence": 0.85,
            "improvements": ["Collect more recent data", "Review feature importance"],
            "data_checks": ["Check for missing values", "Verify label distribution"]
        })
        mock_instance.generate_content.return_value = mock_response
        mock_genai_model.return_value = mock_instance
        
        # Set environment variable for test
        os.environ["GEMINI_API_KEY"] = "fake_key_for_testing"
        
        # Initialize retrainer
        retrainer = ModelRetrainer(
            model_path=self.model_path,
            test_data_path=self.test_path,
            processed_data_path=self.train_path,
            metrics_history_path=self.metrics_history_path,
            monitoring_config=self.monitoring_config
        )
        
        # Test LLM analysis
        metrics_trend = [
            {"f1": 0.85, "roc_auc": 0.88},
            {"f1": 0.82, "roc_auc": 0.85},
            {"f1": 0.78, "roc_auc": 0.80}
        ]
        drift_details = {
            "current_f1": 0.78,
            "current_auc": 0.80,
            "threshold": 0.8,
            "f1_change": -0.04,
            "auc_change": -0.05
        }
        
        result = retrainer._analyze_with_gemini(metrics_trend, drift_details)
        
        # Assert Gemini was called with the expected parameters
        mock_instance.generate_content.assert_called_once()
        self.assertIn("retrain_recommended", result)
        self.assertTrue(result["retrain_recommended"])
        
        # Clean up
        os.environ.pop("GEMINI_API_KEY", None)
    
    def test_initial_training(self):
        """Test initial training scenario"""
        with patch('retrain.train_model') as mock_train:
            # Setup mock
            mock_train.return_value = (self.model_path, self.test_path)
            
            # Delete model to simulate first-time training
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            
            # Initialize retrainer
            retrainer = ModelRetrainer(
                model_path=os.path.join(self.test_dir, "new_model.joblib"),  # Nonexistent path
                test_data_path=self.test_path,
                processed_data_path=self.train_path,
                metrics_history_path=self.metrics_history_path,
                monitoring_config=self.monitoring_config
            )
            
            # Run workflow
            result = retrainer.run_retraining_workflow()
            
            # Assert initial training was performed
            self.assertIn("initial_training", result["actions_taken"])
            mock_train.assert_called_once()

if __name__ == "__main__":
    unittest.main()