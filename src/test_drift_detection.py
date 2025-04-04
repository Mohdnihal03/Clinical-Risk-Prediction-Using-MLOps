# Create a test script test_drift_detection.py
from retrain import ModelRetrainer
import numpy as np
import json
from pathlib import Path

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
    
    # Test 2: Add bad metrics
    retrainer.metrics_history.append({"timestamp": "2023-01-02", "metrics": bad_metrics})
    drift_detected, details = retrainer.detect_performance_drift()
    print(f"Test 2 (below threshold): Drift detected? {drift_detected} (Should be True)")
    print("Details:", json.dumps(details, indent=2))
    
    # Test 3: Significant decrease
    retrainer.metrics_history = [
        {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
        {"timestamp": "2023-01-02", "metrics": {"f1": 0.75, "roc_auc": 0.77}}
    ]
    drift_detected, details = retrainer.detect_performance_drift()
    print(f"Test 3 (significant drop): Drift detected? {drift_detected} (Should be True)")
    print("Details:", json.dumps(details, indent=2))

def test_retraining_workflow():
    """Test the complete retraining workflow with simulated drift"""
    from pathlib import Path  # Add this import at the top
    from retrain import ModelRetrainer
    import tempfile
    import shutil
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup test paths
        model_path = tmpdir / "model.joblib"
        test_path = tmpdir / "test.npz"
        train_path = tmpdir / "train.npz"
        
        # Create dummy data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        np.savez(train_path, X=X, y=y)
        np.savez(test_path, X=X[:20], y=y[:20])
        
        # Initialize with force retrain to create initial model
        retrainer = ModelRetrainer(
            model_path=model_path,
            test_data_path=test_path,
            processed_data_path=train_path,
            performance_threshold=0.8
        )
        retrainer.run_retraining_workflow(force_retrain=True)
        
        # Now test with simulated drift
        retrainer.metrics_history = [
            {"timestamp": "2023-01-01", "metrics": {"f1": 0.85, "roc_auc": 0.88}},
            {"timestamp": "2023-01-02", "metrics": {"f1": 0.75, "roc_auc": 0.77}}  # Below threshold
        ]
        
        print("\n=== Testing Performance Drift Scenario ===")
        result = retrainer.run_retraining_workflow()
        print("Actions:", result["actions_taken"])
        print("Conclusion:", result["conclusion"])
        
        # Verify retraining occurred
        assert "model_retraining" in result["actions_taken"], "Retraining should have occurred"
        assert result["conclusion"] == "Retraining successful", "Retraining should have succeeded"
        
        print("Test passed successfully")
if __name__ == "__main__":
    test_performance_drift()
    test_retraining_workflow()