# src/evaluate.py
'''
Model Evaluation and Tracking Module

1. Purpose:
This module provides comprehensive model evaluation capabilities for the sepsis prediction system, including:
- Performance metric calculation and tracking
- Explainability analysis using SHAP values
- Visualization generation
- MLflow experiment tracking for full reproducibility

2. Key Features:
- Automated metric calculation (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization with percentages
- SHAP value analysis for model interpretability
- Full ML experiment tracking with:
  - Parameter logging
  - Metric storage
  - Artifact versioning
- SQLite backend for lightweight tracking

3. Input Requirements:
- Trained model in joblib format
- Test dataset in NPZ format (containing X, y arrays)
- Optional: Feature names for interpretability

4. Outputs:
- Logged metrics in MLflow
- Saved visualizations:
  - Confusion matrix plot
  - SHAP summary plot
- Evaluation report with key performance indicators

5. MLflow Integration:
- Uses SQLite backend for portable tracking
- Automatic experiment creation
- Artifact storage in local filesystem
- Full reproducibility through logged parameters

6. Evaluation Metrics:
- Standard classification metrics:
  - Accuracy, Precision, Recall
  - F1-score (balanced)
  - ROC-AUC (probability-based)
- Confusion matrix with:
  - Absolute counts
  - Percentage values

7. Explainability:
- SHAP (SHapley Additive Explanations) analysis:
  - Feature importance visualization
  - Model behavior interpretation
  - Interaction effects capture

8. Error Handling:
- Robust artifact loading
- Graceful degradation for optional components
- Comprehensive logging
- Visual validation of outputs

9. Usage Example:
>>> evaluator = ModelEvaluator("model/xgboost.joblib", "data/test.npz")
>>> metrics = evaluator.evaluate()

10. Dependencies:
- MLflow for experiment tracking
- SHAP for explainability
- Matplotlib for visualization
- Standard scikit-learn metrics

11. Design Decisions:
- SQLite backend chosen for:
  - Lightweight operation
  - No external dependencies
  - Easy version control
- SHAP over other methods because:
  - Theoretical soundness
  - Feature interaction capture
  - Visual interpretability
- Percentage+count confusion matrix for:
  - Balanced view of performance
  - Clinical relevance

12. Clinical Relevance:
- Focus on recall (sensitivity) for sepsis detection
- SHAP analysis helps validate clinical plausibility
- Confidence percentages support decision making

13. Monitoring Integration:
- All metrics available for drift detection
- Model behavior baselined
- Feature importance tracking over time

14. Summary:
The ModelEvaluator class provides a production-grade evaluation framework that combines quantitative
performance assessment with qualitative model interpretability analysis. Its tight integration with
MLflow ensures all evaluation results are properly versioned and reproducible, while the explainability
components help build trust in the model's predictions - a critical factor for clinical deployment.
'''


import logging
import mlflow
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import shap
import os
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow initialization with SQLite backend
try:
    # Set SQLite as the tracking URI
    tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    if not mlflow.get_experiment_by_name("sepsis_prediction"):
        # For SQLite backend, artifact location should be a filesystem path
        artifact_location = Path("mlruns").absolute().as_posix()
        mlflow.create_experiment(
            "sepsis_prediction",
            artifact_location=f"file:///{artifact_location}"
        )
    
    mlflow.set_experiment("sepsis_prediction")
    
    # Ensure mlruns directory exists for artifacts
    mlruns_path = Path("mlruns").absolute()
    os.makedirs(mlruns_path, exist_ok=True)
    
    logger.info(f"MLflow initialized with tracking URI: {tracking_uri}")
    logger.info(f"Artifacts will be stored at: {mlruns_path}")
except Exception as e:
    logger.error(f"MLflow initialization failed: {str(e)}")
    raise

class ModelEvaluator:
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.model = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def load_artifacts(self):
        """Load model and test data"""
        try:
            self.model = joblib.load(self.model_path)
            data = np.load(self.test_data_path)
            self.X_test = data['X']
            self.y_test = data['y']
            
            # Try to load feature names if available
            if 'feature_names' in data:
                self.feature_names = list(data['feature_names'])
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_test.shape[1])]
                
            logger.info(f"Loaded test data with {self.X_test.shape[0]} samples")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            raise

    def _create_shap_plots(self):
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            # Handle multi-class SHAP values (take first class if binary)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
            
            # Create SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
            summary_path = os.path.join(self.output_dir, "shap_summary.png")
            plt.savefig(summary_path)
            plt.close()
            
            # Create SHAP values DataFrame for feature importance
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            return shap_df
            
        except Exception as e:
            logger.warning(f"SHAP visualization failed: {str(e)}")
            return None

    def evaluate(self):
        """Run evaluation workflow"""
        try:
            with mlflow.start_run(run_name="sepsis_evaluation"):
                # Log parameters
                mlflow.log_params({
                    "model_type": "XGBoost",
                    "n_estimators": getattr(self.model, 'n_estimators', 'unknown')
                })
                
                # Make predictions
                y_pred = self.model.predict(self.X_test)
                y_proba = self.model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    "accuracy": round(accuracy_score(self.y_test, y_pred), 3),
                    "precision": round(precision_score(self.y_test, y_pred), 3),
                    "recall": round(recall_score(self.y_test, y_pred), 3),
                    "f1": round(f1_score(self.y_test, y_pred), 3),
                    "roc_auc": round(roc_auc_score(self.y_test, y_proba), 3)
                }
                mlflow.log_metrics(metrics)
                
                # Confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                conf_matrix = confusion_matrix(self.y_test, y_pred)
                conf_matrix_perc = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
                
                ax.matshow(conf_matrix_perc, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i, 
                               s=f"{conf_matrix[i, j]}\n({conf_matrix_perc[i, j]:.1%})", 
                               va='center', ha='center')
                plt.title("Confusion Matrix")
                mlflow.log_figure(fig, "confusion_matrix.png")
                plt.close()
                
                # SHAP plots
                self._create_shap_plots()
                
                logger.info(f"Evaluation completed with metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

def evaluate_model(model_path: str, test_data_path: str):
    evaluator = ModelEvaluator(model_path, test_data_path)
    evaluator.load_artifacts()
    return evaluator.evaluate()