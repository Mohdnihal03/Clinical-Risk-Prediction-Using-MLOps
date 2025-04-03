# # src/evaluate.py
# import logging
# import mlflow
# import numpy as np
# import joblib
# from pathlib import Path
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, 
#     f1_score, roc_auc_score, confusion_matrix
# )
# import matplotlib.pyplot as plt
# import shap
# import os

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Improved MLflow initialization for Windows
# try:
#     # Convert Windows path to proper file URI
#     mlruns_path = Path("mlruns").absolute()
#     tracking_uri = f"file:///{mlruns_path.as_posix()}"
    
#     # Ensure directory exists
#     os.makedirs(mlruns_path, exist_ok=True)
    
#     mlflow.set_tracking_uri(tracking_uri)
    
#     # Create experiment if it doesn't exist
#     if not mlflow.get_experiment_by_name("sepsis_prediction"):
#         mlflow.create_experiment("sepsis_prediction", artifact_location=tracking_uri)
#     mlflow.set_experiment("sepsis_prediction")
    
#     logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
#     logger.info(f"MLflow experiment: {mlflow.get_experiment_by_name('sepsis_prediction')}")
# except Exception as e:
#     logger.error(f"MLflow initialization failed: {str(e)}")
#     raise

# class ModelEvaluator:
#     def __init__(self, model_path: str, test_data_path: str):
#         self.model_path = Path(model_path)
#         self.test_data_path = Path(test_data_path)
#         self.model = None
#         self.X_test = None
#         self.y_test = None

#     def load_artifacts(self):
#         """Load model and test data"""
#         try:
#             self.model = joblib.load(self.model_path)
#             data = np.load(self.test_data_path)
#             self.X_test = data['X']
#             self.y_test = data['y']
#             logger.info(f"Loaded test data with shape {self.X_test.shape}")
#         except Exception as e:
#             logger.error(f"Failed to load artifacts: {str(e)}")
#             raise

#     def evaluate(self):
#         """Run full evaluation workflow"""
#         logger.info("Starting MLflow evaluation run...")
#         try:
#             with mlflow.start_run(run_name="sepsis_evaluation"):
#                 # Log parameters
#                 mlflow.log_params({
#                     "model_type": "XGBoost",
#                     "n_estimators": getattr(self.model, 'n_estimators', 'unknown'),
#                     "model_path": str(self.model_path),
#                     "test_data_path": str(self.test_data_path)
#                 })
                
#                 # Make predictions
#                 y_pred = self.model.predict(self.X_test)
#                 y_proba = self.model.predict_proba(self.X_test)[:, 1]
                
#                 # Calculate metrics
#                 metrics = {
#                     "accuracy": accuracy_score(self.y_test, y_pred),
#                     "precision": precision_score(self.y_test, y_pred),
#                     "recall": recall_score(self.y_test, y_pred),
#                     "f1": f1_score(self.y_test, y_pred),
#                     "roc_auc": roc_auc_score(self.y_test, y_proba)
#                 }
#                 mlflow.log_metrics(metrics)
                
#                 # Log confusion matrix
#                 fig, ax = plt.subplots()
#                 conf_matrix = confusion_matrix(self.y_test, y_pred)
#                 ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
#                 for i in range(conf_matrix.shape[0]):
#                     for j in range(conf_matrix.shape[1]):
#                         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
#                 plt.title("Confusion Matrix")
#                 mlflow.log_figure(fig, "confusion_matrix.png")
#                 plt.close()
                
#                 # Log SHAP plot
#                 try:
#                     explainer = shap.TreeExplainer(self.model)
#                     shap_values = explainer.shap_values(self.X_test)
#                     plt.figure()
#                     shap.summary_plot(shap_values, self.X_test, show=False)
#                     plt.tight_layout()
#                     mlflow.log_figure(plt.gcf(), "shap_summary.png")
#                     plt.close()
#                 except Exception as e:
#                     logger.warning(f"SHAP visualization failed: {str(e)}")
                
#                 logger.info(f"Evaluation metrics: {metrics}")
#                 return metrics
                
#         except Exception as e:
#             logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
#             raise

# def evaluate_model(model_path: str, test_data_path: str):
#     evaluator = ModelEvaluator(model_path, test_data_path)
#     evaluator.load_artifacts()
#     return evaluator.evaluate()

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

# MLflow initialization
try:
    mlruns_path = Path("mlruns").absolute()
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    os.makedirs(mlruns_path, exist_ok=True)
    
    mlflow.set_tracking_uri(tracking_uri)
    if not mlflow.get_experiment_by_name("sepsis_prediction"):
        mlflow.create_experiment("sepsis_prediction", artifact_location=tracking_uri)
    mlflow.set_experiment("sepsis_prediction")
    
    logger.info(f"MLflow initialized at {mlflow.get_tracking_uri()}")
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
        self.feature_names = None  # Will store feature names for interpretation

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
                
            logger.info(f"Loaded test data with {self.X_test.shape[0]} samples and {self.X_test.shape[1]} features")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            raise

    def _create_shap_plots(self):
        """Generate user-friendly SHAP visualizations"""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            # 1. Summary Plot with Feature Names
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
            plt.title("Feature Importance and Impact on Predictions", fontsize=12)
            plt.xlabel("Impact on Model Output", fontsize=10)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            plt.close()
            
            # 2. Bar Plot of Mean Absolute SHAP Values (Simpler to understand)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                            plot_type="bar", show=False)
            plt.title("Most Important Features for Sepsis Prediction", fontsize=12)
            plt.xlabel("Average Impact on Prediction", fontsize=10)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_feature_importance.png")
            plt.close()
            
            # 3. Decision Plot for Sample Explanations (Easier to interpret)
            sample_idx = np.random.randint(0, len(self.X_test), size=5)  # Show 5 random samples
            plt.figure(figsize=(12, 8))
            shap.decision_plot(explainer.expected_value, shap_values[sample_idx], 
                             feature_names=self.feature_names)
            plt.title("How Features Affect Individual Predictions", fontsize=12)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_decision_examples.png")
            plt.close()
            
            # Log SHAP values as CSV for further analysis
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            shap_csv_path = "shap_values.csv"
            shap_df.to_csv(shap_csv_path, index=False)
            mlflow.log_artifact(shap_csv_path)
            
        except Exception as e:
            logger.warning(f"SHAP visualization failed: {str(e)}")
            raise

    def evaluate(self):
        """Run full evaluation workflow with improved explanations"""
        try:
            with mlflow.start_run(run_name="sepsis_evaluation"):
                # Log basic model info
                mlflow.log_params({
                    "model_type": "XGBoost",
                    "n_estimators": getattr(self.model, 'n_estimators', 'unknown')
                })
                
                # Make predictions
                y_pred = self.model.predict(self.X_test)
                y_proba = self.model.predict_proba(self.X_test)[:, 1]
                
                # Calculate and log metrics
                metrics = {
                    "accuracy": round(accuracy_score(self.y_test, y_pred), 3),
                    "precision": round(precision_score(self.y_test, y_pred), 3),
                    "recall": round(recall_score(self.y_test, y_pred), 3),
                    "f1": round(f1_score(self.y_test, y_pred), 3),
                    "roc_auc": round(roc_auc_score(self.y_test, y_proba), 3)
                }
                mlflow.log_metrics(metrics)
                
                # Log confusion matrix with percentages
                fig, ax = plt.subplots(figsize=(8, 6))
                conf_matrix = confusion_matrix(self.y_test, y_pred)
                conf_matrix_perc = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
                
                ax.matshow(conf_matrix_perc, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i, 
                               s=f"{conf_matrix[i, j]}\n({conf_matrix_perc[i, j]:.1%})", 
                               va='center', ha='center')
                plt.title("Confusion Matrix (Counts and Percentages)")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                mlflow.log_figure(fig, "confusion_matrix.png")
                plt.close()
                
                # Generate improved SHAP explanations
                self._create_shap_plots()
                
                # Add text explanation of metrics
                metrics_explanation = (
                    "Model Performance Explanation:\n"
                    f"- Accuracy ({metrics['accuracy']}): Percentage of correct predictions\n"
                    f"- Precision ({metrics['precision']}): When predicting sepsis, how often we're correct\n"
                    f"- Recall ({metrics['recall']}): What percentage of actual sepsis cases we detected\n"
                    f"- F1 Score ({metrics['f1']}): Balance between precision and recall\n"
                    f"- ROC AUC ({metrics['roc_auc']}): Model's ability to distinguish classes (1=perfect)"
                )
                mlflow.log_text(metrics_explanation, "performance_explanation.txt")
                
                logger.info("Evaluation completed successfully")
                return metrics
                
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

def evaluate_model(model_path: str, test_data_path: str):
    evaluator = ModelEvaluator(model_path, test_data_path)
    evaluator.load_artifacts()
    return evaluator.evaluate()
