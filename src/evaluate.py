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
        """Generate SHAP visualizations"""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            # Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_names, show=False)
            plt.title("Feature Importance", fontsize=12)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            plt.close()
            
            # Feature Importance Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_names,
                            plot_type="bar", show=False)
            plt.title("Feature Importance Scores", fontsize=12)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_feature_importance.png")
            plt.close()
            
            # Log SHAP values
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            shap_csv_path = "shap_values.csv"
            shap_df.to_csv(shap_csv_path, index=False)
            mlflow.log_artifact(shap_csv_path)
            
        except Exception as e:
            logger.warning(f"SHAP visualization failed: {str(e)}")
            raise

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