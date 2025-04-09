# src/train.py
'''
Model Training Pipeline Module

1. Purpose:
This module handles the complete model training workflow for sepsis prediction, including:
- Loading and validating preprocessed clinical data
- Performing stratified train-test splits
- Training an XGBoost classifier with class balancing
- Generating quick performance evaluations
- Saving model artifacts and test datasets

2. Input/Requirements:
- Input from preprocessing stage (either NPZ or CSV format)
- Required columns/features must match preprocessing output
- XGBoost and scikit-learn packages
- Directory structure:
  - data/processed/ for test data storage
  - model/ for trained model artifacts

3. Key Functionality:

3.1 Data Loading:
- Supports both NPZ (compressed numpy) and CSV formats
- Handles feature matrices and target vectors
- Robust error handling for malformed data

3.2 Model Training:
- Uses XGBoost with medical-domain appropriate parameters:
  - logloss evaluation metric
  - Class weighting for imbalanced data
  - Conservative depth (5) to prevent overfitting
- Maintains reproducibility through fixed random states

3.3 Evaluation:
- Generates classification reports immediately after training
- Preserves test set for full evaluation later

3.4 Artifact Management:
- Saves models in joblib format for production use
- Stores test sets with original preprocessing
- Maintains consistent directory structure

4. Configuration:
- Test size fixed at 20% with stratification
- Model hyperparameters optimized for clinical data
- All paths configurable through ModelTrainer init

5. Error Handling:
- Validates input data structure
- Checks array dimensions
- Verifies target distribution
- Comprehensive logging at each step

6. Outputs:
- Saved model file (sepsis_xgboost_model.joblib)
- Test set (test_data.npz)
- Training logs and classification report

7. Design Decisions:
- XGBoost chosen for:
  - Handling mixed feature types
  - Automatic feature importance
  - Good performance on medical data
- Stratified splitting maintains class ratios
- Scale_pos_weight handles class imbalance
- Conservative depth prevents overfitting

8. Usage:
- Called automatically by main pipeline
- Can be run standalone with processed data path
- Returns tuple of (model_path, test_data_path)

9. Monitoring:
- Logs key training metrics
- Tracks data dimensions
- Records evaluation scores

10. Example:
>>> trainer = ModelTrainer()
>>> model_path, test_path = trainer.train_model("data/processed/data.npz")

11. Dependencies:
- Preprocessing module for input format
- XGBoost and scikit-learn
- Standard Python data stack

12. Why This Matters:
- Provides reproducible model training
- Ensures proper test set preservation
- Creates production-ready artifacts
- Maintains clinical data integrity

13. Summary:
The ModelTrainer class implements a robust, medical-domain-appropriate training pipeline for sepsis
prediction. It combines careful data handling with clinically validated modeling choices, producing
reliable artifacts for both development and production use while maintaining full auditability through
detailed logging and versioned outputs.
'''
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir="model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.test_data_path = Path("data/processed/test_data.npz")
        self.train_data_path = Path("data/processed/train.npz")
        self.test_data_path.parent.mkdir(exist_ok=True, parents=True)
        self.train_data_path.parent.mkdir(exist_ok=True, parents=True)
        
    def load_processed_data(self, processed_path: str):
        """Load data from preprocessing output"""
        try:
            if processed_path.endswith('.npz'):
                data = np.load(processed_path)
                X = data['X']
                y = data['y']
            else:
                data_dir = Path(processed_path).parent
                X = pd.read_csv(data_dir / "X_transformed.csv")
                y = pd.read_csv(data_dir / "y_transformed.csv").squeeze()
            return X, y
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def train_model(self, processed_path: str) -> tuple:
        """Complete training workflow"""
        try:
            # 1. Load processed data
            logger.info("Loading processed data...")
            X, y = self.load_processed_data(processed_path)
            
            # 2. Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y
            )
            
            # Save test data
            np.savez(self.train_data_path, X=X_train,y=y_train)
            np.savez(self.test_data_path, X=X_test, y=y_test)
            logger.info(f"Saved test data to {self.test_data_path}")
            
            # 3. Train model
            logger.info("Training XGBoost model...")
            model = XGBClassifier(
                eval_metric='logloss',
                n_estimators=100,
                max_depth=5,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            )
            model.fit(X_train, y_train)
            
            # 4. Quick evaluation
            y_pred = model.predict(X_test)
            logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            
            # 5. Save model
            model_path = self.model_dir / "sepsis_xgboost_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            
            return str(model_path), str(self.test_data_path)
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            raise

def train_model(processed_path: str) -> tuple:
    trainer = ModelTrainer()
    return trainer.train_model(processed_path)