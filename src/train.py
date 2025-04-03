# # src/train.py
# import logging
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelTrainer:
#     def __init__(self, model_dir="model"):
#         self.model_dir = Path(model_dir)
#         self.model_dir.mkdir(exist_ok=True, parents=True)
        
#     def load_processed_data(self, processed_path: str):
#         """Load data from preprocessing output"""
#         try:
#             if processed_path.endswith('.npz'):
#                 data = np.load(processed_path)
#                 X = data['X']
#                 y = data['y']
#             else:  # Handle CSV output structure
#                 data_dir = Path(processed_path).parent
#                 X = pd.read_csv(data_dir / "X_transformed.csv")
#                 y = pd.read_csv(data_dir / "y_transformed.csv").squeeze()
#             return X, y
#         except Exception as e:
#             raise RuntimeError(f"Data loading failed: {str(e)}")

#     def train_model(self, processed_path: str) -> str:
#         """Complete training workflow"""
#         try:
#             # 1. Load processed data
#             logger.info("Loading processed data...")
#             X, y = self.load_processed_data(processed_path)
            
#             # 2. Train-test split
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, 
#                 test_size=0.2, 
#                 random_state=42,
#                 stratify=y  # Maintain class distribution
#             )
            
#             # 3. Train XGBoost model
#             logger.info("Training XGBoost model...")
#             model = XGBClassifier(
#                 use_label_encoder=False,
#                 eval_metric='logloss',
#                 n_estimators=100,
#                 max_depth=5,
#                 scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle class imbalance
#             )
#             model.fit(X_train, y_train)
            
#             # 4. Evaluate
#             y_pred = model.predict(X_test)
#             logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            
#             # 5. Save model
#             model_path = self.model_dir / "sepsis_xgboost_model.joblib"
#             joblib.dump(model, model_path)
#             logger.info(f"Model saved to {model_path}")
            
#             return str(model_path)
            
#         except Exception as e:
#             logger.error(f"Training error: {str(e)}", exc_info=True)
#             raise RuntimeError(f"Training failed: {str(e)}")

# def train_model(processed_path: str) -> str:
#     """Wrapper function for training"""
#     trainer = ModelTrainer()
#     return trainer.train_model(processed_path)

# src/train.py
# src/train.py
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
        self.test_data_path.parent.mkdir(exist_ok=True, parents=True)
        
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