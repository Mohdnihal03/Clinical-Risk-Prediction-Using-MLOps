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
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_processed_data(self, processed_path: str):
        """Load data from preprocessing output"""
        try:
            if processed_path.endswith('.npz'):
                data = np.load(processed_path)
                X = data['data']
                y = data['target']
            else:  # CSV fallback
                df = pd.read_csv(processed_path)
                X = df.drop(columns=['Sepsis_Label'])
                y = df['Sepsis_Label']
            return X, y
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def train_model(self, processed_path: str) -> str:
        """Complete training workflow"""
        try:
            # 1. Load processed data
            logger.info("Loading processed data...")
            X, y = self.load_processed_data(processed_path)
            
            # 2. Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 3. Train XGBoost model (with your original parameters)
            logger.info("Training XGBoost model...")
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_estimators=100,
                max_depth=5
            )
            model.fit(X_train, y_train)
            
            # 4. Evaluate
            y_pred = model.predict(X_test)
            logger.info("\n" + classification_report(y_test, y_pred))
            
            # 5. Save model
            model_path = r"C:\Users\nihall\Desktop\mlops-clinical-risk-prediction\model\sepsis_xgboost_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

def train_model(processed_path: str) -> str:
    """Wrapper function for training"""
    trainer = ModelTrainer()
    return trainer.train_model(processed_path)