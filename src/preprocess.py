import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalPreprocessor:
    def __init__(self):
        self.num_features = ['Age', 'Heart_Rate', 'BP_Systolic', 'BP_Diastolic',
                           'Temperature', 'Respiratory_Rate', 'WBC_Count', 'Lactate_Level']
        self.cat_features = ['Gender', 'Comorbidities']
        self.text_features = ['Clinical_Notes']

    def preprocess(self, input_path: str):
        """Full preprocessing pipeline"""
        try:
            # Load and validate data
            df = pd.read_csv(input_path)
            
            # Validate all required columns exist
            required_columns = self.num_features + self.cat_features + self.text_features + ['Sepsis_Label']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure text data is properly formatted
            df['Clinical_Notes'] = df['Clinical_Notes'].astype(str).fillna('')
            
            n_samples = len(df)
            logger.info(f"Loaded {n_samples} samples")
            
            # Process numerical features
            logger.info("Processing numerical features...")
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Process categorical features
            logger.info("Processing categorical features...")
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Process text features - convert to list of strings first
            logger.info("Processing text features...")
            text_data = df['Clinical_Notes'].tolist()
            text_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=128,
                    stop_words='english',
                    ngram_range=(1, 2)
                ))
            ])
            
            # Create the full pipeline
            transformers = [
                ('num', num_pipeline, self.num_features),
                ('cat', cat_pipeline, self.cat_features),
                ('text', text_pipeline, 'Clinical_Notes')
            ]
            
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                verbose_feature_names_out=False
            )
            
            # Transform the data
            logger.info("Transforming all features...")
            X = preprocessor.fit_transform(df)
            logger.info(f"Final combined shape: {X.shape}")
            
            # Save results
            output_path = Path("data/processed") / f"processed_{Path(input_path).stem}.npz"
            df.to_csv(r'data\processed\preprocessed.csv')
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
                logger.info(f"Generated {len(feature_names)} feature names")
            except Exception as e:
                logger.warning(f"Could not get feature names: {str(e)}")
                # Create generic feature names as fallback
                feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
                logger.info("Using generic feature names instead")
            
            # Save processed data as numpy arrays
            np.savez(
                output_path,
                X=X,
                y=df['Sepsis_Label'].values,
                feature_names=feature_names,
                allow_pickle=True
            )
            
            # Also save preprocessed dataframe for reference
            df_path = Path("data/processed") / "preprocessed.csv"
            df.to_csv(df_path, index=False)
            
            logger.info(f"Saved processed data to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Preprocessing failed: {str(e)}")

def preprocess_data(input_path: str) -> str:
    """Wrapper function for preprocessing"""
    return ClinicalPreprocessor().preprocess(input_path)