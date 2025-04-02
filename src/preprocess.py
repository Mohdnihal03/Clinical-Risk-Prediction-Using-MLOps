import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalPreprocessor:
    def __init__(self):
        self.num_features = ['Age', 'Heart_Rate', 'BP_Systolic', 'BP_Diastolic',
                           'Temperature', 'Respiratory_Rate', 'WBC_Count', 'Lactate_Level']
        self.cat_features = ['Gender', 'Comorbidities']
        self.text_features = ['Clinical_Notes']
        self.target = 'Sepsis_Label'

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform initial data cleaning"""
        logger.info("Performing initial data cleaning...")
        
        # Convert column names to string
        df.columns = df.columns.astype(str)
        
        # Ensure proper data types
        df[self.num_features] = df[self.num_features].apply(pd.to_numeric, errors='coerce')
        df[self.cat_features] = df[self.cat_features].astype(str).fillna("Unknown")
        df[self.text_features] = df[self.text_features].fillna("").astype(str)
        
        # Drop rows with missing target
        initial_count = len(df)
        df.dropna(subset=[self.target], inplace=True)
        logger.info(f"Dropped {initial_count - len(df)} rows with missing target")
        
        # Fill remaining numerical NAs with median
        df[self.num_features] = df[self.num_features].fillna(df[self.num_features].median())
        
        return df

    def _text_preprocessor(self, text_series):
        """Custom text preprocessing function that handles Series input"""
        return text_series.astype(str).str.lower()

    def _build_pipelines(self):
        """Build the preprocessing pipelines"""
        logger.info("Building preprocessing pipelines...")
        
        # Numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Text pipeline - modified to handle Series input properly
        text_pipeline = Pipeline([
            ('text_preprocessor', FunctionTransformer(
                self._text_preprocessor,
                validate=False,
                feature_names_out='one-to-one'
            )),
            ('vectorizer', TfidfVectorizer(
                stop_words='english',
                max_features=100,
                ngram_range=(1, 2),
                dtype=np.float32  # Added for memory efficiency
            ))
        ])
        
        return num_pipeline, cat_pipeline, text_pipeline

    def preprocess(self, input_path: str):
        """Full preprocessing pipeline"""
        try:
            # Load and validate data
            logger.info(f"Loading data from {input_path}")
            df = pd.read_csv(input_path)
            
            # Validate all required columns exist
            required_columns = self.num_features + self.cat_features + self.text_features + [self.target]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean the data
            df = self._clean_data(df)
            
            # Build pipelines
            num_pipeline, cat_pipeline, text_pipeline = self._build_pipelines()
            
            # Create the full pipeline
            transformers = [
                ('num', num_pipeline, self.num_features),
                ('cat', cat_pipeline, self.cat_features),
                ('text', text_pipeline, self.text_features[0])
            ]
            
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                verbose_feature_names_out=False,
                sparse_threshold=0  # Force dense output
            )
            
            # Transform the data
            logger.info("Transforming all features...")
            X = preprocessor.fit_transform(df)
            y = df[self.target].values
            
            # Ensure X is 2D array
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                
            logger.info(f"Final combined shape: {X.shape}")
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
                logger.info(f"Generated {len(feature_names)} feature names")
            except Exception as e:
                logger.warning(f"Could not get feature names: {str(e)}")
                feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
            
            # Verify shapes match
            if X.shape[1] != len(feature_names):
                raise ValueError(
                    f"Feature dimension mismatch. X has {X.shape[1]} features, "
                    f"but {len(feature_names)} feature names were generated"
                )
            
            # Create output directory
            output_dir = Path("data/processed")
            output_dir.mkdir(exist_ok=True, parents=True)
            model_dir = Path("model")
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Define output paths
            output_path = output_dir / f"processed_{Path(input_path).stem}.npz"
            
            # Save processed data in multiple formats
            np.savez(
                output_path,
                X=X,
                y=y,
                feature_names=feature_names,
                allow_pickle=True
            )
            
            # Save transformed features and target
            pd.DataFrame(X, columns=feature_names).to_csv(
                output_dir / "X_transformed.csv", 
                index=False
            )
            pd.DataFrame(y, columns=[self.target]).to_csv(
                output_dir / "y_transformed.csv", 
                index=False
            )
            
            # Save cleaned data
            df.to_csv(output_dir / "cleaned_data.csv", index=False)
            
            # Save preprocessor
            from joblib import dump
            dump(preprocessor, model_dir / "preprocessor.joblib")
            
            logger.info(f"Successfully saved all processed data to {output_dir}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Preprocessing failed: {str(e)}")