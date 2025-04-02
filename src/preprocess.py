#!/usr/bin/env python
"""
Enhanced Preprocessing Module for Clinical Data
- Handles file input/output
- Integrates PubMedBERT tokenization
- Maintains compatibility with ingestion pipeline
- Adds comprehensive error handling
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataPreprocessor:
    def __init__(self, max_text_length: int = 128):
        self.tokenizer = self._initialize_tokenizer()
        self.max_text_length = max_text_length
        self.feature_spec = {
            'numerical': ['Age', 'Heart_Rate', 'BP_Systolic', 'BP_Diastolic',
                         'Temperature', 'Respiratory_Rate', 'WBC_Count', 'Lactate_Level'],
            'categorical': ['Gender', 'Comorbidities'],
            'text': ['Clinical_Notes']
        }

    def _initialize_tokenizer(self):
        """Initialize tokenizer with error handling"""
        try:
            return AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {e}")
            raise

    def _tokenize_text(self, text: str) -> np.ndarray:
        """Safe text tokenization with padding/truncation"""
        try:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='np'
            )
            return encoded['input_ids'][0]
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {e}")
            return np.zeros(self.max_text_length, dtype=np.int64)

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input dataframe structure"""
        missing_cols = []
        for col_group in self.feature_spec.values():
            missing_cols.extend([col for col in col_group if col not in df.columns])
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning pipeline"""
        # Handle missing values
        num_cols = self.feature_spec['numerical']
        cat_cols = self.feature_spec['categorical']
        
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        df[cat_cols] = df[cat_cols].astype(str).fillna("Unknown")
        
        return df

    def preprocess(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Main preprocessing workflow"""
        if not self._validate_input(df):
            raise ValueError("Invalid input data structure")
        
        df = self._clean_data(df)
        
        # Tokenize clinical notes
        df['Clinical_Notes'] = df['Clinical_Notes'].apply(self._tokenize_text)
        
        return {
            'numerical': df[self.feature_spec['numerical']].values,
            'categorical': df[self.feature_spec['categorical']].values,
            'text': df['Clinical_Notes'].values
        }

def preprocess_data(input_path: Union[str, Path], output_dir: str = "data/processed") -> str:
    """
    File-level preprocessing function
    Args:
        input_path: Path to raw data file (CSV/Parquet)
        output_dir: Directory to save processed data
    Returns:
        Path to processed file
    """
    try:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Read input file
        logger.info(f"Loading data from {input_path}")
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Process data
        preprocessor = ClinicalDataPreprocessor()
        processed = preprocessor.preprocess(df)
        
        # Save processed data
        output_path = output_dir / f"processed_{input_path.name}"
        np.savez(output_path.with_suffix('.npz'), **processed)
        df.to_csv(output_path)
        logger.info(f"Saved processed data to {output_path.with_suffix('.npz')}")
        
        return str(output_path.with_suffix('.npz'))
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        processed_file = preprocess_data("data/raw/sepsis_datasets.csv")
        print(f"Preprocessing complete. Output: {processed_file}")
    except Exception as e:
        print(f"Preprocessing failed: {e}")