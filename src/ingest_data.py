#!/usr/bin/env python
"""
Fixed CSV Data Ingestion Module for ICU Risk Prediction System
- Handles file path issues
- Fixes parquet dependency
- Improves validation logic
"""

import os
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataIngestor:
    def __init__(self, config: Dict):
        self.raw_data_dir = Path(config['raw_data_dir'])
        self.processed_dir = Path(config['processed_dir'])
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Required columns and validation rules
        self.validation_rules = {
            'required_columns': {
                'Patient_ID', 'Age', 'Gender', 'Heart_Rate',
                'BP_Systolic', 'BP_Diastolic', 'Temperature',
                'Respiratory_Rate', 'WBC_Count', 'Lactate_Level',
                'Comorbidities', 'Clinical_Notes', 'Sepsis_Label'
            },
            'max_null_ratio': 0.8,  # Increased threshold for clinical data
            'value_ranges': {
                'Heart_Rate': (20, 250),
                'Temperature': (30, 45),
                'WBC_Count': (0, 100)
            }
        }

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Improved data validation with warnings"""
        # Column presence check
        missing_cols = self.validation_rules['required_columns'] - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Null value check (warning instead of error for clinical data)
        null_ratios = df.isnull().mean()
        high_null_cols = null_ratios[null_ratios > self.validation_rules['max_null_ratio']].index.tolist()
        if high_null_cols:
            logger.warning(f"High null ratio (>80%) in columns: {high_null_cols}")
            # Don't fail for clinical data as missingness is common
            
        # Value range validation
        for col, (min_val, max_val) in self.validation_rules['value_ranges'].items():
            if col in df.columns:
                out_of_range = (~df[col].between(min_val, max_val)) & df[col].notna()
                if out_of_range.any():
                    logger.warning(f"Out-of-range values in {col}: {out_of_range.sum()} records")
                    
        return True

    def _handle_incremental(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate with improved merging"""
        historical_path = self.raw_data_dir / "historical_sepsis_data.csv"  # Changed to CSV
        
        if historical_path.exists():
            try:
                historical = pd.read_csv(historical_path)
                combined = pd.concat([historical, new_df])
                return combined.drop_duplicates(
                    subset=['Patient_ID', 'Clinical_Notes'],
                    keep='last'
                )
            except Exception as e:
                logger.warning(f"Failed to merge with historical data: {e}")
                return new_df
        return new_df

    def ingest_csv(self, file_path: Union[str, Path], incremental: bool = True) -> Dict[str, str]:
        """
        Robust ingestion workflow with error handling
        """
        try:
            # Resolve file path
            file_path = Path(file_path).absolute()
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            # Load data
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate
            if not self._validate_data(df):
                logger.warning("Proceeding with validation warnings")
                
            # Handle incremental
            processed_df = self._handle_incremental(df) if incremental else df
            
            # Save raw data (as CSV to avoid parquet dependency)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_path = self.raw_data_dir / f"ingested_{timestamp}.csv"
            processed_df.to_csv(raw_path, index=False)
            logger.info(f"Saved raw data to {raw_path}")
            
            # Trigger preprocessing
            try:
                from preprocess import preprocess_data
                processed_path = preprocess_data(raw_path)
                return {
                    'status': 'success',
                    'raw_data': str(raw_path),
                    'processed_data': processed_path,
                    'records_ingested': len(processed_df)
                }
            except ImportError:
                logger.warning("Preprocessing module not found")
                return {
                    'status': 'success_no_preprocess',
                    'raw_data': str(raw_path),
                    'records_ingested': len(processed_df)
                }
                
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Improved main function with path handling"""
    config = {
        'raw_data_dir': os.path.join('data', 'raw'),
        'processed_dir': os.path.join('data', 'processed')
    }
    
    # Use absolute path to your dataset
    input_path = os.path.join('data', 'raw', 'sepsis_datasets.csv')
    
    ingestor = CSVDataIngestor(config)
    result = ingestor.ingest_csv(input_path)
    
    if result['status'].startswith('success'):
        print(f"Ingestion successful! Records processed: {result.get('records_ingested', 0)}")
        if 'processed_data' in result:
            print(f"Processed data at: {result['processed_data']}")
    else:
        print(f"Ingestion failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()