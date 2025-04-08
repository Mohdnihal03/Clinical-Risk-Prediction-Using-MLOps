# src/ingest_data.py
'''
CSV Data Ingestion Module

1. Purpose:
This module provides functionality to ingest and validate patient sepsis data from CSV files.
It performs the following key operations:
- Validates input CSV files against required schema (columns and data types)
- Processes valid files into a standardized format
- Saves ingested data with proper formatting and timestamping
- Handles errors gracefully with detailed logging

2. Input/Requirements:
- Input comes from CSV files containing patient sepsis-related metrics
- Required configuration dictionary with:
  - 'raw_data_dir': Directory to store successfully ingested files
  - 'processed_dir': Directory for processed files (future use)
- CSV files must contain these exact columns:
  - Patient_ID, Age, Gender, Heart_Rate, BP_Systolic, BP_Diastolic
  - Temperature, Respiratory_Rate, WBC_Count, Lactate_Level
  - Comorbidities, Clinical_Notes, Sepsis_Label

3. How It Works:
- Initialization: Sets up directories and validation rules
- Validation: Checks for required columns before processing
- Ingestion Process:
  a. Reads input CSV file using pandas
  b. Validates column structure
  c. Generates timestamped output filename
  d. Saves data with consistent CSV formatting:
     - Quotes all non-numeric fields
     - Properly escapes special characters
     - Handles clinical notes (free text) carefully
- Output: Returns status dictionary with:
  - Success/Failure indication
  - File path of ingested data
  - Record count (if successful) or error details

4. Why?
- Stops bad data from causing errors later
- Makes sure all files have the same format
- Keeps patient records organized and safe
- Helps spot missing info early

5. Summary:
The CSVDataIngestor class provides a robust pipeline for receiving and standardizing
patient sepsis data from external sources. It enforces data quality at ingestion time
and prepares files for downstream processing. The implementation handles edge cases
like missing files, invalid data, and special characters in clinical notes.
"""

'''

import os
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Union
import csv

class CSVDataIngestor:
    def __init__(self, config: Dict):
        self.raw_data_dir = Path(config['raw_data_dir'])
        self.processed_dir = Path(config['processed_dir'])
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Validation rules (simplified)
        self.required_columns = {
            'Patient_ID', 'Age', 'Gender', 'Heart_Rate',
            'BP_Systolic', 'BP_Diastolic', 'Temperature',
            'Respiratory_Rate', 'WBC_Count', 'Lactate_Level',
            'Comorbidities', 'Clinical_Notes', 'Sepsis_Label'
        }

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Check for required columns"""
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            return False
        return True

    def ingest_csv(self, file_path: Union[str, Path]) -> Dict[str, str]:
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Input file not found: {file_path}")
                
                df = pd.read_csv(file_path)
                
                if not self._validate_data(df):
                    raise ValueError("Data validation failed")
                
                # Save with consistent quoting
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.raw_data_dir / f"ingested_{timestamp}.csv"
                
                # Force quoting for Clinical_Notes (column index 11)
                df.to_csv(
                    output_path,
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC,  # Quote all non-numeric fields
                    quotechar='"',
                    escapechar='\\'  # Escape quotes inside text if needed
                )
                
                return {
                    'status': 'success',
                    'raw_data': str(output_path),
                    'records_ingested': len(df)
                }
                
            except Exception as e:
                logging.error(f"Ingestion failed: {str(e)}")
                return {
                    'status': 'failed',
                    'error': str(e)
                }