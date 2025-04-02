# src/pipeline.py
import logging
from pathlib import Path
from ingest_data import CSVDataIngestor
from preprocess import preprocess_data
from train import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(input_file="data/raw/sepsis_datasets.csv"):
    """End-to-end pipeline execution"""
    try:
        # Configuration
        config = {
            'raw_data_dir': 'data/raw',
            'processed_dir': 'data/processed'
        }
        
        # 1. Ingestion
        logger.info("Ingesting data...")
        ingestor = CSVDataIngestor(config)  # Pass config here
        ingestion_result = ingestor.ingest_csv(input_file)
        
        if ingestion_result['status'] != 'success':
            raise RuntimeError(f"Ingestion failed: {ingestion_result.get('error')}")
        
        # 2. Preprocessing
        logger.info("Preprocessing data...")
        processed_path = preprocess_data(ingestion_result['raw_data'])
        
        # 3. Training
        logger.info("Training model...")
        model_path = train_model(processed_path)
        
        logger.info(f"Pipeline completed successfully!\n"
                   f"Raw data: {ingestion_result['raw_data']}\n"
                   f"Processed data: {processed_path}\n"
                   f"Model: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()