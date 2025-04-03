# src/pipeline.py
import logging
from pathlib import Path
from ingest_data import CSVDataIngestor
from preprocess import ClinicalPreprocessor
from train import train_model
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(input_file="data/raw/sepsis_datasets.csv"):
    """End-to-end pipeline execution"""
    try:
        # Configuration
        config = {
            'raw_data_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'model_dir': 'model',
            'test_data_path': 'data/processed/test_data.npz'  # New test data path
        }
        
        # 1. Ingestion
        logger.info("Ingesting data...")
        ingestor = CSVDataIngestor(config)
        ingestion_result = ingestor.ingest_csv(input_file)
        
        if ingestion_result['status'] != 'success':
            raise RuntimeError(f"Ingestion failed: {ingestion_result.get('error')}")
        
        # 2. Preprocessing
        logger.info("Preprocessing data...")
        preprocessor = ClinicalPreprocessor()
        processed_path = preprocessor.preprocess(ingestion_result['raw_data'])
        
        # 3. Training (now returns test_data_path)
        logger.info("Training model...")
        model_path, test_data_path = train_model(processed_path)

        logger.info(f"Model path: {model_path}")
        logger.info(f"Test data path: {test_data_path}")
        
        # 4. Evaluation
        logger.info("Starting evaluation...")
        try:
            eval_results = evaluate_model(model_path, test_data_path)
            logger.info(f"Evaluation completed with results: {eval_results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise
        
        logger.info(f"Pipeline completed successfully!\n"
                   f"Raw data: {ingestion_result['raw_data']}\n"
                   f"Processed data: {processed_path}\n"
                   f"Model: {model_path}\n"
                   f"Test data: {test_data_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()