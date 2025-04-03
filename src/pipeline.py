# src/pipeline.py
import logging
import json
import os
from pathlib import Path
from ingest_data import CSVDataIngestor
from preprocess import ClinicalPreprocessor
from train import train_model
from evaluate import evaluate_model
from monitor import DriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_monitoring(config):
    """Initialize monitoring directory and history file"""
    monitor_dir = config['monitoring']['monitor_dir']
    history_file = config['monitoring']['history_file']
    
    # Create monitoring directory if it doesn't exist
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Initialize empty history file if it doesn't exist
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump([], f)
        logger.info(f"Created new monitoring history file at {history_file}")

def run_pipeline(input_file="data/raw/sepsis_datasets.csv", monitor=False):
    """End-to-end pipeline execution with optional monitoring"""
    try:
        # Configuration
        config = {
            'raw_data_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'model_dir': 'model',
            'test_data_path': 'data/processed/test_data.npz',
            'monitoring': {
                'reference_data_path': 'data/processed/reference_data.npz',
                'history_file': 'monitoring/drift_history.json',
                'monitor_dir': 'monitoring',
                'drift_thresholds': {
                    'data_drift_threshold': 0.1,
                    'concept_drift_threshold': 0.05,
                    'label_drift_threshold': 0.1,
                }
            }
        }
        
        # Initialize monitoring system if enabled
        if monitor:
            initialize_monitoring(config)
        
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
        
        # 3. Training
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
        
        # 5. Optional: Drift Monitoring
        if monitor:
            logger.info("Starting drift monitoring...")
            try:
                # Initialize drift detector
                monitoring_config = config['monitoring'].copy()
                monitoring_config['model_path'] = model_path
                
                detector = DriftDetector(monitoring_config)
                
                # Analyze drift using the test data
                drift_results, retrain_recommended = detector.analyze_drift(test_data_path)
                
                # Save the initial reference setup to history
                if 'notes' in drift_results and 'First run' in drift_results['notes']:
                    # Load current history
                    with open(monitoring_config['history_file'], 'r') as f:
                        history = json.load(f)
                    
                    # Add our initial results
                    history.append(drift_results)
                    
                    # Save back to file
                    with open(monitoring_config['history_file'], 'w') as f:
                        json.dump(history, f, indent=2)
                    
                    logger.info(f"Saved initial reference setup to drift history")
                
                logger.info(f"Drift detection results: {drift_results}")
                
                if retrain_recommended:
                    logger.warning("DRIFT DETECTED: Retraining recommended!")
                else:
                    logger.info("No significant drift detected")
                    
            except Exception as e:
                logger.error(f"Drift monitoring failed: {str(e)}", exc_info=True)
        
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
    run_pipeline(monitor=True)