import logging
import json
import os
from pathlib import Path
from datetime import datetime
from ingest_data import CSVDataIngestor
from preprocess import ClinicalPreprocessor
from train import train_model
from evaluate import evaluate_model
from monitor import DriftDetector
from retrain import ModelRetrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_monitoring(config):
    """Initialize monitoring directory and history file"""
    monitor_dir = config['monitoring']['monitor_dir']
    history_file = config['monitoring']['history_file']
    visualizations_dir = config['monitoring']['visualizations_dir']
    
    # Create monitoring directories if they don't exist
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Initialize empty history file if it doesn't exist
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump([], f)
        logger.info(f"Created new monitoring history file at {history_file}")

def get_default_config():
    """Return default configuration dictionary"""
    return {
        'raw_data_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'model_dir': 'model',
        'test_data_path': 'data/processed/test_data.npz',
        'monitoring': {
            'reference_data_path': 'data/processed/reference_data.npz',
            'history_file': 'monitoring/drift_history.json',
            'monitor_dir': 'monitoring',
            'visualizations_dir': 'monitoring/visualizations',
            'drift_thresholds': {
                'data_drift_threshold': 0.1,
                'concept_drift_threshold': 0.05,
                'label_drift_threshold': 0.1,
                'significant_drift_ratio': 0.3
            }
        }
    }

def run_pipeline(input_file="data/raw/sepsis_datasets.csv", monitor=False, retrain_if_needed=False):
    """End-to-end pipeline execution with optional monitoring and retraining"""
    try:
        # Configuration
        config = get_default_config()
        
        # Initialize monitoring system if enabled
        if monitor or retrain_if_needed:
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
        
        # 3. Training/Retraining
        if retrain_if_needed:
            logger.info("Running automated retraining workflow...")
            retrainer = ModelRetrainer(
                processed_data_path=processed_path,
                monitoring_config=config['monitoring']
            )
            retrain_result = retrainer.run_retraining_workflow()
            
            if retrain_result['conclusion'] == "Retraining successful":
                model_path = retrain_result['model_path']
                test_data_path = retrain_result['test_data_path']
                logger.info(f"Retraining successful. Using new model at {model_path}")
            else:
                logger.info("No retraining needed. Proceeding with existing model.")
                model_path, test_data_path = train_model(processed_path)
        else:
            logger.info("Training model...")
            model_path, test_data_path = train_model(processed_path)

        logger.info(f"Model path: {model_path}")
        logger.info(f"Test data path: {test_data_path}")
        
        # Update config with new model path for monitoring
        config['monitoring']['model_path'] = model_path
        
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
                detector = DriftDetector(config['monitoring'])
                
                # Analyze drift using the test data
                drift_results, retrain_recommended = detector.analyze_drift(test_data_path)
                
                # Save the initial reference setup to history
                if 'notes' in drift_results and 'First run' in drift_results['notes']:
                    # Load current history
                    with open(config['monitoring']['history_file'], 'r') as f:
                        history = json.load(f)
                    
                    # Add our initial results
                    history.append(drift_results)
                    
                    # Save back to file
                    with open(config['monitoring']['history_file'], 'w') as f:
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
        
        return {
            'model_path': model_path,
            'test_data_path': test_data_path,
            'eval_results': eval_results,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline(monitor=True, retrain_if_needed=True)