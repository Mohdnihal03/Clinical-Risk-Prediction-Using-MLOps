import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects data, concept, and label drift to determine when model retraining is necessary.
    """
    def __init__(self, config=None):
        if config is None:
            self.config = {
                'reference_data_path': 'data/processed/reference_data.npz',
                'model_path': 'model/sepsis_xgboost_model.joblib',
                'drift_thresholds': {
                    'data_drift_threshold': 0.1,  # KS test threshold
                    'concept_drift_threshold': 0.05,  # Performance drop threshold
                    'label_drift_threshold': 0.1,  # Label distribution threshold
                },
                'monitor_dir': 'monitoring',
                'history_file': 'monitoring/drift_history.json'
            }
        else:
            self.config = config
            
        # Create monitoring directory
        os.makedirs(self.config['monitor_dir'], exist_ok=True)
        
        # Initialize drift history if doesn't exist
        if not os.path.exists(self.config['history_file']):
            with open(self.config['history_file'], 'w') as f:
                json.dump([], f)
                
        # Load reference data if available
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference data for comparison"""
        try:
            if os.path.exists(self.config['reference_data_path']):
                data = np.load(self.config['reference_data_path'])
                self.X_ref = data['X']
                self.y_ref = data['y']
                if 'feature_names' in data:
                    self.feature_names = list(data['feature_names'])
                else:
                    self.feature_names = [f"Feature_{i}" for i in range(self.X_ref.shape[1])]
                logger.info(f"Loaded reference data with shape {self.X_ref.shape}")
            else:
                logger.warning("Reference data not found")
                self.X_ref = None
                self.y_ref = None
                self.feature_names = None
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            self.X_ref = None
            self.y_ref = None
            self.feature_names = None
    
    def _detect_data_drift(self, X_new):
        """
        Detect data drift using Kolmogorov-Smirnov test
        Returns drift score per feature and boolean if drift detected
        """
        if self.X_ref is None:
            logger.warning("No reference data available for data drift detection")
            return {}, False
            
        drift_scores = {}
        drifted_features = []
        
        # Calculate KS statistic for each feature
        for i in range(X_new.shape[1]):
            if i < len(self.feature_names):
                feature_name = self.feature_names[i]
            else:
                feature_name = f"Feature_{i}"
                
            # Skip if all values are the same (causes KS test to fail)
            if len(np.unique(X_new[:, i])) <= 1 or len(np.unique(self.X_ref[:, i])) <= 1:
                drift_scores[feature_name] = 0
                continue
                
            # Perform KS test
            ks_stat, p_value = ks_2samp(self.X_ref[:, i], X_new[:, i])
            drift_scores[feature_name] = ks_stat
            
            # Check if drift detected for this feature
            if ks_stat > self.config['drift_thresholds']['data_drift_threshold']:
                drifted_features.append(feature_name)
        
        # Sort features by drift score
        drift_scores = {k: v for k, v in sorted(
            drift_scores.items(), key=lambda item: item[1], reverse=True
        )}
        
        # Determine if significant drift detected
        data_drift_detected = len(drifted_features) > len(self.feature_names) * 0.3  # If >30% features drifted
        
        if data_drift_detected:
            logger.warning(f"Data drift detected in {len(drifted_features)} features")
        
        return drift_scores, data_drift_detected
    
    def _detect_concept_drift(self, X_new, y_new):
        """
        Detect concept drift by comparing model performance
        between reference data and new data
        """
        try:
            if self.X_ref is None or self.y_ref is None:
                logger.warning("No reference data available for concept drift detection")
                return {}, False
                
            # Load model
            model = joblib.load(self.config['model_path'])
            
            # Get baseline performance on reference data
            y_ref_pred = model.predict(self.X_ref)
            ref_metrics = {
                'accuracy': accuracy_score(self.y_ref, y_ref_pred),
                'precision': precision_score(self.y_ref, y_ref_pred),
                'recall': recall_score(self.y_ref, y_ref_pred),
                'f1': f1_score(self.y_ref, y_ref_pred)
            }
            
            # Get performance on new data
            y_new_pred = model.predict(X_new)
            new_metrics = {
                'accuracy': accuracy_score(y_new, y_new_pred),
                'precision': precision_score(y_new, y_new_pred),
                'recall': recall_score(y_new, y_new_pred),
                'f1': f1_score(y_new, y_new_pred)
            }
            
            # Calculate differences
            metric_diffs = {
                metric: ref_metrics[metric] - new_metrics[metric]
                for metric in ref_metrics
            }
            
            # Determine if concept drift detected based on F1 score
            concept_drift_detected = metric_diffs['f1'] > self.config['drift_thresholds']['concept_drift_threshold']
            
            if concept_drift_detected:
                logger.warning(f"Concept drift detected: F1 score dropped by {metric_diffs['f1']:.4f}")
            
            return {
                'reference_metrics': ref_metrics,
                'new_metrics': new_metrics,
                'differences': metric_diffs
            }, concept_drift_detected
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {str(e)}")
            return {}, False
    
    def _detect_label_drift(self, y_new):
        """
        Detect label drift (changes in class distribution)
        """
        if self.y_ref is None:
            logger.warning("No reference data available for label drift detection")
            return {}, False
        
        # Calculate class distribution in reference data
        ref_class_dist = np.bincount(self.y_ref.astype(int)) / len(self.y_ref)
        
        # Calculate class distribution in new data
        new_class_dist = np.bincount(y_new.astype(int)) / len(y_new)
        
        # Ensure distributions have same shape (in case new data has different classes)
        if len(ref_class_dist) != len(new_class_dist):
            # Extend the smaller one with zeros
            max_len = max(len(ref_class_dist), len(new_class_dist))
            ref_class_dist = np.pad(ref_class_dist, (0, max_len - len(ref_class_dist)))
            new_class_dist = np.pad(new_class_dist, (0, max_len - len(new_class_dist)))
        
        # Calculate absolute difference
        distribution_diff = np.abs(ref_class_dist - new_class_dist).max()
        
        # Determine if label drift detected
        label_drift_detected = distribution_diff > self.config['drift_thresholds']['label_drift_threshold']
        
        if label_drift_detected:
            logger.warning(f"Label drift detected: Class distribution changed by {distribution_diff:.4f}")
        
        return {
            'reference_distribution': ref_class_dist.tolist(),
            'new_distribution': new_class_dist.tolist(),
            'distribution_diff': distribution_diff
        }, label_drift_detected
    
    def analyze_drift(self, new_data_path):
        """Analyze all types of drift and return serializable results"""
        try:
            # Load new data
            data = np.load(new_data_path)
            X_new = data['X']
            y_new = data['y']
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            drift_results = {
                'timestamp': timestamp,
                'data_file': str(new_data_path),  # Convert Path to string
                'drift_detected': False,
                'retraining_recommended': False,
                'data_drift': {
                    'detected': False,
                    'top_features': {},
                    'threshold': float(self.config['drift_thresholds']['data_drift_threshold'])
                },
                'concept_drift': {
                    'detected': False,
                    'metrics': {
                        'reference_metrics': {},
                        'new_metrics': {},
                        'differences': {}
                    },
                    'threshold': float(self.config['drift_thresholds']['concept_drift_threshold'])
                },
                'label_drift': {
                    'detected': False,
                    'metrics': {
                        'reference_distribution': [],
                        'new_distribution': [],
                        'distribution_diff': 0.0
                    },
                    'threshold': float(self.config['drift_thresholds']['label_drift_threshold'])
                }
            }
            
            # Check if reference data exists
            if self.X_ref is None or self.y_ref is None:
                logger.info("Setting current data as reference baseline")
                np.savez(
                    self.config['reference_data_path'],
                    X=X_new,
                    y=y_new,
                    feature_names=np.array(self.feature_names) if hasattr(self, 'feature_names') else None
                )
                drift_results['notes'] = "Initial reference data established"
                self._save_drift_history(drift_results)
                return drift_results, False
            
            # Data drift detection
            data_drift_scores, data_drift_detected = self._detect_data_drift(X_new)
            drift_results['data_drift'].update({
                'detected': bool(data_drift_detected),  # Ensure boolean
                'top_features': {k: float(v) for k, v in 
                            sorted(data_drift_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:5]}
            })
            
            # Concept drift detection
            concept_metrics, concept_drift_detected = self._detect_concept_drift(X_new, y_new)
            drift_results['concept_drift'].update({
                'detected': bool(concept_drift_detected),
                'metrics': {
                    'reference_metrics': {k: float(v) for k, v in concept_metrics['reference_metrics'].items()},
                    'new_metrics': {k: float(v) for k, v in concept_metrics['new_metrics'].items()},
                    'differences': {k: float(v) for k, v in concept_metrics['differences'].items()}
                }
            })
            
            # Label drift detection
            label_metrics, label_drift_detected = self._detect_label_drift(y_new)
            drift_results['label_drift'].update({
                'detected': bool(label_drift_detected),
                'metrics': {
                    'reference_distribution': [float(x) for x in label_metrics['reference_distribution']],
                    'new_distribution': [float(x) for x in label_metrics['new_distribution']],
                    'distribution_diff': float(label_metrics['distribution_diff'])
                }
            })
            
            # Determine overall drift
            drift_results['drift_detected'] = bool(
                drift_results['data_drift']['detected'] or
                drift_results['concept_drift']['detected'] or
                drift_results['label_drift']['detected']
            )
            
            # Recommend retraining
            drift_results['retraining_recommended'] = bool(
                drift_results['concept_drift']['detected'] or
                (drift_results['data_drift']['detected'] and 
                drift_results['label_drift']['detected'])
            )
            
            # Save results
            self._save_drift_history(drift_results)
            
            return drift_results, drift_results['retraining_recommended']
            
        except Exception as e:
            logger.error(f"Drift analysis failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, False
    
    def _save_drift_history(self, drift_results):
        """Save drift analysis results to history file with proper serialization"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config['history_file']), exist_ok=True)
            
            # Load existing history or initialize empty list
            history = []
            if os.path.exists(self.config['history_file']):
                try:
                    with open(self.config['history_file'], 'r') as f:
                        history = json.load(f)
                except (json.JSONDecodeError, IOError):
                    history = []
            
            # Append new results
            history.append(drift_results)
            
            # Write with atomic replacement
            temp_file = self.config['history_file'] + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)  # Use default=str for non-serializable objects
            
            # Atomic rename
            os.replace(temp_file, self.config['history_file'])
            
            logger.info(f"Successfully saved drift history to {self.config['history_file']}")
        except Exception as e:
            logger.error(f"Failed to save drift history: {str(e)}", exc_info=True)
            raise
        
    def _generate_drift_visualizations(self, X_new, y_new, data_drift_scores, 
                                       concept_drift_metrics, label_drift_metrics):
        """Generate visualizations for drift analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("monitoring", exist_ok=True)
        viz_dir = Path(self.config['monitor_dir']) / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Data Drift Visualization
        if data_drift_scores:
            plt.figure(figsize=(10, 6))
            # Get top 10 drifted features
            features = list(data_drift_scores.keys())[:10]
            scores = [data_drift_scores[f] for f in features]
            
            plt.barh(features, scores)
            plt.axvline(x=self.config['drift_thresholds']['data_drift_threshold'], 
                       color='red', linestyle='--', label='Drift Threshold')
            plt.xlabel('Drift Score (KS Statistic)')
            plt.title('Top 10 Features by Drift Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / f"data_drift_{timestamp}.png")
            plt.close()
        
        # 2. Concept Drift Visualization
        if concept_drift_metrics and 'reference_metrics' in concept_drift_metrics:
            plt.figure(figsize=(10, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            ref_values = [concept_drift_metrics['reference_metrics'][m] for m in metrics]
            new_values = [concept_drift_metrics['new_metrics'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, ref_values, width, label='Reference Data')
            plt.bar(x + width/2, new_values, width, label='New Data')
            
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, metrics)
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / f"concept_drift_{timestamp}.png")
            plt.close()
        
        # 3. Label Drift Visualization
        if label_drift_metrics and 'reference_distribution' in label_drift_metrics:
            plt.figure(figsize=(8, 6))
            ref_dist = label_drift_metrics['reference_distribution']
            new_dist = label_drift_metrics['new_distribution']
            
            x = np.arange(len(ref_dist))
            width = 0.35
            
            plt.bar(x - width/2, ref_dist, width, label='Reference Data')
            plt.bar(x + width/2, new_dist, width, label='New Data')
            
            plt.ylabel('Class Proportion')
            plt.title('Label Distribution Comparison')
            plt.xticks(x, ['Class ' + str(i) for i in range(len(ref_dist))])
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / f"label_drift_{timestamp}.png")
            plt.close()

def detect_drift(new_data_path, config=None):
    """
    Wrapper function to detect drift and recommend retraining
    
    Returns:
        tuple: (drift_results, retraining_recommended)
    """
    detector = DriftDetector(config)
    return detector.analyze_drift(new_data_path)