import logging
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Initialize logger at module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class DriftDetector:
    """Complete drift detection with all fixes and robust error handling."""
    
    def __init__(self, config=None):
        # Default configuration with all required keys
        self.default_config = {
            'reference_data_path': 'data/processed/reference_data.npz',
            'model_path': 'models/model.joblib',
            'drift_thresholds': {
                'data_drift_threshold': 0.1,
                'concept_drift_threshold': 0.05,
                'label_drift_threshold': 0.1,
                'evidently_drift_threshold': 0.2,
                'significant_drift_ratio': 0.3
            },
            'monitor_dir': 'monitoring',
            'history_file': 'monitoring/drift_history.json',
            'visualizations_dir': 'monitoring/visualizations'
        }
        
        # Merge configurations
        self.config = {**self.default_config, **(config or {})}
        
        # Ensure all threshold keys exist
        self.config['drift_thresholds'] = {
            **self.default_config['drift_thresholds'],
            **(self.config.get('drift_thresholds', {}))
        }
        
        # Create directories
        os.makedirs(self.config['monitor_dir'], exist_ok=True)
        os.makedirs(self.config['visualizations_dir'], exist_ok=True)
        
        # Initialize history file
        if not os.path.exists(self.config['history_file']):
            with open(self.config['history_file'], 'w') as f:
                json.dump([], f)
                
        # Load reference data
        self.X_ref = None
        self.y_ref = None
        self.feature_names = None
        self._load_reference_data()

    def _load_reference_data(self):
        """Safely load reference data with proper handling of feature names"""
        try:
            if not os.path.exists(self.config['reference_data_path']):
                logger.warning(f"Reference data not found at {self.config['reference_data_path']}")
                return
            
            # Load data first into memory
            with open(self.config['reference_data_path'], 'rb') as f:
                data = np.load(f, allow_pickle=True)
                data_content = {k: data[k] for k in data.files}
            
            self.X_ref = data_content.get('X')
            self.y_ref = data_content.get('y')
            
            # Handle feature names safely - including 0-d array case
            if 'feature_names' in data_content:
                feature_names = data_content['feature_names']
                if feature_names.ndim == 0:  # Handle 0-d array case
                    self.feature_names = [str(feature_names)]
                else:
                    self.feature_names = list(feature_names)
            else:
                self.feature_names = (
                    [f"Feature_{i}" for i in range(self.X_ref.shape[1])] 
                    if self.X_ref is not None else None
                )
            
            logger.info(f"Successfully loaded reference data from {self.config['reference_data_path']}")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}", exc_info=True)
            self.X_ref = self.y_ref = self.feature_names = None

    def _safe_load_data(self, path):
        """Helper method to safely load numpy data"""
        try:
            with open(path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                return {k: data[k] for k in data.files}
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}", exc_info=True)
            return None

    def _detect_data_drift(self, X_new):
        """Comprehensive data drift detection using both KS test and Evidently"""
        if self.X_ref is None or self.feature_names is None:
            logger.warning("No reference data available for drift detection")
            return {'ks_test': {}, 'evidently': {}}, False
            
        # KS Test Drift Detection
        ks_scores = {}
        drifted_features = []
        
        # Ensure we don't exceed feature dimensions
        n_features = min(X_new.shape[1], len(self.feature_names), self.X_ref.shape[1])
        
        for i in range(n_features):
            feature_name = self.feature_names[i]
            try:
                ks_stat, _ = ks_2samp(self.X_ref[:, i], X_new[:, i])
                ks_scores[feature_name] = ks_stat
                if ks_stat > self.config['drift_thresholds']['data_drift_threshold']:
                    drifted_features.append(feature_name)
            except Exception as e:
                logger.warning(f"KS test failed for {feature_name}: {str(e)}")
                ks_scores[feature_name] = 0
        
        ks_drift = len(drifted_features) > n_features * self.config['drift_thresholds']['significant_drift_ratio']
        
        # Evidently Drift Detection
        evidently_metrics = {}
        evidently_drift = False
        
        try:
            ref_df = pd.DataFrame(self.X_ref[:, :n_features], columns=self.feature_names[:n_features])
            new_df = pd.DataFrame(X_new[:, :n_features], columns=self.feature_names[:n_features])
            
            report = Report(metrics=[DataDriftPreset()])
            report.run(current_data=new_df, reference_data=ref_df, column_mapping=None)
            
            # Save report
            report_path = os.path.join(
                self.config['visualizations_dir'], 
                f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            report.save_html(report_path)
            
            # Parse results
            result = json.loads(report.json())['metrics'][0]['result']
            evidently_metrics = {
                'n_features': result['n_features'],
                'n_drifted': result['n_drifted_features'],
                'share_drifted': result['share_drifted_features'],
                'dataset_drift': result['dataset_drift']
            }
            evidently_drift = result['share_drifted_features'] > self.config['drift_thresholds']['evidently_drift_threshold']
            
        except Exception as e:
            logger.error(f"Evidently analysis failed: {str(e)}", exc_info=True)
        
        return {
            'ks_test': {'scores': ks_scores, 'drift_detected': ks_drift},
            'evidently': {'metrics': evidently_metrics, 'drift_detected': evidently_drift}
        }, ks_drift or evidently_drift

    def analyze_drift(self, new_data_path):
        """Complete drift analysis with robust error handling"""
        try:
            # Safely load new data
            new_data = self._safe_load_data(new_data_path)
            if new_data is None or 'X' not in new_data or 'y' not in new_data:
                raise ValueError("Invalid data format - must contain 'X' and 'y'")
                
            X_new, y_new = new_data['X'], new_data['y']
            
            # Initialize results structure
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_file': os.path.basename(new_data_path),
                'drift_detected': False,
                'retraining_recommended': False,
                'data_drift': {'detected': False, 'methods': {}},
                'concept_drift': {'detected': False, 'metrics': {}},
                'label_drift': {'detected': False, 'metrics': {}}
            }
            
            # Handle case where no reference data exists
            if self.X_ref is None:
                logger.info("Initializing new reference dataset")
                try:
                    feature_names = (
                        np.array(self.feature_names) 
                        if hasattr(self, 'feature_names') and self.feature_names is not None 
                        else None
                    )
                    np.savez(
                        self.config['reference_data_path'],
                        X=X_new,
                        y=y_new,
                        feature_names=feature_names
                    )
                    results['note'] = "Initial reference dataset created"
                    self._save_drift_history(results)
                    return results, False
                except Exception as e:
                    logger.error(f"Failed to save reference data: {str(e)}")
                    results['error'] = f"Reference data initialization failed: {str(e)}"
                    return results, False
                
            # Perform all drift detection
            data_drift_results, data_drift_detected = self._detect_data_drift(X_new)
            concept_drift_results, concept_drift_detected = self._detect_concept_drift(X_new, y_new)
            label_drift_results, label_drift_detected = self._detect_label_drift(y_new)
            
            # Update results
            results.update({
                'data_drift': {
                    'detected': data_drift_detected,
                    'methods': {
                        'ks_test': {
                            'detected': data_drift_results['ks_test']['drift_detected'],
                            'top_features': dict(sorted(
                                data_drift_results['ks_test']['scores'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:5]),
                            'threshold': self.config['drift_thresholds']['data_drift_threshold']
                        },
                        'evidently': {
                            'detected': data_drift_results['evidently']['drift_detected'],
                            'metrics': data_drift_results['evidently']['metrics'],
                            'threshold': self.config['drift_thresholds']['evidently_drift_threshold']
                        }
                    }
                },
                'concept_drift': {
                    'detected': concept_drift_detected,
                    'metrics': concept_drift_results,
                    'threshold': self.config['drift_thresholds']['concept_drift_threshold']
                },
                'label_drift': {
                    'detected': label_drift_detected,
                    'metrics': label_drift_results,
                    'threshold': self.config['drift_thresholds']['label_drift_threshold']
                }
            })
            
            # Determine overall results
            results['drift_detected'] = any([
                results['data_drift']['detected'],
                results['concept_drift']['detected'],
                results['label_drift']['detected']
            ])
            
            results['retraining_recommended'] = (
                results['concept_drift']['detected'] or
                (results['data_drift']['detected'] and results['label_drift']['detected'])
            )
            
            self._save_drift_history(results)
            return results, results['retraining_recommended']
            
        except Exception as e:
            logger.error(f"Drift analysis failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, False

    def _detect_concept_drift(self, X_new, y_new):
        """Detect concept drift by comparing model performance"""
        try:
            if self.X_ref is None or self.y_ref is None:
                logger.warning("No reference data for concept drift detection")
                return {}, False
                
            model = joblib.load(self.config['model_path'])
            
            # Reference performance
            y_ref_pred = model.predict(self.X_ref)
            ref_metrics = {
                'accuracy': accuracy_score(self.y_ref, y_ref_pred),
                'precision': precision_score(self.y_ref, y_ref_pred),
                'recall': recall_score(self.y_ref, y_ref_pred),
                'f1': f1_score(self.y_ref, y_ref_pred)
            }
            
            # New data performance
            y_new_pred = model.predict(X_new)
            new_metrics = {
                'accuracy': accuracy_score(y_new, y_new_pred),
                'precision': precision_score(y_new, y_new_pred),
                'recall': recall_score(y_new, y_new_pred),
                'f1': f1_score(y_new, y_new_pred)
            }
            
            # Differences
            metric_diffs = {
                metric: ref_metrics[metric] - new_metrics[metric]
                for metric in ref_metrics
            }
            
            concept_drift = metric_diffs['f1'] > self.config['drift_thresholds']['concept_drift_threshold']
            
            return {
                'reference_metrics': ref_metrics,
                'new_metrics': new_metrics,
                'differences': metric_diffs
            }, concept_drift
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {str(e)}", exc_info=True)
            return {}, False

    def _detect_label_drift(self, y_new):
        """Detect label distribution drift"""
        try:
            if self.y_ref is None:
                logger.warning("No reference labels for label drift detection")
                return {}, False
                
            # Calculate distributions
            ref_dist = np.bincount(self.y_ref.astype(int)) / len(self.y_ref)
            new_dist = np.bincount(y_new.astype(int)) / len(y_new)
            
            # Align distributions
            max_len = max(len(ref_dist), len(new_dist))
            ref_dist = np.pad(ref_dist, (0, max_len - len(ref_dist)))
            new_dist = np.pad(new_dist, (0, max_len - len(new_dist)))
            
            # Calculate difference
            dist_diff = np.abs(ref_dist - new_dist).max()
            label_drift = dist_diff > self.config['drift_thresholds']['label_drift_threshold']
            
            return {
                'reference_distribution': ref_dist.tolist(),
                'new_distribution': new_dist.tolist(),
                'distribution_diff': float(dist_diff)
            }, label_drift
            
        except Exception as e:
            logger.error(f"Label drift detection failed: {str(e)}", exc_info=True)
            return {}, False

    def _save_drift_history(self, results):
        """Save drift results to history file"""
        try:
            # Load existing history
            history = []
            if os.path.exists(self.config['history_file']):
                try:
                    with open(self.config['history_file'], 'r') as f:
                        history = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load history file: {str(e)}")
            
            # Append new results
            history.append(results)
            
            # Save with atomic write
            temp_file = self.config['history_file'] + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            os.replace(temp_file, self.config['history_file'])
            logger.info(f"Saved drift results to {self.config['history_file']}")
            
        except Exception as e:
            logger.error(f"Failed to save drift history: {str(e)}", exc_info=True)
            raise
        
    def _generate_drift_visualizations(self, X_new, y_new, data_drift_scores, 
                                     concept_drift_metrics, label_drift_metrics):
        """Generate visualizations for drift analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Data Drift Visualization (KS Test)
        if data_drift_scores['ks_test']['scores']:
            plt.figure(figsize=(10, 6))
            features = list(data_drift_scores['ks_test']['scores'].keys())[:10]
            scores = [data_drift_scores['ks_test']['scores'][f] for f in features]
            
            plt.barh(features, scores)
            plt.axvline(x=self.config['drift_thresholds']['data_drift_threshold'], 
                       color='red', linestyle='--', label='Drift Threshold')
            plt.xlabel('Drift Score (KS Statistic)')
            plt.title('Top 10 Features by Drift Score (KS Test)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['visualizations_dir'], f"ks_drift_{timestamp}.png"))
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
            plt.savefig(os.path.join(self.config['visualizations_dir'], f"concept_drift_{timestamp}.png"))
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
            plt.savefig(os.path.join(self.config['visualizations_dir'], f"label_drift_{timestamp}.png"))
            plt.close()

def detect_drift(new_data_path, config=None):
    """
    Wrapper function to detect drift and recommend retraining
    
    Args:
        new_data_path: Path to new data in npz format
        config: Optional configuration dictionary
    
    Returns:
        tuple: (drift_results, retraining_recommended)
    """
    detector = DriftDetector(config)
    return detector.analyze_drift(new_data_path)