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
        'model_path': 'model/sepsis_xgboost_model.joblib',
        'drift_thresholds': {
            'data_drift_threshold': 0.1,
            'concept_drift_threshold': 0.05,
            'label_drift_threshold': 0.1,
            'evidently_drift_threshold': 0.2,
            'significant_drift_ratio': 0.3,
            'data_quality_threshold': 0.15  # Added this line
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
            
            # Parse results - this handles the new Evidently report structure
            result = report.as_dict()
            
            # Initialize metrics with default values
            evidently_metrics = {
                'n_features': 0,
                'n_drifted': 0,
                'share_drifted': 0,
                'dataset_drift': False
            }
            
            # Extract metrics from the report
            for metric in result['metrics']:
                if metric['metric'] == 'DatasetDriftMetric':
                    drift_result = metric['result']
                    evidently_metrics.update({
                        'n_features': drift_result.get('number_of_columns', 0),
                        'n_drifted': drift_result.get('number_of_drifted_columns', 0),
                        'share_drifted': drift_result.get('share_of_drifted_columns', 0),
                        'dataset_drift': drift_result.get('dataset_drift', False)
                    })
                    break
            
            evidently_drift = (evidently_metrics['share_drifted'] > self.config['drift_thresholds']['evidently_drift_threshold'])
            
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
                'label_drift': {'detected': False, 'metrics': {}},
                'data_quality_drift': {'detected': False, 'metrics': {}}  # Added this line
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
            quality_drift_results, quality_drift_detected = self._detect_data_quality_drift(X_new)  # Added this line
            
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
                },
                'data_quality_drift': {  # Added this block
                    'detected': quality_drift_detected,
                    'metrics': quality_drift_results,
                    'threshold': self.config['drift_thresholds'].get('data_quality_threshold', 0.15)
                }
            })
            
            # Determine overall results - updated to include quality drift
            results['drift_detected'] = any([
                results['data_drift']['detected'],
                results['concept_drift']['detected'],
                results['label_drift']['detected'],
                results['data_quality_drift']['detected']  # Added this line
            ])
            
            # Updated retraining recommendation logic to include quality drift
            results['retraining_recommended'] = (
                results['concept_drift']['detected'] or
                (results['data_drift']['detected'] and results['label_drift']['detected']) or
                (results['data_quality_drift']['detected'] and 
                results['data_quality_drift']['metrics'].get('dataset_metrics', {}).get('avg_missing_value_diff', 0) > 0.2)
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
        

    def _detect_data_quality_drift(self, X_new):
        """
        Detect data quality drift by comparing statistical properties and data quality metrics
        between reference and new data.
        """
        if self.X_ref is None:
            logger.warning("No reference data for data quality drift detection")
            return {}, False
            
        try:
            quality_metrics = {}
            quality_drift_detected = False
            
            # Ensure we don't exceed feature dimensions
            n_features = min(X_new.shape[1], len(self.feature_names), self.X_ref.shape[1])
            
            # 1. Check for missing values
            ref_missing_ratio = np.isnan(self.X_ref).mean(axis=0)[:n_features]
            new_missing_ratio = np.isnan(X_new).mean(axis=0)[:n_features]
            missing_diff = np.abs(ref_missing_ratio - new_missing_ratio)
            
            # 2. Statistical properties (mean, std, min, max, skewness)
            ref_mean = np.nanmean(self.X_ref, axis=0)[:n_features]
            new_mean = np.nanmean(X_new, axis=0)[:n_features]
            mean_diff = np.abs(ref_mean - new_mean)
            
            ref_std = np.nanstd(self.X_ref, axis=0)[:n_features]
            new_std = np.nanstd(X_new, axis=0)[:n_features]
            std_diff = np.abs(ref_std - new_std)
            
            # Calculate min and max differences
            ref_min = np.nanmin(self.X_ref, axis=0)[:n_features]
            new_min = np.nanmin(X_new, axis=0)[:n_features]
            min_diff = np.abs(ref_min - new_min)
            
            ref_max = np.nanmax(self.X_ref, axis=0)[:n_features]
            new_max = np.nanmax(X_new, axis=0)[:n_features]
            max_diff = np.abs(ref_max - new_max)
            
            # 3. Calculate skewness differences
            def safe_skewness(x):
                if np.all(np.isnan(x)):
                    return 0
                x_no_nan = x[~np.isnan(x)]
                if len(x_no_nan) <= 1:
                    return 0
                return ((x_no_nan - np.mean(x_no_nan)) ** 3).mean() / (np.std(x_no_nan) ** 3)
            
            ref_skew = np.array([safe_skewness(self.X_ref[:, i]) for i in range(n_features)])
            new_skew = np.array([safe_skewness(X_new[:, i]) for i in range(n_features)])
            skew_diff = np.abs(ref_skew - new_skew)
            
            # 4. Check for outliers using IQR method
            def calc_outlier_ratio(data, axis=0):
                q1 = np.nanpercentile(data, 25, axis=axis)
                q3 = np.nanpercentile(data, 75, axis=axis)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if axis == 0:  # Per feature
                    outliers = np.zeros(data.shape[1])
                    for i in range(data.shape[1]):
                        feature_data = data[:, i]
                        valid_data = feature_data[~np.isnan(feature_data)]
                        if len(valid_data) > 0:
                            outliers[i] = np.sum((valid_data < lower_bound[i]) | 
                                            (valid_data > upper_bound[i])) / len(valid_data)
                    return outliers
                return None
            
            ref_outlier_ratio = calc_outlier_ratio(self.X_ref)[:n_features]
            new_outlier_ratio = calc_outlier_ratio(X_new)[:n_features]
            outlier_diff = np.abs(ref_outlier_ratio - new_outlier_ratio)
            
            # 5. Calculate quality score per feature
            quality_scores = {}
            drifted_features = []
            drift_threshold = self.config['drift_thresholds'].get('data_quality_threshold', 0.15)
            
            # Compile scores for each feature
            for i in range(n_features):
                feature_name = self.feature_names[i]
                
                # Composite quality score (weighted average of the different metrics)
                quality_score = (
                    0.3 * missing_diff[i] + 
                    0.15 * (mean_diff[i] / max(ref_std[i], 1e-10)) +  # Normalize by std dev
                    0.15 * (std_diff[i] / max(ref_std[i], 1e-10)) +
                    0.1 * min_diff[i] / max(abs(ref_min[i]), 1e-10) +
                    0.1 * max_diff[i] / max(abs(ref_max[i]), 1e-10) +
                    0.1 * skew_diff[i] + 
                    0.1 * outlier_diff[i]
                )
                
                quality_scores[feature_name] = float(quality_score)  # Convert to Python float for JSON serialization
                
                if quality_score > drift_threshold:
                    drifted_features.append(feature_name)
            
            # Determine if quality drift is detected based on proportion of drifted features
            quality_drift_detected = len(drifted_features) > n_features * self.config['drift_thresholds'].get(
                'significant_drift_ratio', 0.3)
            
            # Detailed metrics for reporting
            quality_metrics = {
                'quality_scores': quality_scores,
                'drifted_features': drifted_features,
                'feature_metrics': {
                    f"Feature_{i}": {
                        'missing_value_diff': float(missing_diff[i]),
                        'mean_diff_normalized': float(mean_diff[i] / max(ref_std[i], 1e-10)),
                        'std_diff_normalized': float(std_diff[i] / max(ref_std[i], 1e-10)),
                        'skewness_diff': float(skew_diff[i]),
                        'outlier_ratio_diff': float(outlier_diff[i])
                    } for i in range(min(5, n_features))  # Limit to top 5 features for report size
                },
                'dataset_metrics': {
                    'avg_missing_value_diff': float(np.mean(missing_diff)),
                    'avg_std_diff': float(np.mean(std_diff / np.maximum(ref_std, 1e-10))),
                    'avg_outlier_diff': float(np.mean(outlier_diff))
                }
            }
            
            # Add visualization for data quality drift
            self._visualize_data_quality_drift(
                quality_scores, 
                drift_threshold, 
                os.path.join(self.config['visualizations_dir'], 
                        f"quality_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            )
            
            return quality_metrics, quality_drift_detected
            
        except Exception as e:
            logger.error(f"Data quality drift detection failed: {str(e)}", exc_info=True)
            return {}, False

    def _visualize_data_quality_drift(self, quality_scores, threshold, filepath):
        """Create visualization for data quality drift"""
        try:
            plt.figure(figsize=(12, 6))
            features = list(quality_scores.keys())
            scores = list(quality_scores.values())
            
            # Sort by score for better visualization
            sorted_indices = np.argsort(scores)[::-1]
            features = [features[i] for i in sorted_indices[:15]]  # Top 15 features
            scores = [scores[i] for i in sorted_indices[:15]]
            
            plt.barh(features, scores)
            plt.axvline(x=threshold, color='red', linestyle='--', label='Drift Threshold')
            plt.xlabel('Data Quality Drift Score')
            plt.title('Data Quality Drift Analysis')
            plt.legend()
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create data quality drift visualization: {str(e)}")

    def _detect_label_drift(self, y_new):
        """Detect label distribution drift"""
        try:
            if self.y_ref is None:
                logger.warning("No reference labels for label drift detection")
                return {}, False
                
            # Calculate distributions with minlength to ensure same shape
            ref_dist = np.bincount(self.y_ref.astype(int), minlength=2) / len(self.y_ref)
            new_dist = np.bincount(y_new.astype(int), minlength=2) / len(y_new)
            
            # Calculate maximum difference
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