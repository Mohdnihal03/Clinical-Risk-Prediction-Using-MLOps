import logging
import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Import existing modules
from train import train_model
from evaluate import evaluate_model
from monitor import DriftDetector

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ModelRetrainer:
    def __init__(
        self,
        model_path: str = "model/sepsis_xgboost_model.joblib",
        test_data_path: str = "data/processed/test_data.npz",
        processed_data_path: str = "data/processed/train.npz",
        performance_threshold: float = 0.75,
        metrics_history_path: str = "monitoring/metrics_history.json",
        monitoring_config: Optional[Dict] = None,
        versions_dir: str = "versions"  # Add versions directory parameter
    ):
        """Initialize ModelRetrainer with default paths and configuration"""
        # Use absolute paths based on project root
        self.project_root = Path.cwd()
        self.model_path = self.project_root / model_path
        self.test_data_path = self.project_root / test_data_path
        self.processed_data_path = self.project_root / processed_data_path
        self.performance_threshold = performance_threshold
        self.metrics_history_path = self.project_root / metrics_history_path
        self.versions_dir = self.project_root / versions_dir  # Set versions directory
        
        # Gemini API setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        else:
            logger.warning("No Gemini API key provided. LLM analysis will be disabled.")
        
        # Monitoring configuration
        self.monitoring_config = monitoring_config or {
            'reference_data_path': str(self.project_root / 'data/processed/reference_data.npz'),
            'model_path': str(self.model_path),
            'monitor_dir': str(self.project_root / 'monitoring'),
            'history_file': str(self.project_root / 'monitoring/drift_history.json'),
            'visualizations_dir': str(self.project_root / 'monitoring/visualizations'),
            'versions_dir': str(self.versions_dir),  # Add versions directory to config
            'drift_thresholds': {
                'data_drift_threshold': 0.1,
                'concept_drift_threshold': 0.05,
                'label_drift_threshold': 0.1,
                'significant_drift_ratio': 0.3
            }
        }

        # Ensure directories exist
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.test_data_path.parent.mkdir(exist_ok=True, parents=True)
        self.metrics_history_path.parent.mkdir(exist_ok=True, parents=True)
        self.versions_dir.mkdir(exist_ok=True, parents=True)  # Create versions directory
        Path(self.monitoring_config['visualizations_dir']).mkdir(exist_ok=True, parents=True)

        # Initialize metrics history
        self.metrics_history = self._load_metrics_history()

    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical metrics data from JSON file"""
        if not self.metrics_history_path.exists():
            logger.info(f"No metrics history found at {self.metrics_history_path}. Starting fresh.")
            return []
        
        try:
            with open(self.metrics_history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics history: {str(e)}")
            return []

    def _save_metrics_history(self):
        """Save updated metrics history to JSON file"""
        try:
            with open(self.metrics_history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics history: {str(e)}")

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using the new generativeai package"""
        if not self.gemini_api_key:
            return "Gemini API key not available"
            
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    top_p=0.85,
                    max_output_tokens=650
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return f"Error calling Gemini API: {str(e)}"

    def _analyze_with_gemini(self, metrics_trend: List[Dict], drift_details: Dict) -> Dict:
        """Use Gemini to analyze performance trends and drift"""
        if not self.gemini_api_key:
            return {"error": "Gemini API key not configured"}
            
        prompt = f"""
        I'm monitoring a sepsis prediction ML model and need your analysis:
        
        Recent performance metrics (newest last):
        {json.dumps(metrics_trend, indent=2)}
        
        Drift detection details:
        {json.dumps(drift_details, indent=2)}
        
        Please analyze:
        1. Potential causes of performance changes
        2. Whether retraining is recommended
        3. Specific improvements to consider
        4. Recommended data quality checks
        
        Return your analysis as JSON with these fields:
        - analysis: Text analysis of the situation
        - retrain_recommended: Boolean recommendation
        - confidence: Confidence score (0-1)
        - improvements: List of suggested improvements
        - data_checks: List of recommended data checks
        """
        
        response = self._call_gemini_api(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "retrain_recommended": True,
                "confidence": 0.7,
                "improvements": [],
                "data_checks": []
            }

    def _initial_training_required(self) -> bool:
        """Check if initial training is needed"""
        return not self.model_path.exists() or not self.test_data_path.exists()

    def perform_initial_training(self) -> Tuple[str, str]:
        """Perform initial model training if no model exists"""
        logger.info("Performing initial model training...")
        model_path, test_data_path = train_model(str(self.processed_data_path))
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.monitoring_config['model_path'] = model_path
        return model_path, test_data_path

    def evaluate_current_model(self) -> Dict[str, float]:
        """Run evaluation on current model and return metrics"""
        try:
            if self._initial_training_required():
                self.perform_initial_training()
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not self.test_data_path.exists():
                raise FileNotFoundError(f"Test data not found at {self.test_data_path}")
                
            metrics = evaluate_model(str(self.model_path), str(self.test_data_path))
            
            metrics_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics": metrics
            }
            self.metrics_history.append(metrics_entry)
            self._save_metrics_history()
            
            return metrics
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def detect_performance_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if model performance has dropped below threshold
        Returns: (drift_detected, drift_details)
        """
        if len(self.metrics_history) < 2:
            logger.warning("Insufficient history - treating as critical case")
            return True, {"reason": "insufficient_history"}
        
        current_metrics = self.metrics_history[-1]["metrics"]
        f1_score = current_metrics.get("f1", 0)
        auc_score = current_metrics.get("roc_auc", 0)
        
        drift_details = {
            "current_f1": f1_score,
            "current_auc": auc_score,
            "threshold": self.performance_threshold,
            "previous_evaluations": len(self.metrics_history) - 1
        }
        
        # Threshold-based detection
        drift_detected = (f1_score < self.performance_threshold or 
                         auc_score < self.performance_threshold)
        
        if drift_detected:
            drift_details["reason"] = "below_threshold"
            logger.warning(f"Performance drift detected! Metrics below threshold: F1={f1_score}, AUC={auc_score}")
        else:
            # Check for significant decrease
            if len(self.metrics_history) >= 2:
                previous_metrics = self.metrics_history[-2]["metrics"]
                f1_decrease = previous_metrics.get("f1", 0) - f1_score
                auc_decrease = previous_metrics.get("roc_auc", 0) - auc_score
                
                drift_details["f1_change"] = -f1_decrease
                drift_details["auc_change"] = -auc_decrease
                
                if f1_decrease > 0.05 or auc_decrease > 0.05:
                    drift_detected = True
                    drift_details["reason"] = "significant_decrease"
                    logger.warning(f"Significant performance decrease detected")
        
        return drift_detected, drift_details

    def check_for_data_drift(self) -> Tuple[Dict, bool]:
        """Check for data drift using monitor.py"""
        detector = DriftDetector(self.monitoring_config)
        return detector.analyze_drift(str(self.test_data_path))

    def save_version_artifacts(self, version_name: str):
        """Save artifacts for version control and comparison"""
        version_dir = self.versions_dir / version_name
        version_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy current model and data
        try:
            import shutil
            if self.model_path.exists():
                shutil.copy2(self.model_path, version_dir / self.model_path.name)
            if self.test_data_path.exists():
                shutil.copy2(self.test_data_path, version_dir / self.test_data_path.name)
            
            # Save metrics
            if self.metrics_history:
                with open(version_dir / "metrics.json", 'w') as f:
                    json.dump(self.metrics_history[-1], f, indent=2)
                    
            logger.info(f"Saved '{version_name}' artifacts to {version_dir}")
        except Exception as e:
            logger.error(f"Failed to save version artifacts: {str(e)}")

    def retrain_model(self) -> Tuple[str, Dict[str, Any]]:
        """Retrain the model and return the new model path and metrics"""
        try:
            # Save current version before retraining
            self.save_version_artifacts("before_drift")
            
            logger.info("Initiating model retraining...")
            new_model_path, new_test_data_path = train_model(str(self.processed_data_path))
            
            self.model_path = Path(new_model_path)
            self.test_data_path = Path(new_test_data_path)
            self.monitoring_config['model_path'] = str(new_model_path)
            
            new_metrics = self.evaluate_current_model()
            
            # Save after retraining
            self.save_version_artifacts("after_drift")
            
            retraining_report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "new_model_path": str(new_model_path),
                "new_test_data_path": str(new_test_data_path),
                "metrics": new_metrics,
                "monitoring_config": self.monitoring_config
            }
            
            logger.info(f"Retraining complete. New model saved to {new_model_path}")
            return new_model_path, retraining_report
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            raise

    def run_retraining_workflow(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Run the complete automated retraining workflow
        """
        workflow_result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "actions_taken": [],
            "monitoring_config": self.monitoring_config
        }
        
        try:
            # Step 1: Initial training if needed
            if self._initial_training_required():
                model_path, test_path = self.perform_initial_training()
                workflow_result.update({
                    "actions_taken": ["initial_training"],
                    "model_path": str(model_path),
                    "test_data_path": str(test_path),
                    "conclusion": "Initial training complete"
                })
                return workflow_result
            
            # Step 2: Evaluate current model
            current_metrics = self.evaluate_current_model()
            workflow_result.update({
                "current_metrics": current_metrics,
                "actions_taken": ["model_evaluation"]
            })
            
            # Step 3: Detect drift conditions
            performance_drift = False
            data_drift = False
            
            if not force_retrain:
                # Check performance drift
                performance_drift, drift_details = self.detect_performance_drift()
                workflow_result.update({
                    "performance_drift": {
                        "detected": performance_drift,
                        "details": drift_details
                    },
                    "actions_taken": workflow_result["actions_taken"] + ["performance_drift_detection"]
                })
                
                # Only check data drift if no performance drift
                if not performance_drift:
                    drift_results, data_drift = self.check_for_data_drift()
                    workflow_result.update({
                        "data_drift": {
                            "results": drift_results,
                            "detected": data_drift
                        },
                        "actions_taken": workflow_result["actions_taken"] + ["data_drift_detection"]
                    })
            
            # Decision to retrain
            should_retrain = force_retrain or performance_drift or data_drift
            
            if not should_retrain:
                workflow_result["conclusion"] = "No retraining needed"
                return workflow_result
            
            # Step 4: Optional Gemini analysis
            if self.gemini_api_key and not force_retrain:
                metrics_trend = [entry["metrics"] for entry in self.metrics_history[-5:]] if len(self.metrics_history) >= 5 else [entry["metrics"] for entry in self.metrics_history]
                gemini_analysis = self._analyze_with_gemini(
                    metrics_trend,
                    workflow_result.get("performance_drift", {}).get("details", {}) | 
                    workflow_result.get("data_drift", {}).get("results", {})
                )
                workflow_result.update({
                    "gemini_analysis": gemini_analysis,
                    "actions_taken": workflow_result["actions_taken"] + ["llm_analysis"]
                })
                
                if not gemini_analysis.get("retrain_recommended", True):
                    workflow_result["conclusion"] = "LLM advised against retraining"
                    return workflow_result
            
            # Step 5: Retrain model
            new_model_path, retraining_report = self.retrain_model()
            workflow_result.update({
                "retraining_report": retraining_report,
                "model_path": str(new_model_path),
                "test_data_path": str(retraining_report["new_test_data_path"]),
                "actions_taken": workflow_result["actions_taken"] + ["model_retraining"],
                "conclusion": "Retraining successful"
            })
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Retraining workflow failed: {str(e)}")
            return {
                **workflow_result,
                "error": str(e),
                "conclusion": "Failed"
            }


def run_automated_retraining(force_retrain: bool = False, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to run the complete retraining workflow
    """
    retrainer = ModelRetrainer(monitoring_config=config)
    return retrainer.run_retraining_workflow(force_retrain=force_retrain)


if __name__ == "__main__":
    force_retrain = os.getenv("FORCE_RETRAIN", "false").lower() == "true"
    result = run_automated_retraining(force_retrain=force_retrain)
    
    print("\nAutomated Retraining Workflow Summary:")
    print(f"Actions taken: {', '.join(result.get('actions_taken', []))}")
    print(f"Conclusion: {result.get('conclusion', 'Unknown')}")
    
    if "current_metrics" in result:
        print("\nCurrent Model Metrics:")
        for metric, value in result["current_metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    if "retraining_report" in result:
        print("\nNew Model Metrics:")
        for metric, value in result["retraining_report"]["metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    if "gemini_analysis" in result:
        print("\nGemini Analysis:")
        print(json.dumps(result["gemini_analysis"], indent=2))