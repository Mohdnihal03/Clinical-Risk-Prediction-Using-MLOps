import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os
import json
import datetime
import requests
from typing import Dict, Any, Tuple, List, Optional

# Import our existing modules
from train import train_model
from evaluate import evaluate_model

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        processed_data_path: str,
        gemini_api_key: Optional[str] = None,
        performance_threshold: float = 0.75,
        metrics_history_path: str = "model/metrics_history.json"
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            logger.warning("No Gemini API key provided. LLM-based analysis will be unavailable.")
        
        self.performance_threshold = performance_threshold
        self.metrics_history_path = Path(metrics_history_path)
        self.metrics_history = self._load_metrics_history()
        
    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical metrics data from JSON file"""
        if not self.metrics_history_path.exists():
            logger.info(f"No metrics history found at {self.metrics_history_path}. Starting fresh.")
            return []
        
        try:
            with open(self.metrics_history_path, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded metrics history with {len(history)} entries")
            return history
        except Exception as e:
            logger.error(f"Failed to load metrics history: {str(e)}")
            return []
    
    def _save_metrics_history(self):
        """Save updated metrics history to JSON file"""
        try:
            # Create directory if it doesn't exist
            self.metrics_history_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(self.metrics_history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Saved metrics history to {self.metrics_history_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {str(e)}")
    
    def evaluate_current_model(self) -> Dict[str, float]:
        """Run evaluation on current model and return metrics"""
        try:
            metrics = evaluate_model(str(self.model_path), str(self.test_data_path))
            
            # Add timestamp and save to history
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
            logger.info("Not enough history to detect drift. Need at least 2 evaluations.")
            return False, {"reason": "insufficient_history"}
        
        # Get latest metrics
        current_metrics = self.metrics_history[-1]["metrics"]
        
        # Check against threshold
        f1_score = current_metrics.get("f1", 0)
        auc_score = current_metrics.get("roc_auc", 0)
        
        drift_details = {
            "current_f1": f1_score,
            "current_auc": auc_score,
            "threshold": self.performance_threshold,
            "previous_evaluations": len(self.metrics_history) - 1
        }
        
        # Simple threshold-based detection
        drift_detected = (f1_score < self.performance_threshold or auc_score < self.performance_threshold)
        
        if drift_detected:
            drift_details["reason"] = "below_threshold"
            logger.warning(f"Performance drift detected! Metrics below threshold: F1={f1_score}, AUC={auc_score}")
        else:
            # Check for significant decrease from previous run
            if len(self.metrics_history) >= 2:
                previous_metrics = self.metrics_history[-2]["metrics"]
                f1_decrease = previous_metrics.get("f1", 0) - f1_score
                auc_decrease = previous_metrics.get("roc_auc", 0) - auc_score
                
                drift_details["f1_change"] = -f1_decrease
                drift_details["auc_change"] = -auc_decrease
                
                # Detect significant decrease (more than 5%)
                if f1_decrease > 0.05 or auc_decrease > 0.05:
                    drift_detected = True
                    drift_details["reason"] = "significant_decrease"
                    logger.warning(f"Performance drift detected! Significant decrease from previous run: F1 change={-f1_decrease}, AUC change={-auc_decrease}")
        
        return drift_detected, drift_details
    
    def analyze_drift_with_llm(self, drift_details: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini (LLM) to analyze drift and recommend actions"""
        if not self.gemini_api_key:
            logger.warning("Cannot analyze with LLM: No Gemini API key provided")
            return {"error": "No Gemini API key provided"}
        
        try:
            # Prepare context for the LLM
            metrics_trend = [entry["metrics"] for entry in self.metrics_history[-5:]] if len(self.metrics_history) >= 5 else [entry["metrics"] for entry in self.metrics_history]
            
            prompt = f"""
            I'm monitoring a sepsis prediction ML model and need your analysis of its performance metrics.
            
            Recent metrics trend (newest last):
            {json.dumps(metrics_trend, indent=2)}
            
            Current drift details:
            {json.dumps(drift_details, indent=2)}
            
            Based on this information:
            1. Analyze the possible causes of any performance degradation
            2. Recommend whether retraining is necessary
            3. Suggest specific improvements for the next model iteration
            4. Recommend any data quality checks that should be performed
            
            Return your analysis as structured JSON with the following fields:
            - analysis: Your analysis of the current situation
            - retrain_recommendation: boolean indicating if retraining is recommended
            - confidence: numerical confidence (0-1) in your recommendation
            - improvement_suggestions: list of specific improvements to try
            - data_quality_checks: list of recommended data quality checks
            """
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            # Parse and validate the response
            try:
                analysis_result = json.loads(response)
                # Ensure required fields are present
                required_fields = ["analysis", "retrain_recommendation", "confidence"]
                for field in required_fields:
                    if field not in analysis_result:
                        analysis_result[field] = None
                return analysis_result
            except json.JSONDecodeError:
                logger.error("LLM returned non-JSON response")
                return {
                    "analysis": "Failed to parse LLM response",
                    "retrain_recommendation": True,  # Default to retraining on error
                    "confidence": 0.5,
                    "raw_response": response
                }
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {"error": str(e), "retrain_recommendation": True}
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API and return the response"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from Gemini response
            if "candidates" in result and len(result["candidates"]) > 0:
                if "content" in result["candidates"][0]:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        return content["parts"][0]["text"]
            
            raise ValueError("Unexpected response structure from Gemini API")
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def retrain_model(self) -> Tuple[str, Dict[str, Any]]:
        """Retrain the model and return the new model path and metrics"""
        try:
            # Run training process using our existing training module
            logger.info("Initiating model retraining...")
            new_model_path, new_test_data_path = train_model(str(self.processed_data_path))
            
            # Update paths to new model and test data
            self.model_path = Path(new_model_path)
            self.test_data_path = Path(new_test_data_path)
            
            # Evaluate new model
            new_metrics = self.evaluate_current_model()
            
            # Create retraining report
            retraining_report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "new_model_path": str(new_model_path),
                "new_test_data_path": str(new_test_data_path),
                "metrics": new_metrics
            }
            
            logger.info(f"Retraining complete. New model saved to {new_model_path}")
            return new_model_path, retraining_report
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            raise
    
    def run_retraining_workflow(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Run the complete automated retraining workflow
        
        Args:
            force_retrain: If True, skip drift detection and retrain anyway
            
        Returns:
            Dict containing workflow results and actions taken
        """
        try:
            workflow_result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "actions_taken": []
            }
            
            # Step 1: Evaluate current model
            logger.info("Step 1: Evaluating current model performance")
            current_metrics = self.evaluate_current_model()
            workflow_result["current_metrics"] = current_metrics
            workflow_result["actions_taken"].append("model_evaluation")
            
            # Step 2: Detect drift (unless force_retrain is True)
            if not force_retrain:
                logger.info("Step 2: Detecting performance drift")
                drift_detected, drift_details = self.detect_performance_drift()
                workflow_result["drift_detected"] = drift_detected
                workflow_result["drift_details"] = drift_details
                workflow_result["actions_taken"].append("drift_detection")
                
                # Skip to LLM analysis if drift is detected
                if not drift_detected:
                    logger.info("No performance drift detected. Skipping retraining.")
                    workflow_result["conclusion"] = "No retraining needed"
                    return workflow_result
            else:
                logger.info("Force retrain flag set. Skipping drift detection.")
                workflow_result["drift_detected"] = True
                workflow_result["drift_details"] = {"reason": "force_retrain"}
            
            # Step 3: LLM Analysis of drift (if API key available)
            if self.gemini_api_key:
                logger.info("Step 3: Analyzing drift with LLM")
                llm_analysis = self.analyze_drift_with_llm(
                    workflow_result.get("drift_details", {"reason": "force_retrain"})
                )
                workflow_result["llm_analysis"] = llm_analysis
                workflow_result["actions_taken"].append("llm_analysis")
                
                # Check if LLM recommends retraining
                should_retrain = llm_analysis.get("retrain_recommendation", True)
                if not should_retrain and not force_retrain:
                    logger.info("LLM does not recommend retraining at this time.")
                    workflow_result["conclusion"] = "LLM advised against retraining"
                    return workflow_result
            
            # Step 4: Retrain model
            logger.info("Step 4: Retraining model")
            new_model_path, retraining_report = self.retrain_model()
            workflow_result["retraining_report"] = retraining_report
            workflow_result["actions_taken"].append("model_retraining")
            
            # Step 5: Log completion and save workflow result
            logger.info(f"Retraining workflow completed successfully. New model: {new_model_path}")
            workflow_result["conclusion"] = "Retraining successful"
            
            # Save workflow result to file
            workflow_path = Path("model/retraining_workflows")
            workflow_path.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = workflow_path / f"workflow_{timestamp}.json"
            
            with open(result_file, 'w') as f:
                json.dump(workflow_result, f, indent=2)
            
            logger.info(f"Workflow result saved to {result_file}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Retraining workflow failed: {str(e)}")
            workflow_result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e),
                "actions_taken": workflow_result.get("actions_taken", []),
                "conclusion": "Failed"
            }
            return workflow_result


def run_automated_retraining(
    model_path: str,
    test_data_path: str,
    processed_data_path: str,
    gemini_api_key: Optional[str] = None,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run the complete retraining workflow
    
    Args:
        model_path: Path to the current model file
        test_data_path: Path to the test data
        processed_data_path: Path to the processed training data
        gemini_api_key: API key for Gemini (optional)
        force_retrain: Force retraining regardless of drift detection
        
    Returns:
        Dict containing workflow results
    """
    retrainer = ModelRetrainer(
        model_path=model_path,
        test_data_path=test_data_path,
        processed_data_path=processed_data_path,
        gemini_api_key=gemini_api_key
    )
    
    return retrainer.run_retraining_workflow(force_retrain=force_retrain)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated model retraining with drift detection")
    parser.add_argument("--model-path", required=True, help="Path to the current model file")
    parser.add_argument("--test-data-path", required=True, help="Path to the test data")
    parser.add_argument("--processed-data-path", required=True, help="Path to the processed training data")
    parser.add_argument("--gemini-api-key", help="API key for Gemini LLM (can also use GEMINI_API_KEY env var)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining regardless of drift detection")
    
    args = parser.parse_args()
    
    result = run_automated_retraining(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        processed_data_path=args.processed_data_path,
        gemini_api_key=args.gemini_api_key,
        force_retrain=args.force_retrain
    )
    
    # Print summary of results
    print("\nAutomated Retraining Workflow Summary:")
    print(f"Actions taken: {', '.join(result.get('actions_taken', []))}")
    print(f"Conclusion: {result.get('conclusion', 'Unknown')}")
    
    if "current_metrics" in result:
        print("\nCurrent Model Metrics:")
        for metric, value in result["current_metrics"].items():
            print(f"  {metric}: {value}")
    
    if "retraining_report" in result and "metrics" in result["retraining_report"]:
        print("\nNew Model Metrics:")
        for metric, value in result["retraining_report"]["metrics"].items():
            print(f"  {metric}: {value}")