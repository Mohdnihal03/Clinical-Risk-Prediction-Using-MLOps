import pytest
import os
import numpy as np
import tempfile
import joblib
from unittest.mock import patch, MagicMock

# Import the evaluate module directly - don't duplicate code
from src.evaluate import ModelEvaluator, evaluate_model
from src.pipeline import run_pipeline, get_default_config

# Create fixture for test data
@pytest.fixture
def sample_test_data():
    X_test = np.random.rand(100, 10)
    y_test = np.random.randint(0, 2, 100)
    feature_names = [f"feature_{i}" for i in range(10)]
    
    # Create a temporary file to save the test data
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f.name, X=X_test, y=y_test, feature_names=feature_names)
        test_data_path = f.name
    
    return test_data_path

# Create fixture for test model
@pytest.fixture
def sample_model():
    # Mock a simple model with predict and predict_proba methods
    model = MagicMock()
    model.predict.return_value = np.random.randint(0, 2, 100)
    model.predict_proba.return_value = np.random.rand(100, 2)
    model.n_estimators = 100
    
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(model, f.name)
        model_path = f.name
    
    return model_path

# Test the ModelEvaluator class
def test_model_evaluator_load_artifacts(sample_model, sample_test_data):
    evaluator = ModelEvaluator(sample_model, sample_test_data)
    evaluator.load_artifacts()
    
    # Check that data was loaded correctly
    assert evaluator.X_test is not None
    assert evaluator.y_test is not None
    assert evaluator.model is not None
    assert len(evaluator.feature_names) == evaluator.X_test.shape[1]

# Test the evaluate method with mocked MLflow
@patch('mlflow.start_run')
@patch('mlflow.log_params')
@patch('mlflow.log_metrics')
@patch('mlflow.log_figure')
def test_model_evaluator_evaluate(mock_log_figure, mock_log_metrics, 
                                 mock_log_params, mock_start_run,
                                 sample_model, sample_test_data):
    # Setup mock for mlflow.start_run context manager
    mock_start_run.return_value.__enter__.return_value = MagicMock()
    
    evaluator = ModelEvaluator(sample_model, sample_test_data)
    evaluator.load_artifacts()
    
    # Run the evaluation
    metrics = evaluator.evaluate()
    
    # Verify that MLflow was called correctly
    mock_start_run.assert_called_once()
    mock_log_params.assert_called_once()
    mock_log_metrics.assert_called_once()
    assert mock_log_figure.call_count >= 1
    
    # Verify that metrics were returned
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics

# Test the evaluate_model function
@patch('src.evaluate.ModelEvaluator')
def test_evaluate_model(mock_evaluator_class, sample_model, sample_test_data):
    # Setup the mock
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = {
        'accuracy': 0.85,
        'precision': 0.8,
        'recall': 0.75,
        'f1': 0.77,
        'roc_auc': 0.9
    }
    mock_evaluator_class.return_value = mock_evaluator
    
    # Call the function
    metrics = evaluate_model(sample_model, sample_test_data)
    
    # Verify that ModelEvaluator was used correctly
    mock_evaluator_class.assert_called_once_with(sample_model, sample_test_data)
    mock_evaluator.load_artifacts.assert_called_once()
    mock_evaluator.evaluate.assert_called_once()
    
    # Check the returned metrics
    assert metrics == mock_evaluator.evaluate.return_value

# Test the pipeline with mocked components
@patch('src.pipeline.CSVDataIngestor')
@patch('src.pipeline.ClinicalPreprocessor')
@patch('src.pipeline.train_model')
@patch('src.pipeline.evaluate_model')
def test_run_pipeline(mock_evaluate, mock_train, mock_preprocessor, 
                     mock_ingestor):
    # Setup mocks
    mock_ingestor.return_value.ingest_csv.return_value = {
        'status': 'success',
        'raw_data': 'data/raw/data.csv'
    }
    mock_preprocessor.return_value.preprocess.return_value = 'data/processed/data.npz'
    mock_train.return_value = ('model/model.pkl', 'data/processed/test_data.npz')
    mock_evaluate.return_value = {
        'accuracy': 0.85,
        'precision': 0.8,
        'recall': 0.75,
        'f1': 0.77,
        'roc_auc': 0.9
    }
    
    # Run the pipeline
    result = run_pipeline(input_file='data/raw/test.csv')
    
    # Verify the expected calls
    mock_ingestor.assert_called_once()
    mock_ingestor.return_value.ingest_csv.assert_called_once_with('data/raw/test.csv')
    mock_preprocessor.assert_called_once()
    mock_preprocessor.return_value.preprocess.assert_called_once()
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()
    
    # Check the pipeline result
    assert result['model_path'] == 'model/model.pkl'
    assert result['test_data_path'] == 'data/processed/test_data.npz'
    assert result['eval_results'] == mock_evaluate.return_value

# Clean up test files
def teardown_module(module):
    # Clean up any temporary files created during tests
    for file in os.listdir(tempfile.gettempdir()):
        if file.endswith('.npz') or file.endswith('.pkl'):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), file))
            except:
                pass