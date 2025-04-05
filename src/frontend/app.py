import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys
import logging
from pathlib import Path
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Add the project directory to the Python path to enable imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import your custom modules (make sure these import statements match your project structure)
try:
    # This line assumes your modules are in a src folder
    sys.path.append('./src')
    from preprocess import ClinicalPreprocessor
    from train import ModelTrainer
except ImportError as e:
    st.error(f"Import error: {str(e)}. Make sure your project modules are properly set up.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sepsis Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Initialize paths
MODEL_PATH = r"model\sepsis_xgboost_model.joblib"
PREPROCESSOR_PATH = r"model\preprocessor.joblib"
TEST_DATA_PATH = r"data\processed\test_data.npz"

def load_model_and_preprocessor():
    """Load the trained model and preprocessor with error handling"""
    model, preprocessor = None, None
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        st.success("✅ Preprocessor loaded successfully")
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
    
    return model, preprocessor

def load_test_data():
    """Load the test data for metrics display"""
    try:
        if not os.path.exists(TEST_DATA_PATH):
            st.warning(f"Test data file not found at {TEST_DATA_PATH}")
            return None, None, None
            
        data = np.load(TEST_DATA_PATH, allow_pickle=True)
        X_test = data['X']
        y_test = data['y']
        
        if 'feature_names' in data:
            feature_names = list(data['feature_names'])
        else:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
            
        st.success("✅ Test data loaded successfully")
        return X_test, y_test, feature_names
    except Exception as e:
        st.warning(f"Error loading test data: {str(e)}")
        return None, None, None

def display_metrics():
    """Display model evaluation metrics"""
    try:
        # Load test data and model
        X_test, y_test, _ = load_test_data()
        model, _ = load_model_and_preprocessor()
        
        if X_test is None or model is None:
            st.warning("Cannot display metrics without test data and model")
            return
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall (Sensitivity)": round(recall_score(y_test, y_pred), 3),
            "F1 Score": round(f1_score(y_test, y_pred), 3),
            "ROC AUC": round(roc_auc_score(y_test, y_pred), 3)
        }
        
        # Create columns for metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['Recall (Sensitivity)']:.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_perc = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.matshow(conf_matrix_perc, cmap='Blues', alpha=0.7)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, 
                       s=f"{conf_matrix[i, j]}\n({conf_matrix_perc[i, j]:.1%})", 
                       va='center', ha='center')
                       
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Sepsis', 'Sepsis'])
        ax.set_yticklabels(['No Sepsis', 'Sepsis'])
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def display_feature_importance():
    """Display feature importance from SHAP values"""
    try:
        model, _ = load_model_and_preprocessor()
        X_test, _, feature_names = load_test_data()
        
        if model is None or X_test is None:
            st.warning("Cannot display feature importance without model and test data")
            return
            
        # Sample a subset of test data for SHAP (for performance)
        sample_size = min(100, X_test.shape[0])
        X_sample = X_test[:sample_size]
        
        # Generate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create and display SHAP summary plot
        st.write("### Feature Importance (SHAP)")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

def create_patient_form():
    """Create input form for patient data"""
    with st.form("patient_data_form"):
        st.write("### Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=65)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=250, value=95)
            bp_systolic = st.number_input("BP Systolic (mmHg)", min_value=50, max_value=250, value=120)
            bp_diastolic = st.number_input("BP Diastolic (mmHg)", min_value=30, max_value=150, value=80)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=38.2, step=0.1)
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=22)
            wbc_count = st.number_input("WBC Count (cells/μL)", min_value=1000, max_value=50000, value=13500)
            lactate_level = st.number_input("Lactate Level (mmol/L)", min_value=0.0, max_value=20.0, value=2.8, step=0.1)
        
        comorbidities = st.multiselect(
            "Comorbidities", 
            options=["Diabetes", "Hypertension", "COPD", "Cardiac Disease", "Immunocompromised", "Renal Disease", "None"],
            default=["None"]
        )
        
        clinical_notes = st.text_area(
            "Clinical Notes", 
            value="Patient presented with fever and elevated heart rate. No recent surgeries.",
            height=100
        )
        
        submitted = st.form_submit_button("Predict Sepsis Risk")
        
        if submitted:
            # Process comorbidities into string format
            comorbidities_str = ", ".join(comorbidities) if comorbidities and "None" not in comorbidities else "None"
            
            # Create dataframe with patient data
            patient_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Heart_Rate': [heart_rate],
                'BP_Systolic': [bp_systolic],
                'BP_Diastolic': [bp_diastolic],
                'Temperature': [temperature],
                'Respiratory_Rate': [respiratory_rate],
                'WBC_Count': [wbc_count],
                'Lactate_Level': [lactate_level],
                'Comorbidities': [comorbidities_str],
                'Clinical_Notes': [clinical_notes]
            })
            
            return patient_data
    
    return None

def load_css():
    """Load custom CSS to beautify the Streamlit app"""
    with open("src/frontend/styles.css", "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def make_prediction(patient_data):
    """Make sepsis prediction for patient data"""
    try:
        model, preprocessor = load_model_and_preprocessor()
        
        if model is None or preprocessor is None:
            st.error("Model or preprocessor not available")
            return
        
        # Preprocess the input data
        X_processed = preprocessor.transform(patient_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_processed)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction
        st.write("### Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk of Sepsis")
            else:
                st.success("✅ Low Risk of Sepsis")
        
        with col2:
            risk_percentage = prediction_proba * 100
            st.write(f"Sepsis Probability: {risk_percentage:.1f}%")
            
            # Create a progress bar for risk visualization
            st.progress(prediction_proba)
        
        # Risk factors
        if prediction_proba > 0.3:
            st.write("### Risk Factors")
            st.info(
                "Key factors that may be contributing to sepsis risk:\n"
                "- Elevated WBC count\n"
                "- Increased lactate levels\n"
                "- Elevated temperature\n"
                "- Increased heart rate"
            )
            
            st.write("### Recommended Actions")
            if prediction_proba > 0.7:
                st.warning(
                    "Urgent intervention recommended:\n"
                    "- Immediate blood cultures\n"
                    "- Administer broad-spectrum antibiotics\n"
                    "- Fluid resuscitation\n"
                    "- Monitor vital signs continuously"
                )
            else:
                st.info(
                    "Close monitoring recommended:\n"
                    "- Serial vital sign measurements\n"
                    "- Repeat laboratory tests in 4-6 hours\n"
                    "- Consider blood cultures if condition worsens"
                )
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def main():
    # Header
    load_css()
    st.title("🏥 Sepsis Prediction System")
    st.write("""
    This application helps healthcare providers assess a patient's risk of developing sepsis
    based on clinical data. Enter the patient's information below to get a prediction.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Tool", "Model Performance", "About"])
    
    st.sidebar.write("---")
    st.sidebar.write("### Model Information")
    st.sidebar.info("""
    This system uses an XGBoost classifier trained on clinical data
    to predict sepsis risk. The model evaluates patient vital signs,
    laboratory values, and clinical notes.
    """)
    
    # Main content based on selected page
    if page == "Prediction Tool":
        patient_data = create_patient_form()
        
        if patient_data is not None:
            make_prediction(patient_data)
            
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        st.write("""
        These metrics show how well the model performs on test data.
        Higher values indicate better performance.
        """)
        
        display_metrics()
        
        st.write("---")
        
        display_feature_importance()
        
    else:  # About page
        st.header("About the Sepsis Prediction System")
        st.write("""
        ### What is Sepsis?
        
        Sepsis is a life-threatening condition that occurs when the body's response to infection 
        causes injury to its own tissues and organs. If not recognized and treated promptly, 
        sepsis can lead to septic shock, multiple organ failure, and death.
        
        ### How Does This Tool Work?
        
        This tool uses machine learning to analyze patient data and identify patterns associated 
        with sepsis development. The model was trained on historical patient data where sepsis 
        outcomes were known. It considers vital signs, laboratory values, and clinical notes to 
        make predictions.
        
        ### Intended Use
        
        This tool is designed to assist healthcare providers in identifying patients at risk for 
        sepsis. It should be used as a supportive tool alongside clinical judgment and established 
        sepsis protocols, not as a replacement for clinical decision-making.
        """)

if __name__ == "__main__":
    main()