import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys
import json
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

import google.generativeai as genai
from streamlit.components.v1 import html
from dotenv import load_dotenv
# Configuration
MODEL_PATH = r"model\sepsis_xgboost_model.joblib"
PREPROCESSOR_PATH = r"model\preprocessor.joblib"
TEST_DATA_PATH = r"data\processed\test_data.npz"
HISTORY_PATH = r"data/patient_history.csv"
COLOR_SCHEME = {"low": "#4CAF50", "medium": "#FFC107", "high": "#F44336"}

# Initialize Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Sepsis Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "patient_history" not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        st.session_state.patient_history = pd.read_csv(HISTORY_PATH)
    else:
        st.session_state.patient_history = pd.DataFrame(columns=[
            "timestamp", "age", "gender", "heart_rate", "bp_systolic",
            "bp_diastolic", "temperature", "respiratory_rate", "wbc_count",
            "lactate_level", "comorbidities", "clinical_notes", 
            "prediction", "probability"
        ])

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(logo.png);
                background-repeat: no-repeat;
                background-size: 80%;
                background-position: 20px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_model_and_preprocessor():
    """Load model and preprocessor with error handling"""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Loading error: {str(e)}")
        return None, None

def load_test_data():
    """Load test data for evaluation metrics"""
    try:
        if not os.path.exists(TEST_DATA_PATH):
            st.warning(f"Test data not found at {TEST_DATA_PATH}")
            return None, None, None
            
        data = np.load(TEST_DATA_PATH, allow_pickle=True)
        X_test = data['X']
        y_test = data['y']
        
        if 'feature_names' in data:
            feature_names = list(data['feature_names'])
        else:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
            
        return X_test, y_test, feature_names
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None, None, None

def display_metrics():
    """Display model evaluation metrics"""
    X_test, y_test, _ = load_test_data()
    model, _ = load_model_and_preprocessor()
     
    if X_test is None or model is None:
        st.warning("Cannot display metrics without test data and model")
        return
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred)
    }
    
    # Display metrics in columns
    cols = st.columns(5)
    for i, (name, value) in enumerate(metrics.items()):
        cols[i].metric(name, f"{value:.3f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['No Sepsis', 'Sepsis'], 
           yticklabels=['No Sepsis', 'Sepsis'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    st.pyplot(fig)

def save_patient_case(data, prediction, probability):
    """Save patient case to history"""
    import datetime
    new_case = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data.iloc[0].to_dict(),
        "prediction": prediction,
        "probability": probability
    }
    
    hist_df = pd.concat([st.session_state.patient_history, pd.DataFrame([new_case])])
    hist_df.to_csv(HISTORY_PATH, index=False)
    st.session_state.patient_history = hist_df

def analyze_with_gemini(metrics_trend, drift_details):
    """Use Gemini to analyze performance trends"""
    if not GEMINI_API_KEY:
        return {"error": "API key missing"}
    
    # Convert Timestamps to strings in metrics_trend
    for entry in metrics_trend:
        if 'timestamp' in entry:
            entry['timestamp'] = str(entry['timestamp'])
    
    # Convert Timestamps in drift_details
    if 'timestamp' in drift_details:
        drift_details['timestamp'] = str(drift_details['timestamp'])
    
    prompt = f"""
    Analyze sepsis model performance:
    Recent metrics (newest last):
    {json.dumps(metrics_trend, indent=2, default=str)}
    
    Drift detection:
    {json.dumps(drift_details, indent=2, default=str)}
    
    Provide JSON analysis with:
    - analysis: Text analysis
    - retrain_recommended: Boolean
    - confidence: 0-1 score
    - improvements: List of suggestions
    - data_checks: List of checks
    
    Respond ONLY with valid JSON in this exact format:
    {{
        "analysis": "your analysis here",
        "retrain_recommended": true/false,
        "confidence": 0.XX,
        "improvements": ["suggestion1", "suggestion2"],
        "data_checks": ["check1", "check2"]
    }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        
        # Try to extract the JSON part from the response
        response_text = response.text
        
        # Sometimes the response might include markdown formatting
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
            
        return json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON response: {str(e)}",
            "raw_response": response.text if 'response' in locals() else None
        }
    except Exception as e:
        return {"error": str(e)}

def create_shap_force_plot(model, patient_data, feature_names):
    """Create SHAP force plot for individual prediction"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        patient_data[0],
        feature_names=feature_names,
        matplotlib=False,
        show=False
    )
    
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html

def display_prediction_result(prediction, probability):
    """Display prediction with color coding"""
    st.write("## Prediction Result")
    
    if probability >= 0.7:
        color = COLOR_SCHEME["high"]
        risk_level = "High Risk"
    elif probability >= 0.4:
        color = COLOR_SCHEME["medium"]
        risk_level = "Medium Risk"
    else:
        color = COLOR_SCHEME["low"]
        risk_level = "Low Risk"
    
    # Risk Header
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {color}20;
                border-left: 0.5rem solid {color}; margin: 1rem 0;">
        <h3 style="color: {color}; margin: 0;">{risk_level}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Probability: {probability:.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar
    st.markdown(f"""
    <div style="margin: 1rem 0; height: 20px; border-radius: 10px; background: #eee;">
        <div style="width: {probability:.0%}; height: 100%; border-radius: 10px;
                    background: {color}; transition: width 0.5s ease;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    return risk_level

def create_patient_form():
    """Enhanced patient input form with sections"""
    with st.form("patient_form"):
        # Personal Information
        with st.expander("**Patient Demographics**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 0, 120, 65)
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Vital Signs
        with st.expander("**Vital Signs**", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                heart_rate = st.number_input("Heart Rate (bpm)", 30, 250, 95)
            with col2:
                bp_systolic = st.number_input("Systolic BP (mmHg)", 50, 250, 120)
            with col3:
                bp_diastolic = st.number_input("Diastolic BP (mmHg)", 30, 150, 80)
            
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 38.2, 0.1)
            with col2:
                respiratory_rate = st.number_input("Respiratory Rate", 5, 60, 22)
        
        # Lab Results
        with st.expander("**Laboratory Results**"):
            col1, col2 = st.columns(2)
            with col1:
                wbc_count = st.number_input("WBC Count (cells/μL)", 1000, 50000, 13500)
            with col2:
                lactate_level = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 2.8, 0.1)
        
        # Medical History
        with st.expander("**Medical History**"):
            comorbidities = st.multiselect(
                "Comorbidities",
                ["Diabetes", "Hypertension", "COPD", "Cardiac Disease", 
                 "Immunocompromised", "Renal Disease", "None"],
                default=["None"]
            )
            clinical_notes = st.text_area(
                "Clinical Notes",
                "Patient presented with fever and elevated heart rate..."
            )
        
        if st.form_submit_button("Predict Sepsis Risk"):
            return pd.DataFrame([{
                "Age": age, "Gender": gender, "Heart_Rate": heart_rate,
                "BP_Systolic": bp_systolic, "BP_Diastolic": bp_diastolic,
                "Temperature": temperature, "Respiratory_Rate": respiratory_rate,
                "WBC_Count": wbc_count, "Lactate_Level": lactate_level,
                "Comorbidities": ", ".join(comorbidities),
                "Clinical_Notes": clinical_notes
            }])
    return None

def patient_history_page():
    """Display historical patient cases"""
    st.header("Patient History")
    
    if not st.session_state.patient_history.empty:
        df = st.session_state.patient_history.sort_values("timestamp", ascending=False)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.selectbox("Filter by Risk Level", 
                                     ["All", "High", "Medium", "Low"])
        with col2:
            date_filter = st.date_input("Filter by Date")
        
        # Apply filters
        if risk_filter != "All":
            df = df[df["prediction"] == (1 if risk_filter == "High" else 0)]
        if date_filter:
            df = df[pd.to_datetime(df["timestamp"]).dt.date == date_filter]
        
        # Display cases
        for _, row in df.iterrows():
            with st.container():
                risk_color = COLOR_SCHEME["high"] if row["prediction"] == 1 else \
                            COLOR_SCHEME["medium"] if row["probability"] >= 0.4 else \
                            COLOR_SCHEME["low"]
                
                st.markdown(f"""
                <div style="padding:1rem; margin:0.5rem 0; border-radius:0.5rem;
                            border-left:0.5rem solid {risk_color}; 
                            background:#f8f9fa;">
                    <div style="display:flex; justify-content:space-between;">
                        <h4>{row['timestamp']}</h4>
                        <div style="color:{risk_color}; font-weight:bold;">
                            {row['probability']:.1%} Risk
                        </div>
                    </div>
                    <p>Age: {row['age']} | Gender: {row['gender']}</p>
                    <p>WBC: {row['wbc_count']} | Lactate: {row['lactate_level']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No historical cases available")

def main():
    # Custom CSS
    st.markdown(f"""
    <style>
        .high-risk {{ color: {COLOR_SCHEME['high']} !important; }}
        .medium-risk {{ color: {COLOR_SCHEME['medium']} !important; }}
        .low-risk {{ color: {COLOR_SCHEME['low']} !important; }}
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "Prediction Tool": prediction_page,
        "Model Performance": performance_page,
        "Patient History": patient_history_page,
        "About": about_page
    }
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(pages.keys()))
    pages[page]()

def prediction_page():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.png", width=400)
    with col2:
        st.title("Sepsis Risk Assessment")
        if st.session_state.get("is_admin", True):  # You'd need to implement admin auth
            if st.sidebar.button("Admin: Retrain Model"):
                with st.spinner("Retraining model..."):
                    try:
                        from retrain import run_automated_retraining
                        result = run_automated_retraining(force_retrain=True)
                        
                        if "error" in result:
                            st.error(f"Retraining failed: {result['error']}")
                        else:
                            st.success("Model retrained successfully!")
                            st.session_state.model, st.session_state.preprocessor = load_model_and_preprocessor()
                    except Exception as e:
                        st.error(f"Error during retraining: {str(e)}")
        patient_data = create_patient_form()
        
        if patient_data is not None:
            model, preprocessor = load_model_and_preprocessor()
            if model and preprocessor:
                try:
                    # Preprocess and predict
                    processed_data = preprocessor.transform(patient_data)
                    probability = model.predict_proba(processed_data)[0][1]
                    prediction = int(probability >= 0.5)
                    
                    # Display results
                    risk_level = display_prediction_result(prediction, probability)
                    save_patient_case(patient_data, prediction, probability)
                    
                    
                    # Clinical Recommendations
                    st.header("Clinical Guidance")
                    if risk_level == "High Risk":
                        st.markdown(f"""
                        <div class="high-risk">
                            <h4>🚨 Immediate Actions Required:</h4>
                            <ul>
                                <li>Obtain blood cultures immediately</li>
                                <li>Administer broad-spectrum antibiotics within 1 hour</li>
                                <li>Initiate fluid resuscitation</li>
                                <li>Continuous vital sign monitoring</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == "Medium Risk":
                        st.markdown(f"""
                        <div class="medium-risk">
                            <h4>⚠️ Recommended Actions:</h4>
                            <ul>
                                <li>Repeat lactate measurement in 2 hours</li>
                                <li>Consider blood cultures</li>
                                <li>Reassess in 1 hour</li>
                                <li>Monitor urine output</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="low-risk">
                            <h4>✅ Monitoring Recommendations:</h4>
                            <ul>
                                <li>Continue routine monitoring</li>
                                <li>Reassess if condition changes</li>
                                <li>Educate patient on warning signs</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

from pathlib import Path  # Make sure this import is at the top

def load_metrics_history():
    """Load metrics history with proper timestamp parsing"""
    try:
        metrics_file = Path(r"model\metrics_history.json")
        if not metrics_file.exists():
            st.warning(f"Metrics history file not found at {metrics_file}")
            return None

        with open(metrics_file, "r") as f:
            raw_data = json.load(f)

        transformed = []
        for entry in raw_data:
            try:
                # Parse timestamp with microseconds support
                timestamp_str = entry.get("timestamp", "")
                if timestamp_str:
                    try:
                        timestamp = pd.to_datetime(timestamp_str, format='ISO8601')
                    except:
                        timestamp = pd.to_datetime(timestamp_str, errors='coerce')
                else:
                    timestamp = pd.NaT

                transformed.append({
                    "timestamp": timestamp,
                    **entry.get("metrics", {})
                })
            except Exception as e:
                st.error(f"Error processing metrics entry: {str(e)}")
                continue

        return pd.DataFrame(transformed).dropna(subset=['timestamp']) if transformed else None

    except Exception as e:
        st.error(f"Error loading metrics history: {str(e)}")
        return None

def load_drift_history():
    """Load drift history with proper timestamp parsing"""
    try:
        drift_file = Path("monitoring/drift_history.json")
        if not drift_file.exists():
            st.warning(f"Drift history file not found at {drift_file}")
            return None

        with open(drift_file, "r") as f:
            raw_data = json.load(f)

        transformed = []
        for entry in raw_data:
            try:
                # Parse timestamp with microseconds support
                timestamp_str = entry.get("timestamp", "")
                if timestamp_str:
                    try:
                        timestamp = pd.to_datetime(timestamp_str, format='ISO8601')
                    except:
                        timestamp = pd.to_datetime(timestamp_str, errors='coerce')
                else:
                    timestamp = pd.NaT

                # Safe nested dictionary access
                data_drift = entry.get("data_drift", {})
                methods = data_drift.get("methods", {})
                ks_test = methods.get("ks_test", {})
                evidently = methods.get("evidently", {})
                
                transformed.append({
                    "timestamp": timestamp,
                    "drift_detected": entry.get("drift_detected", False),
                    "retraining_recommended": entry.get("retraining_recommended", False),
                    "ks_test_detected": ks_test.get("detected", False),
                    "ks_test_score": list(ks_test.get("top_features", {}).values())[0] 
                                    if ks_test.get("top_features") else 0,
                    "evidently_drifted": evidently.get("metrics", {}).get("share_drifted", 0),
                    "concept_drift": entry.get("concept_drift", {}).get("detected", False)
                })
            except Exception as e:
                st.error(f"Error processing drift entry: {str(e)}")
                continue

        return pd.DataFrame(transformed).dropna(subset=['timestamp']) if transformed else None

    except Exception as e:
        st.error(f"Error loading drift history: {str(e)}")
        return None

def colored_metric(label, value, delta=None, color=None):
    """Custom metric with color support using Markdown"""
    if color:
        st.markdown(f"""
        <div style="border-left: 0.25rem solid {color}; padding-left: 1rem;">
            <div style="font-size: 0.8rem; color: #7f7f7f;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{value}</div>
            {f'<div style="font-size: 0.8rem;">{delta}</div>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.metric(label, value, delta)

def performance_page():
    st.title("Model Performance Monitoring")
    
    # Model Metrics
    st.header("Performance Metrics")
    display_metrics()
    
    # Drift Detection
    st.header("Data Drift Analysis")
    
    # Load historical data
    metrics_df = load_metrics_history()
    drift_df = load_drift_history()
    
    if metrics_df is None or drift_df is None:
        st.warning("Could not load monitoring data")
        return
    
    # Ensure proper datetime indexing
    metrics_df = metrics_df.set_index('timestamp').sort_index()
    drift_df = drift_df.set_index('timestamp').sort_index()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Trends", "Drift Analysis", "AI Recommendations"])
    
    with tab1:
        st.subheader("Model Performance Over Time")
        # Select metrics to display
        available_metrics = [col for col in metrics_df.columns 
                           if col not in ['timestamp']]
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            options=available_metrics,
            default=[col for col in ['f1', 'roc_auc'] if col in available_metrics]
        )
        
        if selected_metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            for metric in selected_metrics:
                ax.plot(metrics_df.index, metrics_df[metric], label=metric, marker='o')
            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Score")
            ax.set_title("Model Metrics Over Time")
            ax.grid(True)
            st.pyplot(fig)
        
        # Show metrics table
        st.dataframe(metrics_df.sort_index(ascending=False).head(10))

    with tab2:
        st.subheader("Data Drift Analysis")
        # Show current drift status
        latest_drift = drift_df.iloc[-1]
        cols = st.columns(3)
        
        # Use our custom colored_metric instead of st.metric
        with cols[0]:
            colored_metric(
                "Overall Drift Detected",
                "Yes" if latest_drift['drift_detected'] else "No",
                color="#F44336" if latest_drift['drift_detected'] else "#4CAF50"
            )
        
        with cols[1]:
            colored_metric(
                "KS Test Score",
                f"{latest_drift['ks_test_score']:.3f}",
                delta="Drift" if latest_drift['ks_test_detected'] else "No Drift",
                color="#F44336" if latest_drift['ks_test_detected'] else "#4CAF50"
            )
        
        with cols[2]:
            colored_metric(
                "Evidently Drift",
                f"{latest_drift['evidently_drifted']*100:.1f}%",
                delta="Drift" if latest_drift['evidently_drifted'] > 0.2 else "No Drift",
                color="#F44336" if latest_drift['evidently_drifted'] > 0.2 else "#4CAF50"
            )
            
        # Drift trend visualization
        st.subheader("Drift Scores Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(drift_df.index, drift_df['ks_test_score'], label='KS Test Score', marker='o')
        ax.plot(drift_df.index, drift_df['evidently_drifted'], label='Evidently Drift %', marker='o')
        ax.axhline(y=0.1, color='r', linestyle='--', label='KS Threshold')
        ax.axhline(y=0.2, color='g', linestyle='--', label='Evidently Threshold')
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Drift Score")
        ax.set_title("Drift Detection Over Time")
        ax.grid(True)
        st.pyplot(fig)
        
        # Show drift table
        st.dataframe(drift_df.sort_index(ascending=False).head(10))

    with tab3:
        st.subheader("AI-Powered Recommendations")
        if st.button("Generate Analysis"):
            with st.spinner("Analyzing with Gemini AI..."):
                # Prepare data for AI analysis
                analysis_data = {
                    "metrics_trend": metrics_df.tail(5).reset_index().to_dict('records'),
                    "drift_details": drift_df.iloc[-1].to_dict()
                }
                
                analysis = analyze_with_gemini(
                    metrics_trend=analysis_data["metrics_trend"],
                    drift_details=analysis_data["drift_details"]
                )
                
                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                else:
                    st.subheader("AI Analysis Report")
                    st.write(analysis.get("analysis", ""))
                    
                    cols = st.columns(2)
                    with cols[0]:
                        colored_metric(
                            "Retrain Recommended",
                            "Yes" if analysis.get("retrain_recommended") else "No",
                            color=COLOR_SCHEME["high"] if analysis.get("retrain_recommended") else COLOR_SCHEME["low"]
                        )
                    with cols[1]:
                        colored_metric(
                            "Confidence Level", 
                            f"{analysis.get('confidence', 0)*100:.1f}%",
                            color=COLOR_SCHEME["medium"]
                        )
                    st.subheader("Recommended Actions")
                    for imp in analysis.get("improvements", []):
                        st.write(f"✅ {imp}")
                    
                    st.subheader("Suggested Data Checks")
                    for check in analysis.get("data_checks", []):
                        st.write(f"🔍 {check}")

def about_page():
    st.title("About the System")
    st.markdown("""
                
    ## Sepsis Prediction System
    
    This clinical decision support tool uses machine learning to assess patients' risk of developing sepsis.
    
    **Key Features:**
    - Real-time risk assessment
    - Explainable AI with SHAP values
    - Patient history tracking
    - Model performance monitoring
    
    **Clinical Use:**
    - Early identification of at-risk patients
    - Evidence-based recommendations
    - Documentation support
    
    **Disclaimer:**
    This tool is intended to assist clinical decision making but should not replace professional judgment.
    """)

if __name__ == "__main__":
    main()