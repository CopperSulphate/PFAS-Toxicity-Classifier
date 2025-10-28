import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO, StringIO
from PIL import Image
import os
import plotly.express as px

# Apply custom CSS
st.markdown("""
<style>
    /* Import SF Pro-like font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 32px;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 16px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* Typography */
    h1 {
        font-size: 48px;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-size: 32px;
        font-weight: 600;
        color: #1D1D1F;
    }
    
    h3 {
        font-size: 24px;
        font-weight: 500;
        color: #1D1D1F;
    }
    
    p, .stMarkdown {
        color: #6E6E73;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #007AFF;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
    }
    
    .stButton > button:hover {
        background-color: #0051D5;
        transform: scale(1.02);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #D2D2D7;
        border-radius: 12px;
        padding: 48px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #007AFF;
        background-color: rgba(0, 122, 255, 0.05);
    }
    
    /* Tables */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background-color: #F5F5F7;
        color: #1D1D1F;
        font-weight: 600;
        border: none;
        padding: 16px;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #FAFAFA;
    }
    
    .dataframe tbody tr:hover {
        background-color: #F0F0F2;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Tags/Pills */
    .descriptor-pill {
        display: inline-block;
        background-color: #F5F5F7;
        color: #1D1D1F;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Success/Error states */
    .active-badge {
        color: #34C759;
        font-weight: 600;
    }
    
    .inactive-badge {
        color: #FF3B30;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Model configurations
models = {
    "aid_1030": {
        "name": "AID-1030",
        "description": "ALDH1A1 Inhibition",
        "features_count": 27,
        "type": "Gradient Boosting",
        "icon": "assets/icons/molecule.png",
        "training_file": "training_data/AID_1030_training.csv",
        "model_file": "models/aid_1030_gb_model.pkl",
        "bio_context": "AOP-informed mechanistic description for ALDH1A1 inhibition.",
        "metrics": {"Accuracy": 0.85, "Sensitivity": 0.82, "Specificity": 0.88},
        "training_size": 1000  # Placeholder
    },
    "aid_504444": {
        "name": "AID-504444",
        "description": "Pulmonary Fibrosis",
        "features_count": 26,
        "type": "Random Forest",
        "icon": "assets/icons/lung.png",
        "training_file": "training_data/AID_504444_training.csv",
        "model_file": "models/aid_504444_rf_model.pkl",
        "bio_context": "AOP-informed mechanistic description for pulmonary fibrosis.",
        "metrics": {"Accuracy": 0.87, "Sensitivity": 0.84, "Specificity": 0.90},
        "training_size": 1200  # Placeholder
    },
    "aid_588855": {
        "name": "AID-588855",
        "description": "Lung Cancer & Fibrosis",
        "features_count": 24,
        "type": "Support Vector",
        "icon": "assets/icons/dna.png",
        "training_file": "training_data/AID_588855_training.csv",
        "model_file": "models/aid_588855_svc_model.pkl",
        "bio_context": "AOP-informed mechanistic description for lung cancer and fibrosis.",
        "metrics": {"Accuracy": 0.89, "Sensitivity": 0.86, "Specificity": 0.92},
        "training_size": 1100  # Placeholder
    }
}

# Hero Section
st.markdown("<h1 style='text-align: center;'>PFAS QSTR Model Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 300;'>Mechanistic Screening Platform</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict molecular activities for PFAS compounds using advanced machine learning models.</p>", unsafe_allow_html=True)

# Step 1: Choose Your Model
st.markdown("### Step 1: Choose Your Model")
cols = st.columns(3)
for i, (key, model) in enumerate(models.items()):
    with cols[i]:
        st.markdown(f'<div class="custom-card" style="{"border: 2px solid #007AFF;" if st.session_state.selected_model == key else ""}">', unsafe_allow_html=True)
        if os.path.exists(model["icon"]):
            st.image(Image.open(model["icon"]), width=80)
        st.markdown(f"<h3>{model['name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{model['description']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>{model['features_count']} Features</p>", unsafe_allow_html=True)
        st.markdown(f"<p>{model['type']}</p>", unsafe_allow_html=True)
        if st.button("Select", key=f"select_{key}"):
            st.session_state.selected_model = key
            st.session_state.df = None
            st.session_state.predictions = None
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# If model selected, proceed
if st.session_state.selected_model:
    selected = models[st.session_state.selected_model]
    
    # Model Information Expander
    with st.expander("Model Information"):
        st.markdown(f"**Biological Context:** {selected['bio_context']}")
        st.markdown("**Performance Metrics:**")
        for k, v in selected["metrics"].items():
            st.markdown(f"- {k}: {v}")
        st.markdown(f"**Training Set Size:** {selected['training_size']} compounds")

    # Load features from training data (assume columns: SMILES, descriptors..., Activity)
    training_df = pd.read_csv(selected["training_file"])
    features = list(training_df.columns[1:-1])  # Exclude SMILES and Activity
    
    # Step 2: Review Required Descriptors
    st.markdown("### Step 2: Review Required Descriptors")
    with st.expander("Required Descriptors", expanded=True):
        st.markdown(f"Your dataset must include these {len(features)} molecular descriptors:")
        cols_desc = st.columns(3)
        for i, feat in enumerate(features):
            with cols_desc[i % 3]:
                st.markdown(f'<span class="descriptor-pill">{feat}</span>', unsafe_allow_html=True)
        if st.button("Copy List"):
            st.code(", ".join(features))
        template_df = pd.DataFrame(columns=["SMILES"] + features)
        st.download_button("Download Template", template_df.to_csv(index=False).encode('utf-8'), "example_input_template.csv", "text/csv")

    # Step 3: Upload Your Data
    st.markdown("### Step 3: Upload Your Data")
    uploaded_file = st.file_uploader("Drag and drop your file here or click to browse", type=["csv", "xlsx", "txt"], help="Supported: Excel, CSV, TXT. Maximum size: 200 MB")
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, sep="\t")
            else:
                st.error("Unsupported file format.")
                df = None
            
            if df is not None:
                # Validation
                missing_cols = [f for f in features if f not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                elif "SMILES" not in df.columns:
                    st.error("Missing 'SMILES' column.")
                else:
                    st.success("File validated successfully.")
                    st.session_state.df = df
                    # Data Preview
                    st.markdown(f"### Data Preview: {uploaded_file.name} (Rows: {len(df)})")
                    st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # Generate Predictions Button
    if st.session_state.df is not None:
        if st.button("Generate Predictions", key="predict_button"):
            with st.spinner("Processing predictions..."):
                try:
                    model = joblib.load(selected["model_file"])
                    X = st.session_state.df[features]
                    predictions = model.predict(X)
                    st.session_state.predictions = predictions
                    st.session_state.df['Prediction'] = predictions
                    st.success("Predictions generated successfully.")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    # Results
    if st.session_state.predictions is not None:
        df = st.session_state.df
        predictions = st.session_state.predictions
        total = len(df)
        active = np.sum(predictions == 1)
        inactive = total - active
        
        st.markdown("### Results")
        
        # Summary Cards
        cols_summary = st.columns(3)
        with cols_summary[0]:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"<h3>Total Compounds</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 32px; font-weight: 600;'>{total}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols_summary[1]:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"<h3>Active (1)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 32px; font-weight: 600; color: #34C759;'>{active}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>{active / total * 100:.1f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols_summary[2]:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"<h3>Inactive (0)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 32px; font-weight: 600; color: #FF3B30;'>{inactive}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>{inactive / total * 100:.1f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization (Optional)
        fig = px.pie(values=[active, inactive], names=['Active', 'Inactive'], color_discrete_sequence=['#34C759', '#FF3B30'],
                     title="Active/Inactive Distribution")
        st.plotly_chart(fig)
        
        # Results Table
        st.markdown("#### Prediction Results")
        def format_prediction(val):
            if val == 1:
                return '<span class="active-badge">● Active</span>'
            else:
                return '<span class="inactive-badge">○ Inactive</span>'
        df_display = df[['SMILES', 'Prediction']].copy()
        df_display['#'] = range(1, len(df) + 1)
        df_display = df_display[['#', 'SMILES', 'Prediction']]
        df_display['Prediction'] = df_display['Prediction'].apply(format_prediction)
        st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Download Section
        st.markdown("### Export Results")
        cols_download = st.columns(3)
        
        # CSV
        with cols_download[0]:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 CSV", csv, "results.csv", "text/csv")
        
        # Excel
        with cols_download[1]:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            data = output.getvalue()
            st.download_button("📊 Excel", data, "results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # TXT
        with cols_download[2]:
            txt = df.to_csv(index=False, sep="\t").encode('utf-8')
            st.download_button("📝 TXT", txt, "results.txt", "text/plain")
