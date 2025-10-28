import streamlit as st
import pandas as pd
import time
import base64
import random
import io
import numpy as np

# --- 1. CONFIGURATION AND INITIAL SETUP ---

# Page Configuration
st.set_page_config(
    page_title="PFAS Tox Classifier",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Define the Model Structure and Files
MODELS = {
    "AID- 1030": {
        "title": "ALDH1A1 Inhibition",
        "mech": "Gradient Boosting",
        "icon": "🧪", # Test Tube for Inhibition
        "color": "#007AFF",
        "description": "Predicts inhibition of Aldehyde Dehydrogenase 1 Family Member A1 (ALDH1A1), a key enzyme in the AOP related to liver toxicity.",
        "file": "AID-1030.xlsx", # File path for descriptor loading
    },
    "AID- 504444": {
        "title": "Pulmonary Fibrosis",
        "mech": "Random Forest",
        "icon": "🫁", # Lungs for Pulmonary
        "color": "#5E5CE6",
        "description": "Screens for potential for Pulmonary Fibrosis, critical in understanding respiratory effects of PFAS exposure.",
        "file": "AID-504444.xlsx", # File path for descriptor loading
    },
    "AID- 588855": {
        "title": "Lung Cancer & Fibrosis",
        "mech": "Support Vector Machine",
        "icon": "🧬", # DNA/Cell for Cancer
        "color": "#FF9500", # Use a vibrant orange for tertiary
        "description": "Predicts activity associated with a broader Lung Cancer and Fibrosis pathway endpoint.",
        "file": "AID-588855.xlsx", # File path for descriptor loading
    },
}

# --- Dynamic Descriptor Loading Function ---

@st.cache_data(show_spinner="Loading model descriptor definitions...")
def load_model_descriptors(model_key):
    """
    Reads the descriptor columns from the corresponding Excel file.
    Assumes: 1st column is Serial, Last column is Endpoint, columns in between are Descriptors.
    """
    file_path = MODELS[model_key]["file"]
    try:
        # We need openpyxl installed for pandas to read .xlsx
        df = pd.read_excel(file_path)
        
        all_cols = df.columns.tolist()
        
        # Check for minimum expected columns (Serial, at least one Descriptor, Endpoint)
        if len(all_cols) < 3: 
            st.error(f"Error: Model file '{file_path}' has only {len(all_cols)} columns. Expected at least 3 (Serial, Descriptor(s), Endpoint).")
            return [], 0
            
        # Descriptors are all columns from index 1 up to the second-to-last column
        descriptors = all_cols[1:-1]
        
        # Ensure 'SMILES' is not accidentally listed as a required descriptor if present in the model file
        descriptors = [col for col in descriptors if col.upper() != "SMILES"]

        return descriptors, len(descriptors)
        
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Descriptor file '{file_path}' not found. Please ensure it's in the same directory as this script.")
        return [], 0
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load descriptors from '{file_path}'. Check file format. Error: {e}")
        return [], 0

# --- Update MODELS dictionary with dynamic values ---
for key in MODELS:
    descriptors, features_count = load_model_descriptors(key)
    MODELS[key]["descriptors"] = descriptors
    MODELS[key]["features"] = features_count
    # Remove the file reference from cache/state if desired, but keeping it is fine.


# Define the Template Data (now using dynamically loaded descriptors)
first_model_key = list(MODELS.keys())[0]
TEMPLATE_COLUMNS = ["SMILES"] + MODELS[first_model_key]["descriptors"] 

TEMPLATE_DATA = {
    "SMILES": [
        "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O", # PFOA example
        "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O", # PFHxA example
        "CC(=O)Oc1ccccc1C(=O)O", # Non-PFAS example
    ]
}
# Populate template with dummy descriptor values
for col in TEMPLATE_COLUMNS:
    if col != "SMILES":
        TEMPLATE_DATA[col] = [round(random.uniform(0.1, 10.0), 4) for _ in range(len(TEMPLATE_DATA["SMILES"]))]

TEMPLATE_DF = pd.DataFrame(TEMPLATE_DATA)

# Initialize Session State
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(MODELS.keys())[0] # Default to the first one
# ... (rest of session state initialization)
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "show_hero" not in st.session_state:
    st.session_state.show_hero = True


# --- 2. CUSTOM APPLE-INSPIRED CSS & UI ELEMENTS ---

def apply_apple_style():
    """Applies custom CSS for an Apple-inspired look and feel."""
    st.markdown("""
    <style>
        /* Import Inter for SF Pro-like feel */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            background-color: #FAFAFA; /* Soft warm white */
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1D1D1F; /* Primary text */
            max-width: 1200px;
            margin: auto;
            padding-top: 24px;
        }

        /* Hide Streamlit branding/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        /* Typography */
        h1 {
            font-size: 56px;
            font-weight: 600;
            color: #1D1D1F;
            letter-spacing: -1.5px;
            margin-bottom: 0;
            line-height: 1.1;
        }
        
        h2 {
            font-size: 38px;
            font-weight: 600;
            color: #1D1D1F;
            letter-spacing: -0.5px;
            margin-top: 32px;
            margin-bottom: 16px;
        }
        
        h3 {
            font-size: 28px;
            font-weight: 500;
            color: #1D1D1F;
        }

        p, .stMarkdown {
            color: #6E6E73; /* Secondary text */
            line-height: 1.6;
        }

        /* Custom Card Styles (Main Container) */
        .custom-card {
            background: white;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Subtle shadow */
            margin: 16px 0 32px 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid #D2D2D7; /* Light border */
        }
        
        .custom-card:hover {
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.1);
        }

        /* Model Card Styles */
        .model-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            cursor: pointer;
            border: 2px solid #D2D2D7;
            transition: all 0.3s ease;
            min-height: 250px;
        }

        .model-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }

        .model-card.selected {
            border-color: #007AFF; /* Apple Blue border */
            box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.2);
        }

        /* Buttons (Primary CTA) */
        .stButton > button {
            background-color: #007AFF; /* Apple Blue */
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 500;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
            height: 44px;
        }
        
        .stButton > button:hover {
            background-color: #0066CC;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.4);
        }

        /* Secondary Button (Outlined/Ghost) */
        .secondary-button > button {
            background-color: white;
            color: #007AFF;
            border: 1px solid #007AFF;
            box-shadow: none;
        }

        .secondary-button > button:hover {
            background-color: rgba(0, 122, 255, 0.05);
            color: #0066CC;
            border-color: #0066CC;
        }
        
        /* File uploader (Step 3) */
        .stFileUploader {
            border: 2px dashed #D2D2D7;
            border-radius: 16px;
            padding: 64px;
            text-align: center;
            transition: all 0.3s ease;
            background-color: #FFFFFF;
        }
        
        .stFileUploader:hover {
            border-color: #007AFF;
            background-color: rgba(0, 122, 255, 0.03);
        }

        /* Metrics/Summary Cards */
        [data-testid="stMetric"] {
            background: white;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            border: 1px solid #F0F0F2;
        }

        /* Metric Value Colors */
        [data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 600;
            color: #1D1D1F;
        }

        /* Tags/Pills for Descriptors */
        .descriptor-pill {
            display: inline-block;
            background-color: #F5F5F7;
            color: #1D1D1F;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 4px;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid #E5E5EA;
        }
        
        /* Prediction Badges */
        .active-badge {
            color: #34C759; /* Apple Green */
            font-weight: 600;
        }
        .inactive-badge {
            color: #FF3B30; /* Apple Red */
            font-weight: 600;
        }

    </style>
    """, unsafe_allow_html=True)

# Helper function for rendering the icon
def render_icon(icon, color):
    """Renders a large, colored icon using HTML."""
    st.markdown(f'<div style="font-size: 48px; color: {color}; margin-bottom: 10px;">{icon}</div>', unsafe_allow_html=True)

# Helper function for downloading data
def get_table_download_link(df, filename, text):
    """Generates a link to download a CSV/Excel file."""
    if filename.endswith('.csv'):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        mime = "text/csv"
    elif filename.endswith('.xlsx'):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='QSTR_Results')
        b64 = base64.b64encode(output.getvalue()).decode()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        return "" # Unsupported format
    
    # Styled button HTML
    return f"""
    <a href="data:{mime};base64,{b64}" download="{filename}" style="
        background-color: #F5F5F7; 
        color: #1D1D1F; 
        border: 1px solid #D2D2D7;
        border-radius: 8px; 
        padding: 10px 20px; 
        text-decoration: none; 
        display: inline-flex; 
        align-items: center;
        margin-right: 10px;
        transition: all 0.3s ease;
        font-weight: 500;
    " onmouseover="this.style.backgroundColor='#E5E5E8';" onmouseout="this.style.backgroundColor='#F5F5F7';">
    {text}
    </a>
    """

# --- 3. APPLICATION LOGIC FUNCTIONS ---

@st.cache_data(show_spinner="Running QSTR Models on Data...")
def run_prediction_model(data_df, model_key):
    """Mocks the model prediction, returning dummy results."""
    time.sleep(2) # Simulate model computation time
    
    # Ensure all required descriptor columns are present (Validation step)
    required_descriptor_cols = MODELS[model_key]["descriptors"]
    
    # Check for missing descriptors
    missing_descriptors = [col for col in required_descriptor_cols if col not in data_df.columns]
    if missing_descriptors:
        raise ValueError(f"Input data is missing required molecular descriptor columns: {', '.join(missing_descriptors)}")

    # Create dummy prediction results
    compound_count = len(data_df)
    predictions = np.random.randint(0, 2, compound_count) # 0 or 1
    
    results_df = data_df.copy()
    results_df['Prediction'] = predictions
    results_df['Prediction_Label'] = results_df['Prediction'].apply(lambda x: 'Active' if x == 1 else 'Inactive')
    
    return results_df

# Template Download Function (using native Streamlit button)
def download_template():
    """Generates the example input template for download."""
    return st.download_button(
        label="📄 Download Input Template (CSV)",
        data=TEMPLATE_DF.to_csv(index=False).encode('utf-8'),
        file_name="pfas_qstr_template.csv",
        mime="text/csv",
        key='download_template_btn',
        help="Download a CSV file pre-formatted with all required descriptor columns."
    )

# Data Validation Function
def validate_data(df, model_key):
    """Performs validation checks on the uploaded DataFrame."""
    required_cols_for_input = MODELS[model_key]["descriptors"] + ["SMILES"]
    missing_cols = [col for col in required_cols_for_input if col not in df.columns]
    
    validation_status = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "missing_cols": missing_cols,
    }

    if df.empty:
        validation_status["is_valid"] = False
        validation_status["errors"].append("❌ Critical: Uploaded file is empty.")
    
    if missing_cols:
        validation_status["is_valid"] = False
        validation_status["errors"].append(
            f"❌ Missing Required Columns: The following {len(missing_cols)} columns are missing: {', '.join(missing_cols)}."
        )

    if 'SMILES' not in df.columns and not df.empty:
        validation_status["is_valid"] = False
        validation_status["errors"].append("❌ Critical: 'SMILES' column is required.")
        
    # Check for non-numeric descriptor columns (simple check and coercion attempt)
    numeric_cols = MODELS[model_key]["descriptors"]
    for col in numeric_cols:
        if col in df.columns:
             try:
                 # Attempt to coerce the column to numeric, ignoring non-numeric entries
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 if df[col].isnull().any():
                     validation_status["warnings"].append(
                         f"⚠️ Warning: Non-numeric data found in '{col}'. Null values introduced upon conversion."
                     )
             except Exception:
                validation_status["is_valid"] = False
                validation_status["errors"].append(
                    f"❌ Column '{col}' contains data that could not be converted to numeric."
                )
                
    # Update DataFrame in session state if valid
    if validation_status["is_valid"]:
        st.session_state.input_data = df
    else:
        st.session_state.input_data = None # Clear data if validation fails
        
    return validation_status

# --- 4. UI COMPONENTS (Streamlit Functions) ---

# Model Selection Handler
def select_model_card(key):
    """Sets the selected model and clears downstream data/results."""
    st.session_state.selected_model = key
    st.session_state.prediction_results = None
    st.session_state.input_data = None
    st.session_state.uploaded_file = None
    st.toast(f"Model set to {key}", icon="✅")


# Component 1: Hero Section
def render_hero_section():
    """Renders the large, confident hero section."""
    with st.container():
        st.markdown(f"""
            <div style="text-align: center; padding: 96px 0 128px 0; background-color: #FAFAFA;">
                <h1 style="font-weight: 700; font-size: 64px; letter-spacing: -2.5px;">PFAS QSTR Model Prediction</h1>
                <h1 style="font-weight: 700; color: #007AFF; margin-top: -10px; font-size: 64px; letter-spacing: -2.5px;">Mechanistic Screening Platform</h1>
                <p style="font-size: 20px; font-weight: 400; color: #6E6E73; margin-top: 24px; max-width: 700px; margin-left: auto; margin-right: auto;">
                    Clean, elegant, and intuitive predictive toxicology screening of Per- and Polyfluoroalkyl Substances (PFAS) across AOP-informed bioassays.
                </p>
                <div style="margin-top: 48px;">
                    {st.button("Get Started", 
                               key="get_started_btn", 
                               on_click=lambda: st.session_state.update(show_hero=False),
                               type="primary")}
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.divider()

# Component 2: Model Selection (Card-Based)
def render_model_selection():
    """Renders the three horizontally displayed model selection cards."""
    st.markdown("## Step 1: Choose Your Model")
    st.markdown("<p>Select the quantitative structure-activity relationship (QSTR) model you wish to run your data against.</p>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, (key, model) in enumerate(MODELS.items()):
        is_selected = st.session_state.selected_model == key
        card_class = "model-card selected" if is_selected else "model-card"
        
        with cols[i]:
            # The HTML div acts as the target for the click, then triggers the hidden button
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 22px; color: {model['color']};">{key}</h3>
                    {render_icon(model['icon'], model['color'])}
                </div>
                <h4 style="font-size: 24px; font-weight: 600; color: #1D1D1F; margin: 0 0 8px 0;">{model['title']}</h4>
                <p style="color: #6E6E73; font-size: 14px; margin-bottom: 12px; min-height: 40px;">{model['description'].split('.')[0]}.</p>
                
                <hr style="border: 0; border-top: 1px solid #E5E5EA; margin: 16px 0;">
                
                <div style="font-size: 14px; font-weight: 500; color: #1D1D1F;">
                    <p style="margin: 4px 0;">Features: <strong>{model['features']} Descriptors</strong></p>
                    <p style="margin: 4px 0;">Algorithm: <strong>{model['mech']}</strong></p>
                </div>
                
                <div style="text-align: right; margin-top: 16px;">
                    {st.button("Selected" if is_selected else "Select Model", 
                               key=f"{key}_button", 
                               on_click=select_model_card, 
                               args=(key,), 
                               type="primary" if is_selected else "secondary")}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Component 3: Dynamic Descriptor Display (Expandable)
def render_descriptor_display():
    """Renders the collapsible list of required molecular descriptors."""
    model = MODELS[st.session_state.selected_model]
    
    st.markdown("---")
    st.markdown("## Step 2: Review Required Descriptors")
    
    with st.expander(f"**Required Descriptors for {model['title']} ({model['features']} total)**", expanded=False):
        st.markdown(f"""
            <p style="margin-bottom: 24px;">Your input dataset must include the following **{model['features']}** molecular descriptor columns and a **SMILES** column for compound identification.</p>
        """, unsafe_allow_html=True)
        
        # Descriptor Pills
        descriptors = ["SMILES (Required)"] + model['descriptors']
        descriptor_html = "".join([f'<span class="descriptor-pill">{d}</span>' for d in descriptors])
        st.markd
