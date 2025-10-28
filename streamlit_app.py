import streamlit as st



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

# Define the Model Structure and Descriptors
MODELS = {
    "AID-1030": {
        "title": "ALDH1A1 Inhibition",
        "mech": "Gradient Boosting",
        "icon": "🧪", # Test Tube for Inhibition
        "color": "#007AFF",
        "description": "Predicts inhibition of Aldehyde Dehydrogenase 1 Family Member A1 (ALDH1A1), a key enzyme in the AOP related to liver toxicity.",
        "features": 27,
        "descriptors": [f"Descriptor_{i}" for i in range(1, 28)],
    },
    "AID-504444": {
        "title": "Pulmonary Fibrosis",
        "mech": "Random Forest",
        "icon": "🫁", # Lungs for Pulmonary
        "color": "#5E5CE6",
        "description": "Screens for potential for Pulmonary Fibrosis, critical in understanding respiratory effects of PFAS exposure.",
        "features": 26,
        "descriptors": [f"Descriptor_{i}" for i in range(1, 27)],
    },
    "AID-588855": {
        "title": "Lung Cancer & Fibrosis",
        "mech": "Support Vector Machine",
        "icon": "🧬", # DNA/Cell for Cancer
        "color": "#FF9500", # Use a vibrant orange for tertiary
        "description": "Predicts activity associated with a broader Lung Cancer and Fibrosis pathway endpoint.",
        "features": 24,
        "descriptors": [f"Descriptor_{i}" for i in range(1, 25)],
    },
}

# Define the Template Data
TEMPLATE_COLUMNS = ["SMILES"] + MODELS["AID-1030"]["descriptors"] 
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
    
    # Ensure all required columns are present (Validation step)
    required_cols = MODELS[model_key]["descriptors"]
    if not all(col in data_df.columns for col in required_cols):
        raise ValueError("Input data is missing required molecular descriptor columns.")

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
    required_cols = MODELS[model_key]["descriptors"] + ["SMILES"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
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
                    {st.button("Get Started", key="get_started_btn", on_click=lambda: st.session_state.update(show_hero=False))}
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
        st.markdown(f'<div style="max-height: 250px; overflow-y: auto; padding-right: 15px; margin-bottom: 20px;">{descriptor_html}</div>', unsafe_allow_html=True)

        # Buttons
        col_c, col_d, _ = st.columns([1, 1, 4])
        
        with col_c:
            # Placeholder for Copy List
            st.button("📋 Copy List", help="This function is a placeholder for 'copy to clipboard'.", key="copy_list_btn", type="secondary")
        with col_d:
            download_template()
            
# Component 4 & 5: File Upload and Data Preview
def render_upload_and_preview():
    """Renders the drag-and-drop upload zone and the data preview table."""
    st.markdown("---")
    st.markdown("## Step 3: Upload Your Data")
    
    # Custom File Uploader rendering (using custom CSS class)
    st.markdown("""<div class="stFileUploader">
        <div style="color: #6E6E73; font-size: 18px;">
        <p style="margin-bottom: 16px;">
            <span style="font-size: 48px; color: #007AFF;">☁️</span><br>
            Drag and drop your file here<br>
            or click to browse
        </p>
        </div>
    </div>""", unsafe_allow_html=True)
    
    # We use the native file uploader hidden within the custom markdown block
    uploaded_file = st.file_uploader(
        "Upload",
        type=['csv', 'xlsx', 'txt'],
        accept_multiple_files=False,
        key="file_uploader",
        label_visibility="collapsed" # Hide the default Streamlit label
    )

    # File processing and validation logic
    df = None
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.prediction_results = None # Clear results on new upload
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                
                # Run Validation
                validation_status = validate_data(df, st.session_state.selected_model)
                
                if validation_status["errors"]:
                    st.error("❌ Data Validation Failed. Please fix the following errors:")
                    for error in validation_status["errors"]:
                        st.markdown(f'<p style="color: #FF3B30; margin-left: 15px;">{error}</p>', unsafe_allow_html=True)
                    st.session_state.input_data = None
                else:
                    st.session_state.input_data = df
                    st.success("✅ File successfully uploaded and validated!")
                    if validation_status["warnings"]:
                         st.warning("⚠️ Data Validation Warnings:")
                         for warning in validation_status["warnings"]:
                             st.markdown(f'<p style="color: #FF9500; margin-left: 15px;">{warning}</p>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ An error occurred during file reading or processing: {e}")
                st.session_state.input_data = None
        else:
            st.session_state.input_data = None


    # Data Preview
    if st.session_state.input_data is not None:
        st.markdown("---")
        st.markdown(f"## Data Preview: **{st.session_state.uploaded_file.name}**")
        st.markdown(f"<p>Rows: **{len(st.session_state.input_data)}** | Columns: **{len(st.session_state.input_data.columns)}**</p>", unsafe_allow_html=True)

        # Show only first 10 rows for clean preview
        st.dataframe(
            st.session_state.input_data.head(10).style.set_properties(**{'font-family': 'monospace', 'text-align': 'left'}),
            use_container_width=True,
            hide_index=True
        )
        if len(st.session_state.input_data) > 10:
            st.markdown(f"<p style='text-align: center; color: #6E6E73;'>... showing 10 of {len(st.session_state.input_data)} rows.</p>", unsafe_allow_html=True)

# Component 6: Prediction Button
def render_prediction_button():
    """Renders the main CTA to generate predictions."""
    st.markdown("---")
    st.markdown("## Step 4: Generate Predictions")

    predict_disabled = st.session_state.input_data is None or st.session_state.prediction_results is not None

    if st.session_state.input_data is None:
        st.info("Upload and validate your data in Step 3 to enable prediction.")
    elif st.session_state.prediction_results is not None:
        st.info("Predictions are already available below. Clear results or upload a new file to re-run.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Predictions", 
                     key="predict_btn", 
                     disabled=predict_disabled, 
                     use_container_width=True,
                     type="primary"):
            
            try:
                # Run the mock prediction function
                results_df = run_prediction_model(st.session_state.input_data, st.session_state.selected_model)
                st.session_state.prediction_results = results_df
                st.toast('Predictions successfully generated!', icon='🎉')
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.session_state.prediction_results = None

# Component 7: Results Display
def render_results_display():
    """Renders the summary cards and the detailed results table."""
    if st.session_state.prediction_results is None:
        return

    df_results = st.session_state.prediction_results
    total = len(df_results)
    active_count = df_results['Prediction'].sum()
    inactive_count = total - active_count

    st.markdown("---")
    st.markdown("## Prediction Results")
    
    # Summary Cards
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="Total Compounds", 
            value=total,
            delta="100%",
            delta_color="off"
        )
    with col_b:
        # Custom Metric Card for Active (1)
        st.markdown(f"""
        <div data-testid="stMetric" style="border-left: 4px solid #34C759;">
            <div style="font-size: 14px; color: #6E6E73; font-weight: 500;">Active (1)</div>
            <div style="font-size: 32px; font-weight: 600; color: #34C759; margin: 4px 0;">{active_count}</div>
            <div style="font-size: 14px; color: #34C759;">{active_count/total:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        # Custom Metric Card for Inactive (0)
        st.markdown(f"""
        <div data-testid="stMetric" style="border-left: 4px solid #FF3B30;">
            <div style="font-size: 14px; color: #6E6E73; font-weight: 500;">Inactive (0)</div>
            <div style="font-size: 32px; font-weight: 600; color: #FF3B30; margin: 4px 0;">{inactive_count}</div>
            <div style="font-size: 14px; color: #FF3B30;">{inactive_count/total:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)

    # Detailed Results Table
    st.markdown("### Detailed Prediction Table")
    
    # Custom display for Prediction column
    def format_prediction_label(val):
        color = '#34C759' if val == 'Active' else '#FF3B30'
        icon = '●' if val == 'Active' else '○'
        return f'<span style="color:{color}; font-weight: 600;">{icon} {val}</span>'

    # Filter columns for cleaner display: #, SMILES, Prediction
    display_df = df_results[['SMILES', 'Prediction_Label']].copy()
    display_df.rename(columns={'Prediction_Label': 'Prediction'}, inplace=True)
    
    # Add Index for display
    display_df.insert(0, '#', range(1, 1 + len(display_df)))

    # Apply custom HTML formatting for the prediction column
    # Note: Streamlit's native st.dataframe is typically better but for a custom look, we use HTML rendering here.
    st.markdown(
        display_df.head(20).style.format({'Prediction': format_prediction_label}).to_html(index=False), 
        unsafe_allow_html=True
    )
    if len(display_df) > 20:
        st.markdown(f"<p style='text-align: center; color: #6E6E73; margin-top: 15px;'>... showing 20 of {len(display_df)} results.</p>", unsafe_allow_html=True)

# Component 8: Download Section
def render_download_section():
    """Renders the section to export the results."""
    if st.session_state.prediction_results is None:
        return
    
    df_results = st.session_state.prediction_results
    st.markdown("---")
    st.markdown("## Export Results")
    st.markdown("<p>Choose your preferred format to download the complete results dataset, including the input data and the new **Prediction** column.</p>", unsafe_allow_html=True)

    
    # Filename based on selected model
    filename_base = f"PFAS_QSTR_{st.session_state.selected_model}_Results"
    
    # Create download links using the helper function (uses HTML buttons)
    excel_link = get_table_download_link(df_results, f"{filename_base}.xlsx", "📊 Excel (.xlsx)")
    csv_link = get_table_download_link(df_results, f"{filename_base}.csv", "📄 CSV (.csv)")
    txt_link = get_table_download_link(df_results, f"{filename_base}.txt", "📝 TXT (Tab-Delimited)")
    
    st.markdown(f"""
        <div style="margin-top: 24px;">
            {excel_link}
            {csv_link}
            {txt_link}
        </div>
    """, unsafe_allow_html=True)


# --- 5. MAIN APPLICATION EXECUTION ---

def main():
    apply_apple_style() # Apply the custom CSS at the very start
    
    # Fixed Header/Logo Section
    st.markdown(
        f'<div style="display: flex; align-items: center; padding-top: 18px;">'
        f'<h3 style="color: #007AFF; margin-right: 10px; margin-top: 0; font-weight: 700;">🫧</h3>'
        f'<h3 style="margin: 0; font-weight: 500;">PFAS Toxicity Classifier</h3>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    # Conditional Hero Section
    if st.session_state.show_hero:
        render_hero_section()
    else:
        # Main Steps
        st.markdown("<div id='main_content'>", unsafe_allow_html=True) # Anchor for smooth scrolling (not fully implemented in native Streamlit, but good practice)
        
        # Step 1
        render_model_selection()
        
        # Step 2
        render_descriptor_display()
        
        # Step 3 & Data Preview
        render_upload_and_preview()
        
        # Step 4 (Prediction)
        render_prediction_button()
        
        # Step 5 (Results)
        render_results_display()
        
        # Step 6 (Download)
        render_download_section()
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
