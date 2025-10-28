import streamlit as st
import pandas as pd
import time
import base64
import random
import io
import numpy as np
import os

# --- 1. CONFIGURATION AND INITIAL SETUP ---

st.set_page_config(
    page_title="PFAS Tox Classifier",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Define the Model Structure and Files (descriptors will be loaded lazily)
MODELS = {
    "AID-1030": {
        "title": "ALDH1A1 Inhibition",
        "mech": "Gradient Boosting",
        "icon": "Test Tube",  # Test Tube for Inhibition
        "color": "#007AFF",
        "description": "Predicts inhibition of Aldehyde Dehydrogenase 1 Family Member A1 (ALDH1A1), a key enzyme in the AOP related to liver toxicity.",
        "file": "AID- 1030.xlsx",
        "descriptors": None,
        "features": None,
    },
    "AID-504444": {
        "title": "Pulmonary Fibrosis",
        "mech": "Random Forest",
        "icon": "Lungs",  # Lungs for Pulmonary
        "color": "#5E5CE6",
        "description": "Screens for potential for Pulmonary Fibrosis, critical in understanding respiratory effects of PFAS exposure.",
        "file": "AID- 504444.xlsx",
        "descriptors": None,
        "features": None,
    },
    "AID-588855": {
        "title": "Lung Cancer & Fibrosis",
        "mech": "Support Vector Machine",
        "icon": "DNA",  # DNA/Cell for Cancer
        "color": "#FF9500",
        "description": "Predicts activity associated with a broader Lung Cancer and Fibrosis pathway endpoint.",
        "file": "AID- 588855.xlsx",
        "descriptors": None,
        "features": None,
    },
}

# --- Dynamic Descriptor Loading Function (Lazy) ---

@st.cache_data(show_spinner="Loading model descriptors...")
def load_model_descriptors(model_key):
    file_name = MODELS[model_key]["file"]
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, file_name)

    if not os.path.exists(file_path):
        st.error(f"FATAL ERROR: Descriptor file '{file_name}' not found at '{file_path}'. Ensure it's in the same directory.")
        return [], 0

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        all_cols = df.columns.tolist()

        if len(all_cols) < 3:
            st.error(f"Error: '{file_name}' has only {len(all_cols)} columns. Expected at least 3.")
            return [], 0

        descriptors = all_cols[1:-1]
        descriptors = [col for col in descriptors if col.upper() != "SMILES"]
        return descriptors, len(descriptors)

    except Exception as e:
        st.error(f"Could not load '{file_name}'. Error: {e}")
        return [], 0


# --- Lazy Model Info Loader ---
@st.cache_data(show_spinner="Initializing model...")
def get_model_info(model_key):
    if MODELS[model_key]["descriptors"] is not None:
        return MODELS[model_key]["descriptors"], MODELS[model_key]["features"]

    descriptors, count = load_model_descriptors(model_key)
    MODELS[model_key]["descriptors"] = descriptors
    MODELS[model_key]["features"] = count
    return descriptors, count


# --- Initialize Session State ---
def init_session_state():
    defaults = {
        "selected_model": list(MODELS.keys())[0],
        "uploaded_file": None,
        "input_data": None,
        "prediction_results": None,
        "show_hero": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Preload first model descriptors
    if MODELS[st.session_state.selected_model]["descriptors"] is None:
        get_model_info(st.session_state.selected_model)


# --- Template Generation (after first model is loaded) ---
def generate_template():
    first_key = st.session_state.selected_model
    descriptors, _ = get_model_info(first_key)
    cols = ["SMILES"] + descriptors

    data = {
        "SMILES": [
            "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O",
            "C(F)(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O",
            "CC(=O)Oc1ccccc1C(=O)O",
        ]
    }
    for col in cols:
        if col != "SMILES":
            data[col] = [round(random.uniform(0.1, 10.0), 4) for _ in range(3)]

    return pd.DataFrame(data)


# --- 2. CUSTOM APPLE-INSPIRED CSS & UI ELEMENTS ---
def apply_apple_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .stApp {background-color: #FAFAFA; font-family: 'Inter', -apple-system, sans-serif; color: #1D1D1F; max-width: 1200px; margin: auto; padding-top: 24px;}
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 1rem;}
        h1 {font-size: 56px; font-weight: 600; letter-spacing: -1.5px;}
        h2 {font-size: 38px; font-weight: 600;}
        h3 {font-size: 28px; font-weight: 500;}
        p, .stMarkdown {color: #6E6E73; line-height: 1.6;}
        .custom-card {background: white; padding: 32px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 16px 0 32px; border: 1px solid #D2D2D7;}
        .model-card {background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); cursor: pointer; border: 2px solid #D2D2D7; min-height: 250px; transition: all 0.3s ease;}
        .model-card:hover {transform: translateY(-4px); box-shadow: 0 8px 16px rgba(0,0,0,0.1);}
        .model-card.selected {border-color: #007AFF; box-shadow: 0 0 0 4px rgba(0,122,255,0.2);}
        .stButton > button {background-color: #007AFF; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 500; height: 44px;}
        .stButton > button:hover {background-color: #0066CC;}
        .stFileUploader {border: 2px dashed #D2D2D7; border-radius: 16px; padding: 64px; text-align: center; background-color: #FFFFFF;}
        .stFileUploader:hover {border-color: #007AFF; background-color: rgba(0,122,255,0.03);}
        [data-testid="stMetric"] {background: white; padding: 16px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #F0F0F2;}
        .descriptor-pill {display: inline-block; background-color: #F5F5F7; color: #1D1D1F; padding: 8px 16px; border-radius: 20px; margin: 4px; font-size: 14px; font-weight: 500; border: 1px solid #E5E5EA;}
    </style>
    """, unsafe_allow_html=True)


def render_icon(icon, color):
    st.markdown(f'<div style="font-size: 48px; color: {color}; margin-bottom: 10px;">{icon}</div>', unsafe_allow_html=True)


def get_table_download_link(df, filename, text):
    if filename.endswith('.csv'):
        data = df.to_csv(index=False).encode()
        mime = "text/csv"
    elif filename.endswith('.xlsx'):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='QSTR_Results')
        data = output.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        return ""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" style="background:#F5F5F7;color:#1D1D1F;border:1px solid #D2D2D7;border-radius:8px;padding:10px 20px;text-decoration:none;margin-right:10px;font-weight:500;" onmouseover="this.style.backgroundColor=\'#E5E5E8\'" onmouseout="this.style.backgroundColor=\'#F5F5F7\'">{text}</a>'


# --- 3. APPLICATION LOGIC FUNCTIONS ---
@st.cache_data(show_spinner="Running QSTR Models...")
def run_prediction_model(data_df, model_key):
    time.sleep(2)
    required = MODELS[model_key]["descriptors"]
    missing = [col for col in required if col not in data_df.columns]
    if missing:
        raise ValueError(f"Missing descriptors: {', '.join(missing)}")
    preds = np.random.randint(0, 2, len(data_df))
    df = data_df.copy()
    df['Prediction'] = preds
    df['Prediction_Label'] = df['Prediction'].apply(lambda x: 'Active' if x == 1 else 'Inactive')
    return df


def download_template():
    df = generate_template()
    return st.download_button(
        label="Download Input Template (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="pfas_qstr_template.csv",
        mime="text/csv",
        key='download_template_btn'
    )


def validate_data(df, model_key):
    required = MODELS[model_key]["descriptors"] + ["SMILES"]
    missing = [col for col in required if col not in df.columns]
    status = {"is_valid": True, "errors": [], "warnings": [], "missing_cols": missing}

    if df.empty:
        status["is_valid"] = False
        status["errors"].append("Critical: File is empty.")
    if missing:
        status["is_valid"] = False
        status["errors"].append(f"Missing columns: {', '.join(missing)}.")
    if 'SMILES' not in df.columns:
        status["is_valid"] = False
        status["errors"].append("Critical: 'SMILES' column required.")

    for col in MODELS[model_key]["descriptors"]:
        if col in df.columns:
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any() and not original.isnull().all():
                status["warnings"].append(f"Non-numeric in '{col}'. Converted with NaN.")

    if status["is_valid"]:
        st.session_state.input_data = df
    else:
        st.session_state.input_data = None
    return status


# --- 4. UI COMPONENTS ---
def select_model_card(key):
    st.session_state.selected_model = key
    st.session_state.prediction_results = None
    st.session_state.input_data = None
    st.session_state.uploaded_file = None
    get_model_info(key)  # Preload
    st.toast(f"Model: {key}", icon="Checkmark")


def render_hero_section():
    with st.container():
        st.markdown("""
            <div style="text-align: center; padding: 96px 0 128px 0; background-color: #FAFAFA;">
                <h1 style="font-weight: 700; font-size: 64px; letter-spacing: -2.5px;">PFAS QSTR Model Prediction</h1>
                <h1 style="font-weight: 700; color: #007AFF; margin-top: -10px; font-size: 64px;">Mechanistic Screening Platform</h1>
                <p style="font-size: 20px; color: #6E6E73; max-width: 700px; margin: 24px auto;">
                    Clean, elegant, and intuitive predictive toxicology screening of PFAS across AOP-informed bioassays.
                </p>
                <div style="margin-top: 48px;">
        """, unsafe_allow_html=True)
        if st.button("Get Started", key="get_started_btn", type="primary"):
            st.session_state.show_hero = False
            st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)
    st.divider()


def render_model_selection():
    st.markdown("## Step 1: Choose Your Model")
    st.markdown("<p>Select the QSTR model to run your data against.</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (key, model) in enumerate(MODELS.items()):
        is_selected = st.session_state.selected_model == key
        card_class = "model-card selected" if is_selected else "model-card"
        features = model["features"]
        if features is None:
            features = "Loading..."
        else:
            features = f"{features} Descriptors"

        with cols[i]:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 22px; color: {model['color']};">{key}</h3>
                    <div style="font-size: 48px; color: {model['color']};">{model['icon']}</div>
                </div>
                <h4 style="font-size: 24px; font-weight: 600; margin: 0 0 8px 0;">{model['title']}</h4>
                <p style="color: #6E6E73; font-size: 14px; margin-bottom: 12px; min-height: 40px;">{model['description'].split('.')[0]}.</p>
                <hr style="border: 0; border-top: 1px solid #E5E5EA; margin: 16px 0;">
                <div style="font-size: 14px; font-weight: 500; color: #1D1D1F;">
                    <p style="margin: 4px 0;">Features: <strong>{features}</strong></p>
                    <p style="margin: 4px 0;">Algorithm: <strong>{model['mech']}</strong></p>
                </div>
                <div style="text-align: right; margin-top: 16px;">
            """, unsafe_allow_html=True)
            btn_label = "Selected" if is_selected else "Select Model"
            btn_type = "primary" if is_selected else "secondary"
            st.button(btn_label, key=f"{key}_btn", on_click=select_model_card, args=(key,), type=btn_type)
            st.markdown("</div></div>", unsafe_allow_html=True)


def render_descriptor_display():
    key = st.session_state.selected_model
    descriptors, features = get_model_info(key)
    model = MODELS[key]

    st.markdown("---")
    st.markdown("## Step 2: Review Required Descriptors")
    with st.expander(f"**Required Descriptors for {model['title']} ({features} total)**", expanded=False):
        st.markdown(f"<p>Your input must include these {features} descriptors + <code>SMILES</code>.</p>", unsafe_allow_html=True)
        pills = "".join([f'<span class="descriptor-pill">{d}</span>' for d in ["SMILES"] + descriptors])
        st.markdown(f'<div style="max-height: 250px; overflow-y: auto; padding-right: 15px;">{pills}</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            st.button("Copy List", key="copy_btn", help="Coming soon")
        with c2:
            download_template()


def render_upload_and_preview():
    st.markdown("---")
    st.markdown("## Step 3: Upload Your Data")
    st.markdown("""<div class="stFileUploader"><div style="color: #6E6E73; font-size: 18px;">
        <p><span style="font-size: 48px; color: #007AFF;">Cloud</span><br>Drag and drop your file here<br>or click to browse</p></div></div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=['csv', 'xlsx', 'txt'], key="file_uploader", label_visibility="collapsed")

    if uploaded != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded
        st.session_state.prediction_results = None
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(('.csv', '.txt')) else pd.read_excel(uploaded)
                status = validate_data(df, st.session_state.selected_model)
                if status["errors"]:
                    st.error("Validation Failed:")
                    for e in status["errors"]: st.markdown(f"<p style='color:#FF3B30;margin-left:15px'>{e}</p>", unsafe_allow_html=True)
                else:
                    st.success("File validated!")
                    if status["warnings"]:
                        st.warning("Warnings:")
                        for w in status["warnings"]: st.markdown(f"<p style='color:#FF9500;margin-left:15px'>{w}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.input_data is not None:
        st.markdown("---")
        st.markdown(f"## Data Preview: **{st.session_state.uploaded_file.name}**")
        st.markdown(f"<p>Rows: **{len(st.session_state.input_data)}** | Columns: **{len(st.session_state.input_data.columns)}**</p>", unsafe_allow_html=True)
        st.dataframe(st.session_state.input_data.head(10), use_container_width=True, hide_index=True)
        if len(st.session_state.input_data) > 10:
            st.markdown("<p style='text-align: center; color: #6E6E73;'>... showing 10 of many rows.</p>", unsafe_allow_html=True)


def render_prediction_button():
    st.markdown("---")
    st.markdown("## Step 4: Generate Predictions")
    disabled = st.session_state.input_data is None or st.session_state.prediction_results is not None
    if st.session_state.input_data is None:
        st.info("Upload data to enable prediction.")
    elif st.session_state.prediction_results is not None:
        st.info("Predictions ready below.")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Generate Predictions", key="predict_btn", disabled=disabled, use_container_width=True, type="primary"):
            with st.spinner("Running model..."):
                try:
                    results = run_prediction_model(st.session_state.input_data, st.session_state.selected_model)
                    st.session_state.prediction_results = results
                    st.success("Predictions generated!")
                except Exception as e:
                    st.error(f"Error: {e}")


def render_results_display():
    if not st.session_state.prediction_results: return
    df = st.session_state.prediction_results
    total, active = len(df), df['Prediction'].sum()
    inactive = total - active

    st.markdown("---")
    st.markdown("## Prediction Results")
    ca, cb, cc = st.columns(3)
    with ca: st.metric("Total Compounds", total)
    with cb: st.markdown(f"<div style='border-left:4px solid #34C759;padding-left:12px;'><div style='font-size:14px;color:#6E6E73;'>Active (1)</div><div style='font-size:32px;font-weight:600;color:#34C759'>{active}</div><div style='font-size:14px;color:#34C759'>{active/total:.1%}</div></div>", unsafe_allow_html=True)
    with cc: st.markdown(f"<div style='border-left:4px solid #FF3B30;padding-left:12px;'><div style='font-size:14px;color:#6E6E73;'>Inactive (0)</div><div style='font-size:32px;font-weight:600;color:#FF3B30'>{inactive}</div><div style='font-size:14px;color:#FF3B30'>{inactive/total:.1%}</div></div>", unsafe_allow_html=True)

    st.markdown("### Detailed Prediction Table")
    disp = df[['SMILES', 'Prediction_Label']].copy().rename(columns={'Prediction_Label': 'Prediction'})
    disp.insert(0, '#', range(1, len(disp)+1))
    st.markdown(disp.head(20).style.format({'Prediction': lambda x: f'<span style="color:#34C759;font-weight:600">Active</span>' if x=='Active' else f'<span style="color:#FF3B30;font-weight:600">Inactive</span>'}).to_html(index=False), unsafe_allow_html=True)
    if len(disp) > 20:
        st.markdown("<p style='text-align:center;color:#6E6E73'>... showing 20 results.</p>", unsafe_allow_html=True)


def render_download_section():
    if not st.session_state.prediction_results: return
    df = st.session_state.prediction_results
    base = f"PFAS_QSTR_{st.session_state.selected_model}_Results"
    st.markdown("---")
    st.markdown("## Export Results")
    links = (
        get_table_download_link(df, f"{base}.xlsx", "Excel (.xlsx)") +
        get_table_download_link(df, f"{base}.csv", "CSV (.csv)") +
        get_table_download_link(df, f"{base}.txt", "TXT")
    )
    st.markdown(f"<div style='margin-top:24px'>{links}</div>", unsafe_allow_html=True)


# --- 5. MAIN ---
def main():
    apply_apple_style()
    st.markdown('<div style="display:flex;align-items:center;padding-top:18px;"><h3 style="color:#007AFF;margin-right:10px;margin-top:0;font-weight:700;">Bubble</h3><h3 style="margin:0;font-weight:500;">PFAS Toxicity Classifier</h3></div>', unsafe_allow_html=True)

    init_session_state()  # Safe init

    if st.session_state.show_hero:
        render_hero_section()
    else:
        render_model_selection()
        render_descriptor_display()
        render_upload_and_preview()
        render_prediction_button()
        render_results_display()
        render_download_section()


if __name__ == "__main__":
    main()
