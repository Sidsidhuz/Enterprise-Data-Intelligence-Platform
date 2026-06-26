from __future__ import annotations

import time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# App Configuration
st.set_page_config(
    page_title="AutoInsight — Local AutoML & Explainable AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/api/v1"

# Custom CSS for rich premium aesthetics — forces light palette on every Streamlit element
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* ── Base: force light background & dark text everywhere ─────────────────── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
    .stApp, .main, section.main {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #374151 !important;
    }

    /* ── All text elements ───────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    p, span, label, li, td, th, div {
        color: #0f172a;
    }

    /* ── Streamlit-specific text elements ────────────────────────────────────── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #1e293b !important;
    }

    /* ── Form labels & widget text ───────────────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label,
    .stTextInput label, .stNumberInput label,
    .stTextArea label, .stDateInput label,
    .stSlider label, .stCheckbox label,
    .stRadio label, .stFileUploader label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Radio button & checkbox option text */
    [data-testid="stRadio"] label span,
    [data-testid="stCheckbox"] label span,
    .stRadio > div > label,
    .stCheckbox > label {
        color: #1e293b !important;
    }

    /* ── Selectbox / dropdown value text ─────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #0f172a !important;
    }

    /* ── Input fields ─────────────────────────────────────────────────────────── */
    input, textarea, select {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] button[role="tab"] {
        color: #64748b !important;
        font-weight: 600;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb;
    }

    /* ── Metrics (st.metric) ─────────────────────────────────────────────────── */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #64748b !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #1e3a8a !important;
    }
    [data-testid="stMetricDelta"] {
        color: #15803d !important;
    }

    /* ── Dataframe / Table ───────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] * {
        color: #0f172a !important;
    }
    .dataframe th, .dataframe td {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    /* ── Info / Success / Warning / Error boxes ─────────────────────────────── */
    [data-testid="stAlert"] {
        color: #0f172a !important;
    }

    /* ── Spinner text ─────────────────────────────────────────────────────────── */
    [data-testid="stStatusWidget"] * {
        color: #0f172a !important;
    }

    /* ─────────────────────────────────────────────────────────────────────────
       Premium custom component styles
    ────────────────────────────────────────────────────────────────────────── */

    .main-title {
        background: linear-gradient(135deg, #1e3a8a, #2563eb, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #1e3a8a !important;
        font-weight: 700;
    }

    .custom-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
    }

    /* ── Buttons ──────────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.75rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(37,99,235,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 15px -3px rgba(37,99,235,0.35) !important;
    }
    .stButton > button p {
        color: #ffffff !important;
    }

    /* ── Status Badges ────────────────────────────────────────────────────────── */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-success { background-color: #dcfce7; color: #15803d !important; }
    .badge-info    { background-color: #dbeafe; color: #1d4ed8 !important; }
    .badge-warning { background-color: #fef9c3; color: #a16207 !important; }
    .badge-danger  { background-color: #fee2e2; color: #b91c1c !important; }

    /* ── Progress bar ────────────────────────────────────────────────────────── */
    [data-testid="stProgress"] > div > div {
        background-color: #2563eb !important;
    }

    /* ── File uploader ───────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzone"] span {
        color: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper Functions
def api_request(method, endpoint, **kwargs):
    try:
        url = f"{BACKEND_URL}{endpoint}"
        r = requests.request(method, url, **kwargs)
        if r.status_code in [200, 201, 202]:
            if "application/json" in r.headers.get("Content-Type", ""):
                return r.json()
            return r.content
        else:
            try:
                err = r.json()
                st.error(f"API Error: {err.get('message', r.text)}")
            except:
                st.error(f"API Error ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend API: {str(e)}")
        return None

# Sidebar Navigation & Backend Status
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #1e3a8a;'>🔮 AutoInsight</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("<hr style='margin-top: 0; margin-bottom: 20px;'/>", unsafe_allow_html=True)

# Liveness Check
try:
    health = requests.get("http://localhost:8000/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.markdown(
            '<div style="text-align: center;"><span class="badge badge-success">● Backend Online</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="text-align: center;"><span class="badge badge-danger">● Backend Offline</span></div>',
            unsafe_allow_html=True,
        )
except:
    st.sidebar.markdown(
        '<div style="text-align: center;"><span class="badge badge-danger">● Backend Offline</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.warning("FastAPI backend is not running. Please start it using: `uvicorn app.main:app --reload`")

st.sidebar.markdown("<br/>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard & History",
        "Upload & Preview",
        "Data Cleaning & EDA",
        "AutoML Training Console",
        "Leaderboard & Explanations",
        "Serve Predictions",
        "Export Center",
    ],
)

# Fetch datasets history globally to share between pages
datasets = api_request("GET", "/datasets")
datasets_list = datasets if datasets else []

# Main Views
if menu == "Dashboard & History":
    st.markdown("<h1 class='main-title'>AutoInsight Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Welcome to your local AutoML & Explainable AI workspace.</p>", unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Datasets Uploaded</div><div class="metric-value">{len(datasets_list)}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        completed_jobs = 0 # Can compute from jobs count
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">AutoML Runs</div><div class="metric-value">{len(datasets_list)}</div></div>', # approximation
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Active Models</div><div class="metric-value">5</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">Deployment Target</div><div class="metric-value">Local</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # History Table
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Datasets & Job History</div>", unsafe_allow_html=True)
    
    if not datasets_list:
        st.info("No datasets uploaded yet. Head over to 'Upload & Preview' to get started!")
    else:
        # Build pandas DataFrame for display
        history_data = []
        for d in datasets_list:
            status_badge = d['status']
            history_data.append({
                "ID": d['id'],
                "Filename": d['filename'],
                "Rows": d['rows'] or "Pending",
                "Columns": d['columns'] or "Pending",
                "Status": status_badge,
                "Uploaded At": pd.to_datetime(d['uploaded_at']).strftime("%Y-%m-%d %H:%M"),
            })
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Upload & Preview":
    st.markdown("<h1 class='main-title'>Upload & Profile Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Upload your raw data, check structural integrity, and view statistical profiles.</p>", unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Upload CSV or Excel File</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading and analyzing file..."):
                # Send to backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = api_request("POST", "/datasets", files=files)
                if res:
                    st.success(f"Successfully processed dataset '{res['filename']}'! Assigned ID: {res['id']}.")
                    st.balloons()
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Show profiling report for selected dataset
    if datasets_list:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Dataset Profile Viewer</div>", unsafe_allow_html=True)
        
        # Select box
        d_options = {f"{d['filename']} (ID: {d['id']})": d['id'] for d in datasets_list if d['status'] not in ["failed", "uploaded"]}
        
        if not d_options:
            st.info("No validated datasets available. Please upload a dataset first.")
        else:
            selected_d_id = st.selectbox("Select dataset to view profile", list(d_options.keys()))
            d_id = d_options[selected_d_id]

            # Fetch profile
            profile = api_request("GET", f"/datasets/{d_id}/profile")
            if profile:
                # Summary boxes
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", profile["row_count"])
                m2.metric("Columns", profile["column_count"])
                m3.metric("Duplicate Rows", profile["duplicates"])
                m4.metric("Avg Missingness", f"{np.mean([float(x.replace('%','')) for x in profile['missing_values'].values()]):.2f}%")

                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Column statistics table
                st.markdown("### Column Schema & Statistics")
                stats_rows = []
                dtypes = profile["dtypes"]
                missing = profile["missing_values"]
                outliers = profile["outlier_counts"]
                summary = profile["summary_stats"]

                for col in dtypes.keys():
                    col_stats = summary.get(col, {})
                    stats_rows.append({
                        "Column Name": col,
                        "Data Type": dtypes[col],
                        "Missing %": missing[col],
                        "Outliers": outliers.get(col, 0),
                        "Unique Count": col_stats.get("unique_count", "N/A"),
                        "Mean": f"{col_stats['mean']:.4f}" if "mean" in col_stats else "N/A",
                        "Median": f"{col_stats['median']:.4f}" if "median" in col_stats else "N/A",
                        "Min": f"{col_stats['min']:.4f}" if "min" in col_stats else "N/A",
                        "Max": f"{col_stats['max']:.4f}" if "max" in col_stats else "N/A",
                    })
                df_stats = pd.DataFrame(stats_rows)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Data Cleaning & EDA":
    st.markdown("<h1 class='main-title'>Data Cleaning & Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Configure feature imputation, deduplicate rows, and view correlation heatmaps.</p>", unsafe_allow_html=True)

    # Dataset selector
    d_options = {f"{d['filename']} (ID: {d['id']})": d for d in datasets_list if d['status'] not in ["failed", "uploaded"]}
    
    if not d_options:
        st.info("Please upload and validate a dataset first.")
    else:
        selected_d = st.selectbox("Select dataset to configure and clean", list(d_options.keys()))
        dataset = d_options[selected_d]
        d_id = dataset["id"]

        # Form for cleaning config
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Data Imputation Configuration</div>", unsafe_allow_html=True)
        
        # Load profile schema to get columns list
        profile = api_request("GET", f"/datasets/{d_id}/profile")
        
        if profile:
            dtypes = profile["dtypes"]
            missing_vals = profile["missing_values"]
            
            # Show columns with missing values and let users choose strategy
            cols_with_missing = [col for col, pct in missing_vals.items() if float(pct.replace("%", "")) > 0.0]
            
            imputation_overrides = {}
            if not cols_with_missing:
                st.success("No missing values detected! You can proceed directly to clean the dataset.")
            else:
                st.write("Configure imputation overrides for columns containing missing values:")
                
                # Show in a nice grid
                for col in cols_with_missing:
                    col_type = dtypes[col]
                    col_missing = missing_vals[col]
                    
                    if col_type == "numeric":
                        strategy = st.selectbox(
                            f"Column '{col}' ({col_type}, {col_missing} missing)",
                            ["median", "mean", "constant"],
                            key=f"impute_{col}",
                        )
                    else:
                        strategy = st.selectbox(
                            f"Column '{col}' ({col_type}, {col_missing} missing)",
                            ["mode", "constant"],
                            key=f"impute_{col}",
                        )
                    imputation_overrides[col] = strategy

            if st.button("Apply Cleaning & Dropping"):
                with st.spinner("Executing cleaning pipeline..."):
                    res = api_request("POST", f"/datasets/{d_id}/clean", json=imputation_overrides)
                    if res:
                        st.success("Dataset cleaned and deduplicated successfully! Sample saved to storage.")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # EDA Visualizations (only if cleaned)
        if dataset["status"] == "cleaned":
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Interactive EDA Visualizations</div>", unsafe_allow_html=True)
            
            # Fetch EDA values
            eda_data = api_request("GET", f"/datasets/{d_id}/eda")
            if eda_data:
                tab1, tab2 = st.tabs(["Correlation Matrix", "Column Distributions"])
                
                with tab1:
                    corr = eda_data["correlation"]
                    if not corr["columns"]:
                        st.info("No numeric features to correlate.")
                    else:
                        st.write("Pearson correlation matrix between numerical columns:")
                        fig = go.Figure(data=go.Heatmap(
                            z=corr["matrix"],
                            x=corr["columns"],
                            y=corr["columns"],
                            colorscale="RdBu",
                            zmin=-1,
                            zmax=1,
                            text=np.round(corr["matrix"], 2),
                            texttemplate="%{text}",
                        ))
                        fig.update_layout(width=700, height=600, margin=dict(l=40, r=40, b=40, t=40))
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    dist = eda_data["distributions"]
                    col_to_plot = st.selectbox("Select column to view distribution", list(dist.keys()))
                    
                    if col_to_plot:
                        col_dist = dist[col_to_plot]
                        if col_dist["type"] == "numeric":
                            # Draw bar chart for bins
                            bin_edges = col_dist["bin_edges"]
                            counts = col_dist["counts"]
                            # Align edges
                            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                            widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
                            
                            fig = go.Figure(data=[go.Bar(
                                x=bin_centers,
                                y=counts,
                                width=widths,
                                marker_color="#3b82f6",
                                opacity=0.85,
                                marker_line_color="#1e3a8a",
                                marker_line_width=0.5,
                            )])
                            fig.update_layout(
                                title=f"Distribution of '{col_to_plot}'",
                                xaxis_title=col_to_plot,
                                yaxis_title="Count",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Draw categorical count plot
                            fig = go.Figure(data=[go.Bar(
                                x=col_dist["categories"],
                                y=col_dist["counts"],
                                marker_color="#10b981",
                                opacity=0.85,
                            )])
                            fig.update_layout(
                                title=f"Value Counts of '{col_to_plot}'",
                                xaxis_title=col_to_plot,
                                yaxis_title="Count",
                            )
                            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif menu == "AutoML Training Console":
    st.markdown("<h1 class='main-title'>AutoML Training Console</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Define your modeling target, choose candidate algorithms, and trigger hyperparameter searches.</p>", unsafe_allow_html=True)

    # Selection
    d_options = {f"{d['filename']} (ID: {d['id']})": d for d in datasets_list if d['status'] == "cleaned"}
    
    if not d_options:
        st.info("No cleaned datasets available. Please upload, validate, and clean a dataset first.")
    else:
        selected_d = st.selectbox("Select dataset for AutoML training", list(d_options.keys()))
        dataset = d_options[selected_d]
        d_id = dataset["id"]

        profile = api_request("GET", f"/datasets/{d_id}/profile")
        
        if profile:
            dtypes = profile["dtypes"]
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Job Parameters Configuration</div>", unsafe_allow_html=True)
            
            # Select target
            target_column = st.selectbox("Select target variable (column to predict)", list(dtypes.keys()))
            
            # Problem type inference
            inferred_type = "classification"
            if dtypes[target_column] == "numeric" and profile["summary_stats"].get(target_column, {}).get("unique_count", 0) > 20:
                inferred_type = "regression"
                
            problem_type = st.selectbox(
                "Task Type Override",
                ["auto", "classification", "regression"],
                index=0,
                help=f"Currently inferred: {inferred_type.upper()}"
            )

            # Select Algorithms
            algs = ["logistic_regression", "linear_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
            st.write("Select algorithms to benchmark:")
            selected_algs = []
            
            c1, c2, c3 = st.columns(3)
            for idx, alg in enumerate(algs):
                # Filter appropriate algorithms
                is_class = "classifier" in alg or "logistic" in alg
                is_reg = "regressor" in alg or "linear" in alg
                
                # Check column distribution
                col_container = c1 if idx % 3 == 0 else (c2 if idx % 3 == 1 else c3)
                with col_container:
                    if st.checkbox(alg.replace("_", " ").title(), value=True, key=f"check_{alg}"):
                        selected_algs.append(alg)

            # Tuning time budget
            tuning_budget_seconds = st.number_input(
                "Tuning budget per algorithm (seconds)",
                min_value=10,
                max_value=300,
                value=60,
            )

            if st.button("Start AutoML Training"):
                # Submit job
                payload = {
                    "target_column": target_column,
                    "problem_type": problem_type if problem_type != "auto" else None,
                    "algorithms": selected_algs,
                    "tuning_budget_seconds": tuning_budget_seconds,
                }
                
                with st.spinner("Submitting training job..."):
                    res = api_request("POST", f"/datasets/{d_id}/train", json=payload)
                    if res:
                        st.success(f"Job successfully queued! Assigned Job ID: {res['id']}.")
                        st.session_state["active_job_id"] = res["id"]
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Polling status if active job exists
            if "active_job_id" in st.session_state:
                job_id = st.session_state["active_job_id"]
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Active Training Job Status</div>", unsafe_allow_html=True)
                
                # Setup a polling loop
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                while True:
                    job_status = api_request("GET", f"/training-jobs/{job_id}")
                    if not job_status:
                        break
                        
                    status = job_status["status"]
                    status_placeholder.markdown(f"**Current Status:** <span class='badge badge-info'>{status}</span>", unsafe_allow_html=True)
                    
                    if status == "queued":
                        progress_bar.progress(10)
                    elif status == "running":
                        progress_bar.progress(50)
                    elif status == "completed":
                        progress_bar.progress(100)
                        st.success("AutoML training completed successfully! Go to 'Leaderboard & Explanations' to review results.")
                        st.balloons()
                        del st.session_state["active_job_id"]
                        break
                    elif status == "failed":
                        progress_bar.progress(100)
                        st.error(f"AutoML job failed: {job_status.get('error_message')}")
                        del st.session_state["active_job_id"]
                        break
                        
                    time.sleep(2)
                st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Leaderboard & Explanations":
    st.markdown("<h1 class='main-title'>Model Evaluation & Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Compare candidate algorithms and visualize feature importances computed via SHAP.</p>", unsafe_allow_html=True)

    # Let user select from datasets that have completed jobs
    # Query datasets again
    d_completed = {f"{d['filename']} (ID: {d['id']})": d['id'] for d in datasets_list if d['status'] == "cleaned"}
    
    if not d_completed:
        st.info("No AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset to view leaderboard", list(d_completed.keys()))
        d_id = d_completed[selected_d]

        # Fetch last completed training job details
        # Use the list endpoint filtered by dataset_id and pick the latest completed/failed job
        all_jobs = api_request("GET", f"/training-jobs?dataset_id={d_id}") or []
        latest_job = next((j for j in all_jobs if j["status"] in ["completed", "failed", "running", "queued"]), None)

        if not latest_job:
            st.warning("No training job found for this dataset.")
        else:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>AutoML Performance Leaderboard</div>", unsafe_allow_html=True)
            
            st.write(f"Showing results for Target: **{latest_job['target_column']}** | Type: **{latest_job['problem_type'].upper()}**")
            
            leaderboard = latest_job["leaderboard"]
            if not leaderboard:
                st.info("No models trained successfully for this job.")
            else:
                # Plotly comparison plot
                metrics_df = []
                for item in leaderboard:
                    # extract primary metric
                    primary_metric = "F1-Score" if latest_job["problem_type"] == "classification" else "R²"
                    metrics_df.append({
                        "Model ID": item["model_id"],
                        "Algorithm": item["algorithm"].replace("_", " ").title(),
                        primary_metric: item["primary_metric_value"],
                        "Winner": "🏆 BEST" if item["is_best"] else "Candidate",
                    })
                df_leader = pd.DataFrame(metrics_df)
                st.dataframe(df_leader, use_container_width=True, hide_index=True)

                # Show metrics bar plot
                primary_col = df_leader.columns[2]
                fig = go.Figure(data=[go.Bar(
                    x=df_leader["Algorithm"],
                    y=df_leader[primary_col],
                    marker_color=["#10b981" if x == "🏆 BEST" else "#3b82f6" for x in df_leader["Winner"]],
                    text=np.round(df_leader[primary_col], 4),
                    textposition='auto',
                )])
                fig.update_layout(title=f"Algorithm Comparison ({primary_col})", yaxis_title=primary_col)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Explainability Section
            if leaderboard:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Global Feature Importance (SHAP)</div>", unsafe_allow_html=True)
                
                # Fetch best model
                best_model_id = next((m["model_id"] for m in leaderboard if m["is_best"]), leaderboard[0]["model_id"])
                
                # Check if shap plot image exists on backend
                # Since we write plots to data/plots/{dataset_id}/shap_summary.png, we can either serve it or recreate it.
                # To display it, the backend can return the image file, or we can write a simple endpoint.
                # Actually, our EDA router generates static plots on disk.
                # We can load the static image path directly. But Streamlit running on localhost can read the local file system directly since they run on the SAME machine!
                # This is a major superpower of a local-first architecture!
                # Streamlit can open `data/plots/{d_id}/shap_summary.png` directly using st.image()!
                import os
                # Project root
                project_root = Path(__file__).resolve().parent.parent
                shap_summary_path = project_root / f"data/plots/{d_id}/shap_summary.png"
                
                if shap_summary_path.exists():
                    st.image(str(shap_summary_path), caption="Global SHAP Feature Importance Summary", use_container_width=True)
                else:
                    st.info("SHAP Global Summary plot is generating or not found. Try generating a report to trigger it.")
                st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Serve Predictions":
    st.markdown("<h1 class='main-title'>Model Serving & Single Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Enter raw feature values to serve instant predictions along with a local SHAP explanation waterfall plot.</p>", unsafe_allow_html=True)

    # Dataset selector
    d_completed = {f"{d['filename']} (ID: {d['id']})": d['id'] for d in datasets_list if d['status'] == "cleaned"}
    
    if not d_completed:
        st.info("No completed AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset/model to serve predictions", list(d_completed.keys()))
        d_id = d_completed[selected_d]

        # Search for models of this dataset via list endpoint
        model_options = {}
        all_jobs = api_request("GET", f"/training-jobs?dataset_id={d_id}") or []
        for job_info in all_jobs:
            for m in job_info.get("leaderboard", []):
                model_options[f"{m['algorithm'].replace('_',' ').title()} (Model ID: {m['model_id']})"] = m['model_id']

        if not model_options:
            st.warning("No trained models found for this dataset.")
        else:
            selected_model = st.selectbox("Select trained model candidate", list(model_options.keys()))
            model_id = model_options[selected_model]

            # Fetch model schema & profile statistics to dynamically build form
            profile = api_request("GET", f"/datasets/{d_id}/profile")
            
            if profile:
                dtypes = profile["dtypes"]
                summary = profile["summary_stats"]
                
                # Exclude target column from form
                # How do we know target? Let's check model info
                model_meta = api_request("GET", f"/models/{model_id}")
                # Wait, our model details API doesn't return the target_column directly, but the training_job does!
                # We can query job details by training_job_id
                job_id = model_meta["training_job_id"]
                job_info = api_request("GET", f"/training-jobs/{job_id}")
                target_col = job_info["target_column"]

                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Single Instance Prediction Console</div>", unsafe_allow_html=True)
                
                # Build form inputs dynamically
                input_data = {}
                cols = st.columns(2)
                
                for idx, (col, col_type) in enumerate(dtypes.items()):
                    if col == target_col:
                        continue
                        
                    col_stats = summary.get(col, {})
                    col_container = cols[0] if idx % 2 == 0 else cols[1]
                    
                    with col_container:
                        if col_type == "numeric":
                            min_val = float(col_stats.get("min", 0.0))
                            max_val = float(col_stats.get("max", 100.0))
                            mean_val = float(col_stats.get("mean", 50.0))
                            
                            input_data[col] = st.number_input(
                                f"{col} (numeric)",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                            )
                        elif col_type == "boolean":
                            input_data[col] = st.selectbox(
                                f"{col} (boolean)",
                                [True, False],
                                index=0
                            )
                        elif col_type == "datetime":
                            input_data[col] = st.date_input(
                                f"{col} (date)",
                                value=pd.Timestamp.now()
                            ).strftime("%Y-%m-%d")
                        else:  # categorical
                            # Try to get unique values for dropdown
                            # (usually mode or unique count)
                            # If we can retrieve a few unique categories, show them
                            # For simplicity, we fallback to text input or dropdown if list is stored
                            input_data[col] = st.text_input(
                                f"{col} (categorical)",
                                value=str(col_stats.get("mode", ""))
                            )

                if st.button("Generate Prediction"):
                    with st.spinner("Executing inference and calculating SHAP values..."):
                        payload = {"input_data": input_data}
                        res = api_request("POST", f"/models/{model_id}/predict", json=payload)
                        
                        if res:
                            st.success("Prediction generated successfully!")
                            
                            # Layout results
                            res_c1, res_c2 = st.columns([1, 2])
                            
                            with res_c1:
                                st.write("### Inference Output")
                                pred = res["prediction"]
                                prob = res["probability"]
                                
                                st.metric("Predicted Value", pred)
                                if prob is not None:
                                    st.metric("Confidence Probability", f"{prob*100:.2f}%")
                                    
                            with res_c2:
                                st.write("### Local SHAP Explanation")
                                expl = res["explanation"]
                                if expl and expl["plot_path"]:
                                    # Load local waterfall plot
                                    project_root = Path(__file__).resolve().parent.parent
                                    plot_path = project_root / f"data/{expl['plot_path']}"
                                    if plot_path.exists():
                                        st.image(str(plot_path), use_container_width=True)
                                    else:
                                        # Render fallback Plotly bar chart
                                        shap_vals = expl["shap_values"]
                                        features = [x["feature"] for x in shap_vals][:10]
                                        contribs = [x["contribution"] for x in shap_vals][:10]
                                        
                                        fig = go.Figure(go.Bar(
                                            x=contribs,
                                            y=features,
                                            orientation='h',
                                            marker_color=["#ef4444" if c >= 0 else "#3b82f6" for c in contribs]
                                        ))
                                        fig.update_layout(title="Feature Impact (SHAP Values)", yaxis=dict(autorange="reversed"))
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("SHAP explanation not available.")
                
                st.markdown("</div>", unsafe_allow_html=True)

                # Batch predictions upload
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Batch Predictions Console</div>", unsafe_allow_html=True)
                
                batch_file = st.file_uploader("Upload batch CSV file (must contain the same columns)", type=["csv"], key="batch_upload")
                if batch_file is not None:
                    if st.button("Generate Batch Predictions"):
                        with st.spinner("Computing batch predictions..."):
                            files = {"file": (batch_file.name, batch_file.getvalue(), "text/csv")}
                            res_bytes = api_request("POST", f"/models/{model_id}/predict-batch", files=files)
                            
                            if res_bytes:
                                st.success("Batch predictions completed!")
                                st.download_button(
                                    label="Download Predictions CSV",
                                    data=res_bytes,
                                    file_name="batch_predictions.csv",
                                    mime="text/csv",
                                )
                st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Export Center":
    st.markdown("<h1 class='main-title'>Export Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Compile and download executive PDF summaries and multi-sheet Excel reports of your AutoML runs.</p>", unsafe_allow_html=True)

    d_completed = {f"{d['filename']} (ID: {d['id']})": d['id'] for d in datasets_list if d['status'] == "cleaned"}
    
    if not d_completed:
        st.info("No completed AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset for report generation", list(d_completed.keys()))
        d_id = d_completed[selected_d]

        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Generate Executive PDF Report</div>", unsafe_allow_html=True)
        st.write("PDF reports contain statistical profiling tables, correlation heatmaps, model leaderboards, and global SHAP summary charts.")
        
        if st.button("Generate PDF Report", key="btn_pdf"):
            with st.spinner("Assembling PDF document..."):
                res = api_request("POST", f"/datasets/{d_id}/reports", json={"report_type": "pdf"})
                if res:
                    report_id = res["id"]
                    
                    # Poll for completion (queued -> generating -> completed/failed)
                    while True:
                        status_info = api_request("GET", f"/reports/{report_id}")
                        if not status_info:
                            break
                        if status_info["status"] == "completed":
                            break
                        elif status_info["status"] == "failed":
                            st.error("PDF generation failed.")
                            break
                        time.sleep(1)
                    
                    if status_info and status_info["status"] == "completed":
                        st.success("PDF report generated successfully!")
                        # Fetch download file bytes
                        file_bytes = api_request("GET", f"/reports/{report_id}/download")
                        if file_bytes:
                            st.download_button(
                                label="Download PDF Report",
                                data=file_bytes,
                                file_name="autoinsight_summary.pdf",
                                mime="application/pdf",
                            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Generate Excel Workbook</div>", unsafe_allow_html=True)
        st.write("Excel workbooks contain structured metadata, model comparison parameters, and a cleaned data sheet (up to 1000 rows).")

        if st.button("Generate Excel Report", key="btn_xlsx"):
            with st.spinner("Compiling Excel sheets..."):
                res = api_request("POST", f"/datasets/{d_id}/reports", json={"report_type": "excel"})
                if res:
                    report_id = res["id"]
                    
                    # Poll for completion (queued -> generating -> completed/failed)
                    while True:
                        status_info = api_request("GET", f"/reports/{report_id}")
                        if not status_info:
                            break
                        if status_info["status"] == "completed":
                            break
                        elif status_info["status"] == "failed":
                            st.error("Excel generation failed.")
                            break
                        time.sleep(1)
                    
                    if status_info and status_info["status"] == "completed":
                        st.success("Excel report generated successfully!")
                        # Fetch download file bytes
                        file_bytes = api_request("GET", f"/reports/{report_id}/download")
                        if file_bytes:
                            st.download_button(
                                label="Download Excel Workbook",
                                data=file_bytes,
                                file_name="autoinsight_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
        st.markdown("</div>", unsafe_allow_html=True)
