import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, lars_path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import project_utils
import os
import time

# --- Page Config & Custom Theme ---
st.set_page_config(page_title="Diabetes Data Story", layout="wide", page_icon="🧬")

# Custom CSS for "Expert" UI/UX
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #0E1117; /* Standard Streamlit Dark - Clean & Neutral */
        font-family: 'Inter', sans-serif;
    }

    /* Fix Top Header being white */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important; /* Pure White for Headings */
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, div, label, li, span {
        color: #E6E6E6 !important; /* High Contrast Off-White for Text */
        font-size: 1.05rem; /* Slightly larger for readability */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22; /* GitHub Dark Sidebar */
        border-right: 1px solid #30363D;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 20px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }

    .stMarkdown, .stPlotlyChart, .stMetric, .stDataFrame {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Clean Cards - No heavy glassmorphism */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #4A90E2 !important; /* Professional Soft Blue */
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] {
        background-color: #1F242D; /* Solid dark card background */
        border: 1px solid #30363D;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #4A90E2;
    }

    /* Custom Buttons - Clean & Visible */
    .stButton>button {
        background-color: #238636; /* GitHub Green - High Visibility */
        color: white !important;
        border: 1px solid rgba(240,246,252,0.1);
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #2EA043;
        border-color: #8B949E;
        transform: scale(1.02);
    }

</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
DATA_PATH = "Project_details/diabetes.data.txt"

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        return project_utils.load_data(DATA_PATH)
    return None

df = get_data()

if df is None:
    st.error("Data file not found. Please ensure 'diabetes.data.txt' is in 'Project_details/'.")
    st.stop()

target_col = "Y"
features = [c for c in df.columns if c != target_col]
X = df[features]
y = df[target_col]

# --- Navigation ---
st.sidebar.markdown("## 🧬 Data Story")
st.sidebar.markdown("Navigate the analysis chapters:")
chapter = st.sidebar.radio("", [
    "1. The Patient Data",
    "2. The Quest for Predictors",
    "3. The Algorithm Race",
    "4. The LARS Path"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Created By")
st.sidebar.markdown("**Rafiul Haider**")
st.sidebar.markdown("UID: U02002983")
st.sidebar.markdown("---")
st.sidebar.info("Designed with Streamlit & Plotly")

# --- Helper for Consistent Plot Styling ---
def style_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#C0C7D1"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Chapter 1: The Patient Data ---
if chapter == "1. The Patient Data":
    st.title("Chapter 1: The Patient Data")
    st.markdown("""
    <div style='font-size: 1.2rem; line-height: 1.6;'>
    Welcome. We are analyzing a dataset of <b>442 diabetes patients</b>. 
    Our goal is to understand the biological factors driving disease progression.
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Metrics with animation delay simulation
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df), delta="Dataset Size")
    col2.metric("Clinical Features", len(features), delta="Variables")
    col3.metric("Avg Progression", f"{y.mean():.1f}", delta="Target Mean")
    
    st.markdown("### 🧊 Multi-Dimensional Vitals")
    st.markdown("Interact with the data in 3D space. Rotate, zoom, and explore.")
    
    # Animated 3D Scatter (using size/color for dimensions)
    # We can animate by 'SEX' to show differences if we treat it as a frame, or just show a rich static 3D plot
    # Let's make it rich.
    
    fig_3d = px.scatter_3d(
        df, x='AGE', y='BMI', z='BP', color='Y',
        size='S6', # Adding another dimension (Blood Sugar) as size
        color_continuous_scale='Viridis',
        title="Patient Vitals: Age vs BMI vs BP (Size = Blood Sugar, Color = Progression)",
        opacity=0.8,
        hover_data=features
    )
    fig_3d = style_plot(fig_3d)
    fig_3d.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
    ))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("### 🔍 Correlation Matrix")
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig_corr = style_plot(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Chapter 2: The Quest for Predictors ---
elif chapter == "2. The Quest for Predictors":
    st.title("Chapter 2: The Quest for Predictors")
    st.markdown("""
    <div style='font-size: 1.2rem;'>
    To predict the future, we must find the strongest signals in the noise.
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Single Best Feature", "👯 Best Pair"])
    
    with tab1:
        if st.button("Reveal Best Feature", key="btn1"):
            with st.spinner("Analyzing correlations..."):
                time.sleep(0.5) # UX delay for effect
                best_feature, best_mse, model = project_utils.get_best_feature(df, target_col)
            
            st.success(f"**{best_feature}** is the single best predictor.")
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("MSE Score", f"{best_mse:.2f}")
                st.metric("Coefficient", f"{model.coef_[0]:.2f}")
            
            with col_b:
                fig = px.scatter(df, x=best_feature, y=target_col, color=target_col, 
                               color_continuous_scale='Inferno', title=f"{best_feature} vs Disease Progression")
                
                # Add regression line
                X_feat = df[[best_feature]]
                y_pred = model.predict(X_feat)
                fig.add_traces(go.Scatter(x=df[best_feature], y=y_pred, mode='lines', 
                                        name='Regression Line', line=dict(color='#00FF00', width=4)))
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.button("Reveal Best Pair", key="btn2"):
            with st.spinner("Testing combinations..."):
                time.sleep(0.5)
                best_pair, best_mse, model = project_utils.get_best_pair(df, target_col)
            
            st.success(f"The best duo is **{best_pair[0]}** and **{best_pair[1]}**.")
            
            st.markdown("#### 🌌 3D Regression Plane")
            
            x_range = np.linspace(df[best_pair[0]].min(), df[best_pair[0]].max(), 20)
            y_range = np.linspace(df[best_pair[1]].min(), df[best_pair[1]].max(), 20)
            xx, yy, zz = project_utils.get_regression_plane(model, x_range, y_range)
            
            fig_plane = go.Figure(data=[
                go.Scatter3d(x=df[best_pair[0]], y=df[best_pair[1]], z=df[target_col], 
                           mode='markers', marker=dict(size=4, color=df[target_col], colorscale='Viridis', opacity=0.8), 
                           name='Actual Data'),
                go.Surface(x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.4, name='Prediction Plane', showscale=False)
            ])
            fig_plane = style_plot(fig_plane)
            fig_plane.update_layout(
                title=f"Regression Plane: {best_pair[0]} & {best_pair[1]}",
                scene=dict(
                    xaxis_title=best_pair[0], 
                    yaxis_title=best_pair[1], 
                    zaxis_title='Progression',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                )
            )
            st.plotly_chart(fig_plane, use_container_width=True)

# --- Chapter 3: The Algorithm Race ---
elif chapter == "3. The Algorithm Race":
    st.title("Chapter 3: The Algorithm Race")
    st.markdown("### 🏎️ Linear Regression vs. XGBoost")
    
    col1, col2 = st.columns(2)
    
    # Run models
    lr_model = project_utils.train_linear_model(X, y)
    y_pred_lr = lr_model.predict(X)
    mse_lr = mean_squared_error(y, y_pred_lr)
    
    with col1:
        st.markdown("#### 🏛️ Linear Regression")
        st.metric("MSE", f"{mse_lr:.2f}")
        fig_lr = px.scatter(x=y, y=y_pred_lr, labels={'x': 'Actual', 'y': 'Predicted'}, 
                          color=np.abs(y - y_pred_lr), color_continuous_scale='Reds')
        fig_lr.add_shape(type="line", line=dict(dash='dash', color='white'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
        fig_lr = style_plot(fig_lr)
        fig_lr.update_layout(showlegend=False)
        st.plotly_chart(fig_lr, use_container_width=True)

    with col2:
        st.markdown("#### 🚀 XGBoost")
        if project_utils.XGBOOST_AVAILABLE:
            xgb_model = project_utils.train_xgboost(X, y)
            y_pred_xgb = xgb_model.predict(X)
            mse_xgb = mean_squared_error(y, y_pred_xgb)
            
            st.metric("MSE", f"{mse_xgb:.2f}", delta=f"{mse_lr - mse_xgb:.2f} improvement")
            
            fig_xgb = px.scatter(x=y, y=y_pred_xgb, labels={'x': 'Actual', 'y': 'Predicted'},
                               color=np.abs(y - y_pred_xgb), color_continuous_scale='Greens')
            fig_xgb.add_shape(type="line", line=dict(dash='dash', color='white'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            fig_xgb = style_plot(fig_xgb)
            fig_xgb.update_layout(showlegend=False)
            st.plotly_chart(fig_xgb, use_container_width=True)
        else:
            st.warning("XGBoost unavailable.")

    st.markdown("---")
    st.markdown("### 📈 Learning Curve Animation")
    st.markdown("Watch how the model learns as we feed it more data.")
    
    sizes = [20, 50, 100, 200]
    res = project_utils.calculate_mse_vs_sample_size(X, y, sizes)
    
    # Create an animated frame for learning curve? 
    # Hard to animate line chart growth in simple plotly express without frames structure.
    # Let's use a static but beautiful chart.
    
    fig_sample = go.Figure()
    fig_sample.add_trace(go.Scatter(x=res['Sample Size'], y=res['Training MSE'], mode='lines+markers', name='Training MSE', line=dict(color='#00ff88', width=3)))
    fig_sample.add_trace(go.Scatter(x=res['Sample Size'], y=res['Validation MSE'], mode='lines+markers', name='Validation MSE', line=dict(color='#ff0055', width=3)))
    fig_sample = style_plot(fig_sample)
    fig_sample.update_layout(title="Bias-Variance Tradeoff", xaxis_title="Sample Size", yaxis_title="MSE")
    st.plotly_chart(fig_sample, use_container_width=True)

# --- Chapter 4: The LARS Path ---
elif chapter == "4. The LARS Path":
    st.title("Chapter 4: The LARS Path")
    st.markdown("""
    ### 🛤️ The Path of Least Angle
    Visualizing the **LARS algorithm** (Efron et al., 2004).
    As we relax the regularization (moving right), variables enter the model one by one.
    """)
    
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean())
    
    # Use lars_path directly
    alphas, active, coefs = lars_path(X_std.values, y_std.values, method='lar')
    
    # Animated LARS Path
    # We can animate the "drawing" of the lines.
    
    fig_lars = go.Figure()
    
    # Professional Qualitative Colors (Plotly Safe)
    safe_colors = px.colors.qualitative.Safe
    
    # Add lines
    for i, feature in enumerate(features):
        color = safe_colors[i % len(safe_colors)]
        fig_lars.add_trace(go.Scatter(
            x=np.sum(np.abs(coefs.T), axis=1), 
            y=coefs[i], 
            mode='lines', 
            name=feature,
            line=dict(width=2, color=color),
            hovertemplate=f"<b>{feature}</b><br>Norm: %{{x:.2f}}<br>Coef: %{{y:.2f}}"
        ))
        
    fig_lars = style_plot(fig_lars)
    fig_lars.update_layout(
        title="LARS Path: Feature Entry Order",
        xaxis_title="L1 Norm (Model Complexity)",
        yaxis_title="Coefficient Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_lars, use_container_width=True)
    
    st.info("💡 **Insight**: The order in which lines diverge from zero represents the relative importance of features as determined by the LARS algorithm.")

    st.markdown("---")
    st.markdown("### 📚 Reference")
    st.markdown("""
    **"Least Angle Regression"**
    *   **Authors**: Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani.
    *   **Publication**: Annals of Statistics.
    *   **Volume/Issue/Pages**: Vol. 32, No. 2, 407–499.
    *   **Year**: 2004.
    *   **Link**: [Project Euclid](https://projecteuclid.org/euclid.aos/1083178935)
    """)
