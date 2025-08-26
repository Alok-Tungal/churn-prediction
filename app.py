import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np  
import sys 
 
# Patch custom functions if needed
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout settings
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model, scaler, and model_columns from the pickle file
@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, model_columns = pickle.load(f)
    return model, scaler, model_columns

# Load everything
data = load_data()
model, scaler, model_columns = load_model()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ================== 🏠 MAIN PAGE: CHURN PREDICTION ==================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Enter important customer details to predict churn likelihood.")

    # Use only most relevant features
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (months)', 0, 100, 12)
        monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
        total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

    # Build user input
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        f'Contract_{contract}': 1,
        f'PaymentMethod_{payment}': 1,
        f'InternetService_{internet}': 1,
    }

    # Convert to DataFrame and fill missing model columns
    user_df = pd.DataFrame([input_dict])
    for col in model_columns:
        if col not in user_df.columns:
            user_df[col] = 0  # fill others with 0
    user_df = user_df[model_columns]  # ensure correct order

    # Scale and predict
    if st.button("🔍 Predict Churn"):
        try:
            input_scaled = scaler.transform(user_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

            # Show Feature Importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Feature Importance (Top 5)")
                feat_df = pd.DataFrame({
                    'feature': model_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(5)
                fig, ax = plt.subplots()
                ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# ================== 📈 INSIGHTS ==================
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Insights & Visualizations")

    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(ax.containers[0], fmt='%.1f%%')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(ax.containers[0], fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("### 🧠 Key Business Insights")
    st.markdown("""
    - Month-to-month contracts show the highest churn.
    - Electronic checks are most churn-prone.
    - Short-tenure and high-monthly-charge customers are likely to churn.
    """)

# ================== 📄 RAW DATA ==================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")



app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="📡 Telecom Churn Insights", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Small CSS for nicer look
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #FBFBFD; }
    header {background: linear-gradient(90deg,#0ea5e9,#7c3aed); padding: 12px 20px;}
    header .css-1v3fvcr { color: white; font-size: 20px; font-weight:700; }
    .metric { border-radius: 12px; padding: 8px; }
    .kpi { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_df(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def simple_preprocess(df, numeric_features=None):
    df_proc = df.copy()
    # Drop obvious id columns
    for c in df_proc.columns:
        if c.lower() in ("customerid","id","cust_id"):
            df_proc = df_proc.drop(columns=c)
    # Infer numeric features if not provided
    if numeric_features is None:
        numeric_features = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    # Fill numeric missing
    imputer = SimpleImputer(strategy="median")
    df_proc[numeric_features] = imputer.fit_transform(df_proc[numeric_features])
    # Label encode categoricals
    cat_cols = df_proc.select_dtypes(include=["object","category"]).columns.tolist()
    le_map = {}
    for c in cat_cols:
        df_proc[c] = df_proc[c].fillna("UNKNOWN")
        le = LabelEncoder()
        df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        le_map[c] = le
    return df_proc, numeric_features, le_map

@st.cache_data
def train_quick_model(df, target_col="Churn"):
    # simple model training with common numeric features
    label_map = None
    df2 = df.copy()
    if target_col not in df2.columns:
        return None, None
    # convert churn to binary if needed
    if df2[target_col].dtype == object:
        df2[target_col] = df2[target_col].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
    y = df2[target_col]
    # choose numeric features or create them
    X = df2.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    if X.shape[1] < 1:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf, X.columns.tolist()

def model_predict_proba(model, model_features, input_df):
    # Ensure columns align
    X = input_df.copy()
    missing = [c for c in model_features if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0
    X = X[model_features]
    probs = model.predict_proba(X)[:, 1]
    return probs

# -------------------------
# Sidebar - Navigation
# -------------------------
st.sidebar.image("https://i.imgur.com/7b3XJ8G.png", width=220) if st.sidebar.button(" ") else None
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Overview", "Insights", "Prediction", "Actions & Export", "About"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"], help="Dataset with a 'Churn' column is preferred.")
st.sidebar.markdown("**Model file (optional)**")
model_file = st.sidebar.file_uploader("Upload churn_model.pkl (joblib)", type=["pkl","joblib"])
st.sidebar.markdown("---")
st.sidebar.write("Tip: If no model file is provided, the app will auto-train a quick RandomForest (if `Churn` present).")

# -------------------------
# Load data & model
# -------------------------
df = None
if uploaded_file:
    try:
        df = load_df(uploaded_file)
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))

model = None
model_features = []
if model_file:
    try:
        model = joblib.load(model_file)
        st.sidebar.success("Loaded model file.")
    except Exception as e:
        st.sidebar.error("Failed to load model: " + str(e))

# If no model file but data includes Churn, auto-train
if not model and df is not None and "Churn" in df.columns:
    with st.sidebar.expander("Auto-train quick model"):
        st.write("No model provided — training quick RandomForest on numeric features. This is for demo/prediction only.")
        if st.button("Train now"):
            with st.spinner("Training quick model..."):
                model, model_features = train_quick_model(df, target_col="Churn")
                if model is None:
                    st.sidebar.error("Auto-train failed. Check that dataset has numeric features + target 'Churn'.")
                else:
                    st.sidebar.success("Model trained.")
else:
    # If model present and we can infer features later when user supplies data for prediction
    pass

# -------------------------
# Page: Overview
# -------------------------
if page == "Overview":
    st.markdown("<header><div style='color:white; font-size:22px;'>📡 Telecom Churn Insights — Dashboard</div></header>", unsafe_allow_html=True)
    st.write("")
    if df is None:
        st.info("Upload your dataset from the sidebar to see interactive insights. Recommended columns: CustomerID, gender, tenure, MonthlyCharges, TotalCharges, Contract, Churn.")
        st.stop()

    st.markdown("## Overview")
    # KPIs
    total_customers = len(df)
    churn_rate = None
    if "Churn" in df.columns:
        churn_rate = df["Churn"].apply(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0).mean()
    avg_tenure = df["tenure"].mean() if "tenure" in df.columns else np.nan
    avg_monthly = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns([1.6,1,1,1])
    c1.markdown(f"<div class='kpi'><h3>Total Customers</h3><h2>{total_customers}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><h3>Churn Rate</h3><h2>{(churn_rate*100):.2f}%</h2></div>" if churn_rate is not None else "<div class='kpi'><h3>Churn Rate</h3><h2>--</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'><h3>Avg Tenure</h3><h2>{avg_tenure:.1f} months</h2></div>" if not np.isnan(avg_tenure) else "<div class='kpi'><h3>Avg Tenure</h3><h2>--</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi'><h3>Avg Monthly</h3><h2>₹{avg_monthly:.2f}</h2></div>" if not np.isnan(avg_monthly) else "<div class='kpi'><h3>Avg Monthly</h3><h2>--</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Churn Distribution")
    if "Churn" in df.columns:
        fig = px.pie(df, names="Churn", hole=0.5, title="Churn vs Retained", color="Churn",
                     color_discrete_map={ "Yes":"#EF4444","No":"#10B981"} )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `Churn` column found. Upload a labeled dataset to see distribution.")

# -------------------------
# Page: Insights
# -------------------------
elif page == "Insights":
    st.markdown("<header><div style='color:white; font-size:22px;'>📈 Insights</div></header>", unsafe_allow_html=True)
    if df is None:
        st.info("Upload dataset to explore insights.")
        st.stop()

    st.markdown("## Interactive Visualizations")
    # Filters
    st.sidebar.markdown("### Filters")
    filter_cols = []
    if "gender" in df.columns:
        genders = df["gender"].dropna().unique().tolist()
        sel_genders = st.sidebar.multiselect("Gender", genders, default=genders)
        df = df[df["gender"].isin(sel_genders)]
        filter_cols.append("gender")

    # Contract churn bar
    if "Contract" in df.columns:
        st.subheader("Churn by Contract Type")
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group", title="Contract vs Churn")
        st.plotly_chart(fig, use_container_width=True)

    # Monthly charges box
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        st.subheader("Monthly Charges distribution by Churn")
        fig2 = px.box(df, x="Churn", y="MonthlyCharges", points="all", title="MonthlyCharges vs Churn")
        st.plotly_chart(fig2, use_container_width=True)

    # Tenure scatter
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        st.subheader("Tenure vs Monthly Charges (size = TotalCharges)")
        size_col = "TotalCharges" if "TotalCharges" in df.columns else None
        fig3 = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn" if "Churn" in df.columns else None,
                          size=size_col, hover_data=df.columns, title="Tenure vs MonthlyCharges")
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation
    st.subheader("Correlation Heatmap (numeric features)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to compute correlation heatmap.")

    # Top reasons (heuristic)
    st.subheader("Top Reasons / Signals for Churn (heuristic view)")
    reasons = []
    if "Contract" in df.columns:
        m = df.groupby(["Contract","Churn"]).size().unstack(fill_value=0)
        if ("Yes" in m.columns) or (1 in m.columns):
            st.markdown("**Contract types with higher churn**")
            st.dataframe(m)
    if "PaymentMethod" in df.columns:
        pm = df.groupby(["PaymentMethod","Churn"]).size().unstack(fill_value=0)
        st.markdown("**Payment Methods churn summary**")
        st.dataframe(pm)

# -------------------------
# Page: Prediction
# -------------------------
elif page == "Prediction":
    st.markdown("<header><div style='color:white; font-size:22px;'>🧠 Prediction</div></header>", unsafe_allow_html=True)
    if df is None:
        st.info("Upload dataset to use prediction form or provide a model file.")
        st.stop()

    st.markdown("### Live Prediction Form")
    st.write("Fill the customer details. The app will try to shape inputs for the loaded/trained model.")

    # Prepare model_features if possible
    if model is not None and not model_features:
        # Attempt to infer numeric features from uploaded df if model expects numeric array (best-effort)
        model_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Prepare input form using common columns if available
    with st.form("predict_form"):
        cols = st.columns(3)
        # Common telecom fields — if not present, provide generic inputs
        tenure = cols[0].slider("Tenure (months)", 0, 72, int(df["tenure"].median()) if "tenure" in df.columns else 12)
        monthly = cols[1].number_input("MonthlyCharges", min_value=0.0, value=float(df["MonthlyCharges"].median()) if "MonthlyCharges" in df.columns else 50.0)
        total = cols[2].number_input("TotalCharges", min_value=0.0, value=float(df["TotalCharges"].median()) if "TotalCharges" in df.columns else monthly*tenure)
        contract = st.selectbox("Contract", options= df["Contract"].unique().tolist() if "Contract" in df.columns else ["Month-to-month","One year","Two year"])
        payment = st.selectbox("Payment Method", options=df["PaymentMethod"].unique().tolist() if "PaymentMethod" in df.columns else ["Electronic check","Mailed check","Bank transfer"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build input dataframe
        input_dict = {}
        # Use numeric features if model expects them
        if model is not None and model_features:
            # create a row with zeros
            input_df = pd.DataFrame(columns=model_features)
            input_df.loc[0] = 0
            # Try to map common fields into numeric features
            if "tenure" in model_features:
                input_df.loc[0, "tenure"] = tenure
            if "MonthlyCharges" in model_features:
                input_df.loc[0, "MonthlyCharges"] = monthly
            if "TotalCharges" in model_features:
                input_df.loc[0, "TotalCharges"] = total
            # If contract encoded numerically, try to find a matching numeric column (best effort)
            probs = model_predict_proba(model, model_features, input_df)
            prob = float(probs[0])
        else:
            # No model available — we attempt to train quick model now if dataset has Churn
            quick_model = None
            quick_feats = []
            if "Churn" in df.columns:
                quick_model, quick_feats = train_quick_model(df, target_col="Churn")
            if quick_model is not None:
                input_df = pd.DataFrame(columns=quick_feats)
                input_df.loc[0] = 0
                if "tenure" in quick_feats:
                    input_df.loc[0, "tenure"] = tenure
                if "MonthlyCharges" in quick_feats:
                    input_df.loc[0, "MonthlyCharges"] = monthly
                if "TotalCharges" in quick_feats:
                    input_df.loc[0, "TotalCharges"] = total
                probs = model_predict_proba(quick_model, quick_feats, input_df)
                prob = float(probs[0])
            else:
                st.error("No model available and auto-train not possible (no labeled 'Churn' column).")
                st.stop()

        # Show result visually
        st.markdown("#### Churn Probability")
        st.progress(int(prob*100))
        if prob >= 0.7:
            st.markdown(f"<h3 style='color:#B91C1C;'>High Risk — {prob:.2f}</h3>", unsafe_allow_html=True)
            st.markdown("**Recommended action:** Offer retention plan (discount / extra data), priority support call, escalate to retention team.")
        elif prob >= 0.4:
            st.markdown(f"<h3 style='color:#D97706;'>Medium Risk — {prob:.2f}</h3>", unsafe_allow_html=True)
            st.markdown("**Recommended action:** CRM follow-up, targeted offer, check complaint logs.")
        else:
            st.markdown(f"<h3 style='color:#047857;'>Low Risk — {prob:.2f}</h3>", unsafe_allow_html=True)
            st.markdown("**Recommended action:** Loyalty reward, cross-sell opportunity.")

# -------------------------
# Page: Actions & Export
# -------------------------
elif page == "Actions & Export":
    st.markdown("<header><div style='color:white; font-size:22px;'>🛠️ Actions & Export</div></header>", unsafe_allow_html=True)
    if df is None:
        st.info("Upload dataset to generate actions and download predictions.")
        st.stop()

    st.markdown("### Batch Scoring (simulate real-time batch)")
    if "Churn" not in df.columns and model is None:
        st.warning("No Churn label in data and no model loaded — please upload labeled data or model.")
    # Allow user to score entire dataset if model present or auto-train
    scorer = model
    feat_list = model_features
    if scorer is None and "Churn" in df.columns:
        st.info("Auto-training model for batch scoring (quick RandomForest).")
        scorer, feat_list = train_quick_model(df, target_col="Churn")

    if scorer is not None:
        if st.button("Score dataset now"):
            st.info("Scoring — creating churn probability column...")
            # Prepare inputs (very simple: select numeric columns)
            X = df.select_dtypes(include=[np.number]).copy()
            missing = [c for c in feat_list if c not in X.columns]
            for c in missing:
                X[c] = 0
            X = X[feat_list]
            probs = scorer.predict_proba(X)[:, 1]
            df_out = df.copy()
            df_out["churn_probability"] = probs
            st.success("Scoring complete! Preview:")
            st.dataframe(df_out.head(200))
            # Download
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download scored CSV", data=csv, file_name="scored_customers.csv", mime="text/csv")
            # Show top high-risk customers
            st.markdown("#### Top high-risk customers")
            top = df_out.sort_values("churn_probability", ascending=False).head(20)
            st.dataframe(top[["churn_probability"] + [c for c in df_out.columns if c != "churn_probability"]].head(20))
    else:
        st.error("No model available for batch scoring. Provide a model file or labeled dataset.")

# -------------------------
# Page: About
# -------------------------
elif page == "About":
    st.markdown("<header><div style='color:white; font-size:22px;'>ℹ️ About this App</div></header>", unsafe_allow_html=True)
    st.write("""
    **Telecom Churn Insights** — demo app built for showcasing a polished churn analytics and prediction workflow.

    Features:
    - Clean KPI cards, interactive Plotly charts, correlation heatmap
    - Live prediction form with probability + business recommendation
    - Auto-training fallback (simple RandomForest) if you upload labeled data but no model
    - Downloadable scored CSV for operational use

    👉 Tips:
    - Best results when uploaded CSV has: `CustomerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `Churn`.
    - For production, replace quick RandomForest with a properly validated model and persist `LabelEncoder` mappings for categorical features.

    Built with ❤️ — customize visuals or let me add a theme switcher / QR share / SHAP explainability (if you want deeper interpretability).
    """)



# 3rd 


# app.py
# ---------------------------------------------
# Telecom Churn Insights (Self-contained)
# - Auto-generates dataset on first run
# - Trains & caches model (model.pkl)
# - Beautiful insights + predictions + actions
# - Live simulation + SHAP explainability
# ---------------------------------------------

# import os
# import io
# import json
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_auc_score, accuracy_score, precision_recall_fscore_support,
#     confusion_matrix, classification_report, RocCurveDisplay
# )
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# # ---------- Page config ----------
# st.set_page_config(
#     page_title="📡 Telecom Churn – Real-Time Insights",
#     page_icon="📶",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------- Styled header / CSS ----------
# st.markdown("""
# <style>
# /* App background */
# .stApp { background: #0b1220; }

# /* Gradient hero bar */
# .hero {
#   background: linear-gradient(90deg, #0ea5e9 0%, #7c3aed 100%);
#   padding: 18px 22px; border-radius: 14px; color: white;
#   box-shadow: 0 12px 40px rgba(0,0,0,0.25);
# }

# /* KPI cards */
# .kpi {
#   background: #0f172a;
#   border: 1px solid #1e293b;
#   border-radius: 14px;
#   padding: 14px 16px;
#   box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
# }
# .kpi h3 { color: #94a3b8; font-weight:600; margin: 0 0 8px; }
# .kpi h2 { color: #e2e8f0; font-weight:800; margin: 0; }

# /* Section card */
# .section {
#   background: #0f172a;
#   border: 1px solid #1e293b;
#   border-radius: 16px;
#   padding: 16px;
#   margin-top: 12px;
# }

# /* Text colors */
# h2, h3, h4, p, label, span, .stText, .stMarkdown { color: #e2e8f0 !important; }
# .sidebar .sidebar-content { background: #0b1220 !important; }

# /* Plotly container padding fix */
# .block-container { padding-top: 1.2rem; padding-bottom: 6rem; }

# /* Buttons */
# .stButton>button {
#   background: linear-gradient(90deg, #3b82f6, #a855f7);
#   border: 0; color: white; font-weight: 700; padding: 0.6rem 1rem;
#   border-radius: 10px; box-shadow: 0 6px 30px rgba(168,85,247,0.25);
# }
# .stDownloadButton>button {
#   background: #0ea5e9; color: #0b1220; font-weight: 800; border: 0; border-radius: 10px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------- Utility: Generate realistic telecom dataset ----------
# def generate_telco_data(n=7000, seed=7):
#     rng = np.random.default_rng(seed)
#     gender = rng.choice(["Male", "Female"], size=n)
#     senior = rng.choice([0, 1], p=[0.84, 0.16], size=n)
#     partner = rng.choice(["Yes", "No"], p=[0.48, 0.52], size=n)
#     dependents = rng.choice(["Yes", "No"], p=[0.30, 0.70], size=n)
#     tenure = rng.integers(0, 73, size=n)

#     contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.57, 0.22, 0.21], size=n)
#     paperless = np.where(contract == "Month-to-month",
#                          rng.choice(["Yes", "No"], p=[0.85, 0.15], size=n),
#                          rng.choice(["Yes", "No"], p=[0.55, 0.45], size=n))
#     payment = rng.choice(
#         ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
#         p=[0.33, 0.16, 0.26, 0.25],
#         size=n
#     )
#     phone = rng.choice(["Yes", "No"], p=[0.90, 0.10], size=n)
#     mult_lines = np.where(phone == "Yes",
#                           rng.choice(["Yes", "No"], p=[0.50, 0.50], size=n),
#                           "No")
#     internet = rng.choice(["DSL", "Fiber optic", "No"], p=[0.30, 0.53, 0.17], size=n)

#     # Services only if internet != "No"
#     def service_choice(mask, y_prob=0.45):
#         vals = np.array(["Yes", "No"], dtype=object)
#         out = np.full(n, "No", dtype=object)
#         idx = np.where(mask)[0]
#         out[idx] = rng.choice(vals, p=[y_prob, 1-y_prob], size=len(idx))
#         return out

#     have_net = internet != "No"
#     online_sec = service_choice(have_net, 0.45)
#     online_bkp = service_choice(have_net, 0.50)
#     device_prot = service_choice(have_net, 0.50)
#     tech_supp = service_choice(have_net, 0.40)
#     stream_tv = service_choice(have_net, 0.52)
#     stream_movies = service_choice(have_net, 0.50)

#     # Charges
#     base = np.where(internet == "DSL", 25,
#             np.where(internet == "Fiber optic", 45, 15)).astype(float)
#     base += np.where(phone == "Yes", 5, 0)
#     base += np.where(mult_lines == "Yes", 4, 0)
#     base += np.where(online_sec == "Yes", 4, 0)
#     base += np.where(online_bkp == "Yes", 4, 0)
#     base += np.where(device_prot == "Yes", 4, 0)
#     base += np.where(tech_supp == "Yes", 5, 0)
#     base += np.where(stream_tv == "Yes", 6, 0)
#     base += np.where(stream_movies == "Yes", 6, 0)
#     base += rng.normal(0, 2.5, size=n)

#     monthly = np.clip(base, 10, None)
#     total = monthly * tenure + rng.normal(0, 40, size=n)
#     total = np.clip(total, 0, None)

#     # Churn probability (heuristic logistic-style mix)
#     churn_logit = (
#         1.25 * (contract == "Month-to-month").astype(float)
#         - 0.8 * (contract == "Two year").astype(float)
#         + 0.35 * (payment == "Electronic check").astype(float)
#         + 0.25 * (internet == "Fiber optic").astype(float)
#         + 0.15 * (paperless == "Yes").astype(float)
#         + 0.30 * (senior).astype(float)
#         + 0.015 * (monthly - monthly.mean())  # higher monthly increases risk slightly
#         - 0.03 * tenure                        # loyalty reduces risk
#     )
#     # Normalize to 0..1 with sigmoid
#     churn_prob = 1 / (1 + np.exp(-churn_logit))
#     churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

#     df = pd.DataFrame({
#         "customerID": [f"C{100000 + i}" for i in range(n)],
#         "gender": gender,
#         "SeniorCitizen": senior,
#         "Partner": partner,
#         "Dependents": dependents,
#         "tenure": tenure,
#         "PhoneService": phone,
#         "MultipleLines": mult_lines,
#         "InternetService": internet,
#         "OnlineSecurity": online_sec,
#         "OnlineBackup": online_bkp,
#         "DeviceProtection": device_prot,
#         "TechSupport": tech_supp,
#         "StreamingTV": stream_tv,
#         "StreamingMovies": stream_movies,
#         "Contract": contract,
#         "PaperlessBilling": paperless,
#         "PaymentMethod": payment,
#         "MonthlyCharges": np.round(monthly, 2),
#         "TotalCharges": np.round(total, 2),
#         "Churn": churn,
#     })
#     # A few NaNs like real data
#     mask_nan = (df["tenure"] == 0) & (df["TotalCharges"] < 50)
#     df.loc[mask_nan, "TotalCharges"] = np.nan
#     return df

# # ---------- Ensure assets (dataset + model) ----------
# DATA_PATH = "telecom_churn.csv"
# MODEL_PATH = "model.pkl"
# META_PATH = "model_meta.json"

# def ensure_assets():
#     if not os.path.exists(DATA_PATH):
#         df = generate_telco_data(n=7000, seed=10)
#         df.to_csv(DATA_PATH, index=False)
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
#         train_and_save()

# @st.cache_data
# def load_data():
#     return pd.read_csv(DATA_PATH)

# @st.cache_resource
# def load_model():
#     clf = joblib.load(MODEL_PATH)
#     with open(META_PATH, "r", encoding="utf-8") as f:
#         meta = json.load(f)
#     return clf, meta

# def make_pipeline(df):
#     # Prepare features/target
#     y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
#     X = df.drop(columns=["Churn", "customerID"], errors="ignore")

#     num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

#     pre = ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(with_mean=False), num_cols),
#             ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False,
#     )
#     model = RandomForestClassifier(
#         n_estimators=400, max_depth=None, min_samples_leaf=2,
#         random_state=42, n_jobs=-1, class_weight="balanced_subsample"
#     )
#     pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
#     return pipe, num_cols, cat_cols

# def train_and_save():
#     df = generate_telco_data(n=7000, seed=11)
#     df.to_csv(DATA_PATH, index=False)

#     pipe, num_cols, cat_cols = make_pipeline(df)
#     y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
#     X = df.drop(columns=["Churn", "customerID"], errors="ignore")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.18, stratify=y, random_state=42
#     )
#     pipe.fit(X_train, y_train)

#     # Persist
#     joblib.dump(pipe, MODEL_PATH)
#     # Get feature names after fit
#     ohe = pipe.named_steps["pre"].named_transformers_["cat"]
#     cat_features = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
#     feat_names = num_cols + cat_features
#     meta = {
#         "numeric_features": num_cols,
#         "categorical_features": cat_cols,
#         "feature_names_after_pre": feat_names,
#         "train_size": int(X_train.shape[0]),
#         "test_size": int(X_test.shape[0]),
#     }
#     with open(META_PATH, "w", encoding="utf-8") as f:
#         json.dump(meta, f, indent=2)

# # Bootstrap assets
# ensure_assets()
# df = load_data()
# model, meta = load_model()

# # ---------- Sidebar ----------
# st.sidebar.markdown(
#     "<div class='hero'><b>📡 Telecom Churn</b><br/><small>Real-Time Insight Dashboard</small></div>",
#     unsafe_allow_html=True,
# )
# st.sidebar.write("")
# nav = st.sidebar.radio(
#     "Navigate",
#     ["🏠 Overview", "📊 Insights", "🧩 Segments", "🔮 Predict", "📈 Live Monitor", "🧠 Explainability", "✅ Performance", "⚙️ Settings"],
# )

# # ---------- Helper visuals ----------
# def metric_card(label, value):
#     st.markdown(f"<div class='kpi'><h3>{label}</h3><h2>{value}</h2></div>", unsafe_allow_html=True)

# def prob_color(p):
#     if p >= 0.7: return "#ef4444"
#     if p >= 0.4: return "#f59e0b"
#     return "#10b981"

# def recommend_actions(prob, row=None):
#     actions = []
#     if prob >= 0.7:
#         actions += [
#             "Priority retention call within 24h",
#             "Offer 20% discount or double data for 3 months",
#             "Network check at location; escalate tickets",
#         ]
#     elif prob >= 0.4:
#         actions += [
#             "Proactive care call & issue diagnosis",
#             "Targeted add-on or plan optimization",
#             "Educate on e-billing / autopay benefits",
#         ]
#     else:
#         actions += [
#             "Loyalty rewards (1 GB bonus / OTT add-on)",
#             "Cross-sell family plan / device protection",
#         ]
#     return actions

# # ---------- Overview ----------
# if nav == "🏠 Overview":
#     st.markdown("<div class='hero'><h2>📡 Telecom Churn — Real-Time Insights</h2><p>Visualize, predict, and act — all in one place.</p></div>", unsafe_allow_html=True)
#     st.write("")

#     total_customers = len(df)
#     churn_rate = (df["Churn"].str.lower().isin(["yes","y","1","true","t"])).mean()
#     avg_tenure = df["tenure"].mean()
#     avg_monthly = df["MonthlyCharges"].mean()
#     est_monthly_rev = (df["MonthlyCharges"].sum())
#     est_revenue_at_risk = (df.loc[df["Churn"].str.lower().isin(["yes"]),"MonthlyCharges"].mean() * churn_rate * total_customers)

#     c1, c2, c3, c4, c5 = st.columns([1.15,1,1,1,1.4])
#     metric_card("Total Customers", f"{total_customers:,}")
#     with c2: metric_card("Churn Rate", f"{churn_rate*100:,.2f}%")
#     with c3: metric_card("Avg Tenure", f"{avg_tenure:,.1f} mo")
#     with c4: metric_card("Avg Monthly", f"₹{avg_monthly:,.2f}")
#     with c5: metric_card("Revenue / mo", f"₹{est_monthly_rev:,.0f}")

#     with st.container():
#         colA, colB = st.columns([1,1])
#         with colA:
#             st.markdown("<div class='section'>", unsafe_allow_html=True)
#             st.subheader("Churn Distribution")
#             fig = px.pie(df, names="Churn", hole=0.5, color="Churn",
#                          color_discrete_map={"Yes":"#ef4444", "No":"#10b981"},
#                          title="Churn vs Retained")
#             st.plotly_chart(fig, use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with colB:
#             st.markdown("<div class='section'>", unsafe_allow_html=True)
#             st.subheader("Contract Mix")
#             fig2 = px.histogram(df, x="Contract", color="Churn", barmode="group", title="Contract vs Churn")
#             st.plotly_chart(fig2, use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Insights ----------
# elif nav == "📊 Insights":
#     st.markdown("<div class='hero'><h2>📊 Insights</h2><p>Explore drivers & patterns behind churn.</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     # Quick filters
#     fcol1, fcol2, fcol3 = st.columns(3)
#     contracts = sorted(df["Contract"].dropna().unique().tolist())
#     payments = sorted(df["PaymentMethod"].dropna().unique().tolist())
#     internet = sorted(df["InternetService"].dropna().unique().tolist())
#     sel_contracts = fcol1.multiselect("Contract", contracts, default=contracts)
#     sel_payments = fcol2.multiselect("Payment Method", payments, default=payments)
#     sel_internet = fcol3.multiselect("Internet Service", internet, default=internet)

#     view = df[
#         df["Contract"].isin(sel_contracts)
#         & df["PaymentMethod"].isin(sel_payments)
#         & df["InternetService"].isin(sel_internet)
#     ].copy()

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Monthly Charges by Churn")
#         fig = px.box(view, x="Churn", y="MonthlyCharges", points="all", color="Churn")
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.subheader("Tenure vs Monthly Charges")
#         fig = px.scatter(view, x="tenure", y="MonthlyCharges", color="Churn",
#                          hover_data=["Contract","PaymentMethod","InternetService"])
#         st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Correlation (numeric)")
#     num_df = view.select_dtypes(include=[np.number])
#     if num_df.shape[1] >= 2:
#         corr = num_df.corr(numeric_only=True)
#         fig, ax = plt.subplots(figsize=(8,5))
#         sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
#     else:
#         st.info("Not enough numeric columns for correlation.")

#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Segments ----------
# elif nav == "🧩 Segments":
#     st.markdown("<div class='hero'><h2>🧩 Customer Segments</h2><p>Where do risks concentrate?</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     df2 = df.copy()
#     df2["TenureBand"] = pd.cut(df2["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6","7-12","13-24","25-48","49-72"])
#     df2["ChargeBand"] = pd.cut(df2["MonthlyCharges"], bins=[-1, 30, 60, 90, 120, 999], labels=["≤30","31-60","61-90","91-120",">120"])

#     pivot = (df2
#         .groupby(["TenureBand","ChargeBand"])["Churn"]
#         .apply(lambda s: (s.str.lower()=="yes").mean())
#         .reset_index(name="ChurnRate")
#     )
#     fig = px.density_heatmap(pivot, x="TenureBand", y="ChargeBand", z="ChurnRate", color_continuous_scale="Reds",
#                              title="Churn Rate Heatmap (by Tenure & Monthly Charges)")
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Predict ----------
# elif nav == "🔮 Predict":
#     st.markdown("<div class='hero'><h2>🔮 Predict Churn</h2><p>Score a single customer and get actions.</p></div>", unsafe_allow_html=True)
#     st.write("")

#     # Build a form with common features (match our generated columns)
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         senior = st.selectbox("SeniorCitizen", [0,1])
#         partner = st.selectbox("Partner", ["Yes","No"])
#         dependents = st.selectbox("Dependents", ["Yes","No"])
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#     with c2:
#         phone = st.selectbox("PhoneService", ["Yes","No"])
#         multiple = st.selectbox("MultipleLines", ["Yes","No"])
#         internet = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
#         online_sec = st.selectbox("OnlineSecurity", ["Yes","No"])
#         online_bkp = st.selectbox("OnlineBackup", ["Yes","No"])
#     with c3:
#         device = st.selectbox("DeviceProtection", ["Yes","No"])
#         tech = st.selectbox("TechSupport", ["Yes","No"])
#         stream_tv = st.selectbox("StreamingTV", ["Yes","No"])
#         stream_movies = st.selectbox("StreamingMovies", ["Yes","No"])
#         contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
#         paperless = st.selectbox("PaperlessBilling", ["Yes","No"])
#         payment = st.selectbox("PaymentMethod", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

#     # Auto-compute monthly/total charges suggestion
#     base = 15.0
#     base += 10 if internet == "No" else (25 if internet == "DSL" else 45)
#     base += 5 if phone == "Yes" else 0
#     base += 4 if multiple == "Yes" else 0
#     base += 4 if online_sec == "Yes" else 0
#     base += 4 if online_bkp == "Yes" else 0
#     base += 4 if device == "Yes" else 0
#     base += 5 if tech == "Yes" else 0
#     base += 6 if stream_tv == "Yes" else 0
#     base += 6 if stream_movies == "Yes" else 0

#     colm = st.columns(2)
#     with colm[0]:
#         monthly = st.number_input("MonthlyCharges", min_value=10.0, value=float(np.round(base,2)))
#     with colm[1]:
#         total = st.number_input("TotalCharges", min_value=0.0, value=float(np.round(monthly*tenure,2)))

#     # Build DF
#     row = pd.DataFrame([{
#         "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
#         "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
#         "InternetService": internet, "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
#         "DeviceProtection": device, "TechSupport": tech, "StreamingTV": stream_tv,
#         "StreamingMovies": stream_movies, "Contract": contract, "PaperlessBilling": paperless,
#         "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
#     }])

#     if st.button("🚀 Predict Now"):
#         prob = float(model.predict_proba(row)[0][1])
#         pred = int(prob >= 0.5)
#         color = prob_color(prob)
#         st.markdown(f"<div class='section'><h3>Result</h3><h2 style='color:{color}'>Churn Probability: {prob:.2f}</h2></div>", unsafe_allow_html=True)
#         if pred == 1:
#             st.error("Customer likely to **CHURN**")
#         else:
#             st.success("Customer likely to **STAY**")

#         st.subheader("Recommended Actions")
#         for a in recommend_actions(prob, row):
#             st.markdown(f"- {a}")

# # ---------- Live Monitor ----------
# elif nav == "📈 Live Monitor":
#     st.markdown("<div class='hero'><h2>📈 Live Monitor</h2><p>Simulate streaming customers and triage risk.</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     n_new = st.slider("How many new customers to simulate?", 10, 1000, 100, step=10)
#     if st.button("Generate & Score Batch"):
#         new_df = generate_telco_data(n=n_new, seed=int(time.time()) % 10_000)
#         X_new = new_df.drop(columns=["Churn","customerID"], errors="ignore")
#         probs = model.predict_proba(X_new)[:,1]
#         out = new_df.copy()
#         out["churn_probability"] = probs
#         out["risk_band"] = pd.cut(out["churn_probability"], bins=[-0.01,0.4,0.7,1.0], labels=["Low","Medium","High"])
#         st.success(f"Scored {len(out)} customers.")
#         st.dataframe(out.sort_values("churn_probability", ascending=False).head(50))

#         # Actions summary
#         summary = out["risk_band"].value_counts().reindex(["High","Medium","Low"]).fillna(0).astype(int)
#         fig = px.bar(summary, title="Risk Distribution (Batch)", labels={"value":"Count","index":"Risk Band"})
#         st.plotly_chart(fig, use_container_width=True)

#         # Download
#         csv = out.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇️ Download Scored Batch (CSV)", csv, file_name="scored_batch.csv", mime="text/csv")

#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Explainability (SHAP) ----------
# elif nav == "🧠 Explainability":
#     st.markdown("<div class='hero'><h2>🧠 Explainability</h2><p>Understand feature impact on churn.</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     # Feature importances from RF
#     try:
#         clf = model.named_steps["clf"]
#         pre = model.named_steps["pre"]
#         ohe = pre.named_transformers_["cat"]
#         cat_cols = pre.transformers_[1][2]
#         num_cols = pre.transformers_[0][2]
#         feat_names = list(num_cols) + (list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else [])
#         importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
#         fig = px.bar(importances[::-1], orientation="h", title="Top 20 Feature Importances")
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.info(f"Could not compute feature importances: {e}")

#     st.markdown("---")
#     st.subheader("SHAP Summary (sampled)")
#     st.caption("NOTE: Calculated on a sample for speed. Works best with tree-based models.")

#     use_shap = st.checkbox("Compute SHAP now (may take ~10–30s on first run)", value=False)
#     if use_shap:
#         try:
#             import shap  # heavy import, keep inside
#             # Sample data
#             sample = df.sample(min(800, len(df)), random_state=42).drop(columns=["Churn","customerID"], errors="ignore")
#             X_trans = model.named_steps["pre"].transform(sample)
#             clf = model.named_steps["clf"]
#             explainer = shap.TreeExplainer(clf)
#             shap_values = explainer.shap_values(X_trans)
#             # class 1 (churn) shap values
#             sv = shap_values[1] if isinstance(shap_values, list) else shap_values
#             feat_names = meta.get("feature_names_after_pre", None)
#             st.write("Rendering SHAP summary plot...")
#             fig, ax = plt.subplots(figsize=(8,6))
#             shap.summary_plot(sv, features=X_trans, feature_names=feat_names, show=False)
#             st.pyplot(fig)
#         except Exception as e:
#             st.error(f"SHAP failed: {e}")

#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Performance ----------
# elif nav == "✅ Performance":
#     st.markdown("<div class='hero'><h2>✅ Model Performance</h2><p>How well does the model capture churn?</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     # Re-split just for reporting (same seed → stable)
#     y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
#     X = df.drop(columns=["Churn","customerID"], errors="ignore")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, stratify=y, random_state=42)

#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:,1]

#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_prob)
#     prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

#     c1, c2, c3, c4 = st.columns(4)
#     metric_card("Accuracy", f"{acc*100:,.2f}%")
#     metric_card("ROC AUC", f"{auc:,.3f}")
#     metric_card("Precision", f"{prec:,.3f}")
#     metric_card("Recall", f"{rec:,.3f}")

#     colx, coly = st.columns(2)
#     with colx:
#         cm = confusion_matrix(y_test, y_pred)
#         fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
#                         labels=dict(x="Predicted", y="Actual", color="Count"),
#                         title="Confusion Matrix")
#         st.plotly_chart(fig, use_container_width=True)
#     with coly:
#         # Quick ROC curve
#         from sklearn.metrics import roc_curve
#         fpr, tpr, _ = roc_curve(y_test, y_prob)
#         roc_fig = go.Figure()
#         roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
#         roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
#         roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
#         st.plotly_chart(roc_fig, use_container_width=True)

#     st.subheader("Classification Report")
#     report = classification_report(y_test, y_pred, target_names=["No Churn","Churn"], output_dict=True)
#     rep_df = pd.DataFrame(report).T
#     st.dataframe(rep_df.style.background_gradient(cmap="PuBu"))

#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------- Settings ----------
# elif nav == "⚙️ Settings":
#     st.markdown("<div class='hero'><h2>⚙️ Settings</h2><p>Retrain or regenerate synthetic data.</p></div>", unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='section'>", unsafe_allow_html=True)

#     st.write("You can regenerate the dataset and retrain the model (useful for demos).")
#     regen_n = st.slider("Dataset size", 2000, 20000, 7000, step=1000)
#     if st.button("Regenerate dataset & retrain"):
#         with st.spinner("Regenerating & training..."):
#             # Create new dataset
#             df_new = generate_telco_data(n=regen_n, seed=int(time.time()) % 10000)
#             df_new.to_csv(DATA_PATH, index=False)
#             # Retrain
#             train_and_save()
#             # Clear caches
#             st.cache_data.clear()
#             st.cache_resource.clear()
#         st.success("Done! Restart the app or switch tabs to see updates.")
#     st.markdown("</div>", unsafe_allow_html=True)



