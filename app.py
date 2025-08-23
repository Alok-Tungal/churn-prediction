# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle
# import numpy as np
# import sys

# # Patch custom functions if needed
# def ordinal_encode_func(df): return df
# sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# # Layout settings
# st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
# sns.set(style='whitegrid')
# plt.rcParams['figure.figsize'] = (8, 5)

# # Load Data
# @st.cache_data
# def load_data():
#     return pd.read_csv('Churn_data.csv')

# # Load model, scaler, and model_columns from the pickle file
# @st.cache_resource
# def load_model():
#     with open('advanced_churn_model.pkl', 'rb') as f:
#         model, scaler, model_columns = pickle.load(f)
#     return model, scaler, model_columns

# # Load everything
# data = load_data()
# model, scaler, model_columns = load_model()

# # Sidebar Navigation
# st.sidebar.title("🔍 Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# # ================== 🏠 MAIN PAGE: CHURN PREDICTION ==================
# if page == "🏠 Churn Prediction":
#     st.title("🔮 Telecom Churn Prediction")
#     st.markdown("Enter important customer details to predict churn likelihood.")

#     # Use only most relevant features
#     col1, col2 = st.columns(2)
#     with col1:
#         tenure = st.slider('Tenure (months)', 0, 100, 12)
#         monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
#         total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
#     with col2:
#         contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
#         payment = st.selectbox('Payment Method', [
#             'Electronic check', 'Mailed check',
#             'Bank transfer (automatic)', 'Credit card (automatic)'
#         ])
#         internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

#     # Build user input
#     input_dict = {
#         'tenure': tenure,
#         'MonthlyCharges': monthly,
#         'TotalCharges': total,
#         f'Contract_{contract}': 1,
#         f'PaymentMethod_{payment}': 1,
#         f'InternetService_{internet}': 1,
#     }

#     # Convert to DataFrame and fill missing model columns
#     user_df = pd.DataFrame([input_dict])
#     for col in model_columns:
#         if col not in user_df.columns:
#             user_df[col] = 0  # fill others with 0
#     user_df = user_df[model_columns]  # ensure correct order

#     # Scale and predict
#     if st.button("🔍 Predict Churn"):
#         try:
#             input_scaled = scaler.transform(user_df)
#             prediction = model.predict(input_scaled)[0]
#             probability = model.predict_proba(input_scaled)[0][1] * 100

#             if prediction == 1:
#                 st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
#             else:
#                 st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

#             # Show Feature Importance if available
#             if hasattr(model, 'feature_importances_'):
#                 st.subheader("📊 Feature Importance (Top 5)")
#                 feat_df = pd.DataFrame({
#                     'feature': model_columns,
#                     'importance': model.feature_importances_
#                 }).sort_values('importance', ascending=False).head(5)
#                 fig, ax = plt.subplots()
#                 ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
#                 ax.invert_yaxis()
#                 ax.set_xlabel("Importance")
#                 st.pyplot(fig)

#         except Exception as e:
#             st.error(f"❌ Prediction Error: {str(e)}")

# # ================== 📈 INSIGHTS ==================
# elif page == "📈 Insights & Graphs":
#     st.title("📈 Churn Insights & Visualizations")

#     st.subheader("✅ Churn Distribution")
#     churn_counts = data['Churn'].value_counts()
#     fig, ax = plt.subplots()
#     ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
#     ax.bar_label(ax.containers[0])
#     st.pyplot(fig)

#     st.subheader("📑 Churn by Contract Type")
#     churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
#     fig, ax = plt.subplots()
#     ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
#     ax.bar_label(ax.containers[0], fmt='%.1f%%')
#     ax.set_ylabel('Churn Rate (%)')
#     st.pyplot(fig)

#     st.subheader("💳 Churn by Payment Method")
#     churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
#     churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
#     fig, ax = plt.subplots()
#     ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
#     ax.bar_label(ax.containers[0], fmt='%.1f%%')
#     st.pyplot(fig)

#     st.markdown("### 🧠 Key Business Insights")
#     st.markdown("""
#     - Month-to-month contracts show the highest churn.
#     - Electronic checks are most churn-prone.
#     - Short-tenure and high-monthly-charge customers are likely to churn.
#     """)

# # ================== 📄 RAW DATA ==================
# elif page == "📄 Raw Data":
#     st.title("📄 Raw Dataset")
#     st.dataframe(data)
#     st.caption(f"Total Records: {len(data)}")



# app.py
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

