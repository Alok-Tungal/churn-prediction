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





# app.py
"""
Robust Telecom Churn Streamlit app (self-contained).
- Auto-loads local dataset/model if present (Churn_data.csv, advanced_churn_model.pkl, model.pkl)
- If missing, generates synthetic dataset and trains a stable pipeline
- Beautiful UI (Plotly + styled CSS)
- Safe SHAP behind checkbox (optional heavy import)
- Handles various pickle formats (Pipeline OR legacy (model, scaler, model_columns))
"""

import os
import time
import json
import joblib
import inspect
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, classification_report
)

# ---------------- page config & CSS ----------------
st.set_page_config(page_title="📡 Telecom Churn — Insights", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071126 0%, #091127 100%); color: #E6EEF3; }
    .hero { background: linear-gradient(90deg,#0ea5e9 0%, #7c3aed 100%); padding: 18px 22px; border-radius: 12px; color: white; box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
    .kpi { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px; }
    .section { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; margin-top: 12px; }
    .metric h2 { margin: 0; }
    .stButton>button { background: linear-gradient(90deg,#3b82f6,#a855f7); color: white; font-weight: 700; border-radius: 8px; }
    .stDownloadButton>button { background: #10b981; color: white; font-weight: 700; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- compatibility helpers ----------------
def make_onehot_encoder(**kwargs):
    """
    Build OneHotEncoder with compatible arg across sklearn versions.
    """
    sig = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    elif "sparse" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)
    else:
        return OneHotEncoder(handle_unknown="ignore", **kwargs)

# -------------- synthetic data generator (realistic) --------------
def generate_telco_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0,1], p=[0.84,0.16], size=n)
    partner = rng.choice(["Yes","No"], p=[0.48,0.52], size=n)
    dependents = rng.choice(["Yes","No"], p=[0.30,0.70], size=n)
    tenure = rng.integers(0, 73, size=n)
    contract = rng.choice(["Month-to-month","One year","Two year"], p=[0.57,0.22,0.21], size=n)
    payment = rng.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
                         p=[0.33,0.16,0.26,0.25], size=n)
    phone = rng.choice(["Yes","No"], p=[0.90,0.10], size=n)
    mult_lines = np.where(phone=="Yes", rng.choice(["Yes","No"], p=[0.50,0.50], size=n), "No")
    internet = rng.choice(["DSL","Fiber optic","No"], p=[0.30,0.53,0.17], size=n)

    def svc(mask, p=0.45):
        vals = np.array(["Yes","No"], dtype=object)
        out = np.full(n, "No", dtype=object)
        idx = np.where(mask)[0]
        out[idx] = rng.choice(vals, p=[p, 1-p], size=len(idx))
        return out

    have_net = internet != "No"
    online_security = svc(have_net, 0.45)
    online_backup = svc(have_net, 0.5)
    device_prot = svc(have_net, 0.5)
    tech_support = svc(have_net, 0.4)
    streaming_tv = svc(have_net, 0.52)
    streaming_movies = svc(have_net, 0.5)

    base = np.where(internet=="DSL", 25, np.where(internet=="Fiber optic", 45, 15)).astype(float)
    base += np.where(phone=="Yes", 5, 0)
    base += np.where(mult_lines=="Yes", 4, 0)
    base += np.where(online_security=="Yes", 4, 0)
    base += np.where(online_backup=="Yes", 4, 0)
    base += np.where(device_prot=="Yes", 4, 0)
    base += np.where(tech_support=="Yes", 5, 0)
    base += np.where(streaming_tv=="Yes", 6, 0)
    base += np.where(streaming_movies=="Yes", 6, 0)
    base += rng.normal(0, 2.5, size=n)

    monthly = np.clip(base, 10, None)
    total = monthly * tenure + rng.normal(0, 40, size=n)
    total = np.clip(total, 0, None)

    churn_logit = (
        1.2 * (contract == "Month-to-month").astype(float)
        - 0.75 * (contract == "Two year").astype(float)
        + 0.35 * (payment == "Electronic check").astype(float)
        + 0.25 * (internet == "Fiber optic").astype(float)
        + 0.25 * (senior).astype(float)
        + 0.02 * (monthly - monthly.mean())
        - 0.03 * tenure
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"C{100000+i}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes","No"], p=[0.6,0.4], size=n),
        "PaymentMethod": payment,
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total, 2),
        "Churn": churn,
    })

    mask_nan = (df["tenure"]==0) & (df["TotalCharges"]<50)
    df.loc[mask_nan, "TotalCharges"] = np.nan
    return df

# ---------------- filenames ----------------
DATA_FILES_PREFER = ["Churn_data.csv", "telecom_churn.csv"]
MODEL_FILES_PREFER = ["advanced_churn_model.pkl", "model.pkl"]

DATA_PATH = None
MODEL_PATH = None
META_PATH = "model_meta.json"

# ---------------- load or create assets ----------------
def ensure_dataset():
    global DATA_PATH
    for f in DATA_FILES_PREFER:
        if os.path.exists(f):
            DATA_PATH = f
            return
    # not found → generate
    DATA_PATH = DATA_FILES_PREFER[0]
    df = generate_telco_data(n=5000, seed=int(time.time())%10000)
    df.to_csv(DATA_PATH, index=False)

def find_model_path():
    global MODEL_PATH
    for f in MODEL_FILES_PREFER:
        if os.path.exists(f):
            MODEL_PATH = f
            return
    MODEL_PATH = None

# ---------------- robust model loader ----------------
class LegacyModelWrapper:
    """
    Wrap older pickles of form (model, scaler, model_columns) to expose predict/predict_proba on raw DataFrame.
    """
    def __init__(self, model, scaler, model_columns):
        self.model = model
        self.scaler = scaler
        self.columns = list(model_columns)

    def _prepare(self, X):
        Xc = X.copy()
        # ensure columns present
        for c in self.columns:
            if c not in Xc.columns:
                Xc[c] = 0
        Xc = Xc[self.columns]
        # try to cast to numeric
        try:
            Xc = Xc.astype(float)
        except Exception:
            Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if hasattr(self.scaler, "transform"):
            return self.scaler.transform(Xc)
        return Xc.values

    def predict_proba(self, X):
        Xp = self._prepare(X)
        return self.model.predict_proba(Xp)

    def predict(self, X):
        return self.model.predict(self._prepare(X))

def load_model_if_exists() -> Optional[object]:
    """
    Attempts to load a model file (joblib/pickle).
    Returns either a Pipeline-like object with predict_proba, or None.
    """
    find_model_path()
    if MODEL_PATH is None:
        return None
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not joblib.load {MODEL_PATH}: {e}")
        return None

    # If it's a Pipeline or estimator with predict_proba, return it
    if hasattr(obj, "predict_proba") and hasattr(obj, "predict"):
        return obj

    # If it's a tuple like (model, scaler, model_columns)
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        model, scaler, model_columns = obj
        return LegacyModelWrapper(model, scaler, model_columns)

    # unknown format
    st.warning("Loaded model has unknown structure; auto retrain will be used.")
    return None

# ---------------- training pipeline ----------------
def train_and_save(data_path, model_path="model.pkl", meta_path=META_PATH, random_state=42):
    df = pd.read_csv(data_path)
    # coerce numeric columns
    for c in ["tenure","MonthlyCharges","TotalCharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
    X = df.drop(columns=["Churn","customerID"], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    ohe = make_onehot_encoder()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ], remainder="drop", verbose_feature_names_out=False
    )
    model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample")
    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, stratify=y, random_state=random_state)
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, model_path)

    # get feature names after preprocessor for meta
    try:
        feat_names = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        feat_names = list(num_cols) + cat_cols

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": feat_names,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return pipe, meta

# ---------------- cached loaders ----------------
@st.cache_data
def load_data_cached(path):
    df = pd.read_csv(path)
    # convert numeric-looking columns
    for c in ["tenure","MonthlyCharges","TotalCharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

@st.cache_resource
def load_model_cached():
    mdl = load_model_if_exists()
    meta = {}
    if mdl is not None:
        # try to load meta if exists
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        return mdl, meta
    # model file not found -> train
    with st.spinner("Training model (first run)... this may take 1-2 minutes"):
        pipe, meta = train_and_save(DATA_PATH, model_path="model.pkl")
    # prefer to persist path name as model.pkl for future
    try:
        mdl = joblib.load("model.pkl")
    except:
        mdl = pipe
    return mdl, meta

# ---------------- bootstrap assets ----------------
ensure_dataset()
model_obj = None
try:
    df = load_data_cached(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset {DATA_PATH}: {e}")
    df = generate_telco_data(n=2000, seed=123)
    df.to_csv(DATA_PATH, index=False)

model_obj, meta = load_model_cached()

# ---------------- UI ----------------
st.markdown("<div class='hero'><h2>📡 Telecom Churn Insights</h2><small>Auto dataset & model — no uploads required</small></div>", unsafe_allow_html=True)
st.write("")

page = st.sidebar.selectbox("Navigate", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data", "🧾 Batch Score", "⚙️ Model Info"], index=0)

# ---------- PAGE: Churn Prediction ----------
if page == "🏠 Churn Prediction":
    st.title("🔮 Churn Prediction — Single Customer")
    st.markdown("Enter customer details (most important features).")

    # form inputs
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=float(df["MonthlyCharges"].median()) if "MonthlyCharges" in df.columns else 50.0)
        total = st.number_input("Total Charges", min_value=0.0, value=float(df["TotalCharges"].median() if "TotalCharges" in df.columns else monthly*tenure))
        contract = st.selectbox("Contract", sorted(df["Contract"].unique()) if "Contract" in df.columns else ["Month-to-month","One year","Two year"])
    with c2:
        payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()) if "PaymentMethod" in df.columns else ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        internet = st.selectbox("Internet Service", sorted(df["InternetService"].unique()) if "InternetService" in df.columns else ["DSL","Fiber optic","No"])
        phone = st.selectbox("PhoneService", ["Yes","No"] if "PhoneService" in df.columns else ["Yes","No"])

    # build a raw row that matches training columns
    input_row = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    user_df = pd.DataFrame([input_row])

    if st.button("🔍 Predict Churn"):
        try:
            # Ensure numeric types
            for col in ["tenure","MonthlyCharges","TotalCharges"]:
                if col in user_df.columns:
                    user_df[col] = pd.to_numeric(user_df[col], errors="coerce").fillna(0.0)

            proba = None
            if model_obj is None:
                st.error("No model available.")
            else:
                proba = float(model_obj.predict_proba(user_df)[0][1])
            st.metric("Churn probability", f"{proba:.3f}")
            if proba >= 0.7:
                st.error(f"High risk — churn probability {proba:.2%}")
            elif proba >= 0.4:
                st.warning(f"Medium risk — churn probability {proba:.2%}")
            else:
                st.success(f"Low risk — churn probability {proba:.2%}")

            # show top feature importance if model is a pipeline with RF
            try:
                if hasattr(model_obj, "named_steps") and "clf" in model_obj.named_steps:
                    clf = model_obj.named_steps["clf"]
                    # feature names
                    try:
                        feat_names = list(model_obj.named_steps["pre"].get_feature_names_out())
                    except Exception:
                        feat_names = meta.get("feature_names", None)
                    if feat_names is not None and hasattr(clf, "feature_importances_"):
                        fi = pd.Series(clf.feature_importances_, index=feat_names).nlargest(7)
                        fig = px.bar(fi.sort_values(), orientation="h", title="Top feature importances")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- PAGE: Insights & Graphs ----------
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Insights & Visualizations")
    st.markdown("Interactive charts. Use filters to slice data.")

    # filters
    contracts = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
    payments = ["All"] + sorted(df["PaymentMethod"].dropna().unique().tolist())
    ints = ["All"] + sorted(df["InternetService"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    sel_contract = c1.selectbox("Contract", contracts)
    sel_payment = c2.selectbox("Payment Method", payments)
    sel_internet = c3.selectbox("Internet Service", ints)

    view = df.copy()
    if sel_contract != "All":
        view = view[view["Contract"] == sel_contract]
    if sel_payment != "All":
        view = view[view["PaymentMethod"] == sel_payment]
    if sel_internet != "All":
        view = view[view["InternetService"] == sel_internet]

    # KPIs
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total customers", f"{len(view):,}")
    churn_rate = (view["Churn"].str.lower() == "yes").mean()
    colB.metric("Churn rate", f"{churn_rate*100:.2f}%")
    colC.metric("Avg tenure", f"{view['tenure'].mean():.1f} mo")
    colD.metric("Avg monthly", f"₹{view['MonthlyCharges'].mean():.2f}")

    st.markdown("---")
    # Churn pie
    fig = px.pie(view, names="Churn", hole=0.45, color="Churn", color_discrete_map={"Yes":"#EF4444","No":"#10B981"},
                 title="Churn vs Retained")
    st.plotly_chart(fig, use_container_width=True)

    # Tenure vs Monthly
    fig2 = px.scatter(view, x="tenure", y="MonthlyCharges", color="Churn", hover_data=["Contract","PaymentMethod"], title="Tenure vs MonthlyCharges")
    st.plotly_chart(fig2, use_container_width=True)

    # Churn by Contract
    churn_by_contract = (df.groupby("Contract")["Churn"].apply(lambda s: (s.str.lower()=="yes").mean())).reset_index(name="churn_rate")
    fig3 = px.bar(churn_by_contract, x="Contract", y="churn_rate", title="Churn Rate by Contract", labels={"churn_rate":"Churn Rate"})
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap
    num_df = view.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig4, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=False)
        st.pyplot(fig4)

# ---------- PAGE: Raw Data ----------
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(df)
    st.caption(f"Total records: {len(df)}")

# ---------- PAGE: Batch Score ----------
elif page == "🧾 Batch Score":
    st.title("🧾 Batch Score (simulate real-time)")
    st.markdown("Score the whole dataset and download the scored CSV.")

    if st.button("Score dataset now"):
        try:
            X = df.drop(columns=["Churn","customerID"], errors="ignore")
            probs = model_obj.predict_proba(X)[:,1]
            out = df.copy()
            out["churn_probability"] = probs
            out["risk_band"] = pd.cut(out["churn_probability"], bins=[-0.01,0.4,0.7,1.0], labels=["Low","Medium","High"])
            st.dataframe(out.sort_values("churn_probability", ascending=False).head(200))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored CSV", csv, file_name="scored_customers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# ---------- PAGE: Model Info ----------
elif page == "⚙️ Model Info":
    st.title("⚙️ Model Info & Performance")
    st.write("Model metadata:")
    st.json(meta)

    # quick performance snapshot
    try:
        y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
        X = df.drop(columns=["Churn","customerID"], errors="ignore")
        preds = model_obj.predict(X)
        probs = model_obj.predict_proba(X)[:,1]
        st.write("Accuracy (informal):", f"{accuracy_score(y, preds):.3f}")
        st.write("ROC AUC (informal):", f"{roc_auc_score(y, probs):.3f}")
        cm = confusion_matrix(y, preds)
        st.write("Confusion matrix:")
        st.write(cm)
        st.text(classification_report(y, preds, digits=3))
    except Exception as e:
        st.warning(f"Could not compute performance on full dataset: {e}")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ by Alok — polished for real-world demos.")



