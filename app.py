import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Safe patching
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout
st.set_page_config(page_title="📊 Churn Prediction App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

data = load_data()
model = load_model()

# Sidebar nav
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights", "📄 Raw Data"])

# =================== 🏠 Churn Prediction ===================
if page == "🏠 Churn Prediction":
    st.title("🔮 Predict Customer Churn")

    st.markdown("Enter a few important customer details:")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("💰 Monthly Charges", 0.0, 200.0, 70.0)
    with col2:
        contract = st.selectbox("📝 Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

    # Create input DataFrame
    input_dict = {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "Contract": [contract],
        "InternetService": [internet_service]
    }
    input_df = pd.DataFrame(input_dict)

    # Combine with dummy row to ensure encoding matches
    dummy_row = data.iloc[[0]].copy()
    for col in input_df.columns:
        dummy_row[col] = input_df[col].values[0]
    input_encoded = pd.get_dummies(dummy_row)

    # Align columns with training data
    model_input_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_encoded.columns
    for col in model_input_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_input_columns]

    # Predict
    if st.button("🔍 Predict Churn"):
        try:
            pred = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0][1] * 100
            if pred == 1:
                st.error(f"⚠️ Customer likely to churn (Probability: {prob:.1f}%)")
            else:
                st.success(f"✅ Customer unlikely to churn (Probability: {100 - prob:.1f}%)")
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# =================== 📈 Insights Tab ===================
elif page == "📈 Insights":
    st.title("📈 Churn Insights")

    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.bar_label(bars)
    st.pyplot(fig)

    st.subheader("📝 Churn by Contract Type")
    contract_churn = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    bars = ax.bar(contract_churn.index, contract_churn.values, color='#ffa600')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.subheader("🌐 Churn by Internet Service")
    internet_churn = data.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    bars = ax.bar(internet_churn.index, internet_churn.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("### 🔍 Key Observations")
    st.markdown("""
    - Month-to-month contracts have the highest churn.
    - Fiber optic users tend to churn more.
    - Lower tenure customers are at higher churn risk.
    """)

# =================== 📄 Raw Data ===================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)
    st.caption(f"Total records: {len(data)}")
