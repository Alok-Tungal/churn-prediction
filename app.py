import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# --- If pickle used custom function, patch it here ---
def ordinal_encode_func(df):
    # Placeholder if your model was trained with a custom encoder
    return df

import sys
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func
# -----------------------------------------------------

# Set Streamlit layout
st.set_page_config(page_title="📊 Telecom Churn Dashboard", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('churn_data.csv')  # ✅ Ensure this file exists in your folder

# Load model and scaler
@st.cache_resource
def load_advanced_model():
    with open('churn_pred.pkl', 'rb') as f:
        model, scaler = pickle.load(f)  # ✅ Fix: unpack only 2 values

    # ✅ Manually define expected input columns
    model_columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No'
    ]
    
    return model, scaler, model_columns

# Load resources
data = load_data()
model, scaler, model_columns = load_advanced_model()

# Title and KPI
st.title("📊 Telecom Customer Churn Dashboard")
st.caption("Explore churn patterns, predict churn, and understand why.")
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
st.metric("📉 Overall Churn Rate", f"{churn_rate:.2f} %")

# Tabs
tab_viz, tab_predict = st.tabs(["📈 Analysis & Insights", "🔮 Predict Churn"])

# === 📈 ANALYSIS TAB ===
with tab_viz:
    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B','#4ECDC4'])
    ax.bar_label(bars)
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes',0)*100
    fig, ax = plt.subplots()
    bars = ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(bars, fmt='%.1f%%')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes',0)*100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### ✏️ **Key Business Insights**")
    st.markdown("""
    - Highest churn for month-to-month contracts & electronic check payments.
    - Higher monthly & total charges linked to churn.
    - Short-tenure customers churn more.
    """)

# === 🔮 PREDICT TAB ===
with tab_predict:
    st.subheader("🔮 Predict Customer Churn")
    st.markdown("Enter customer details below:")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (months)', min_value=0, max_value=100, value=12)
        monthly = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)
        total = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=2500.0)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

    # Prepare input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total],
        f'Contract_{contract}': [1],
        f'PaymentMethod_{payment}': [1],
        f'InternetService_{internet}': [1]
    })

    # Fill missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    if st.button('Predict'):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        if pred == 1:
            st.error(f"⚠️ Likely to churn! (Probability: {prob:.1f}%)")
        else:
            st.success(f"✅ Not likely to churn (Probability: {100 - prob:.1f}%)")

        # Optional: Feature importance (only if model supports it)
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Importance (Top 5)")
            feat_df = pd.DataFrame({
                'feature': model_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            fig, ax = plt.subplots()
            bars = ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            st.pyplot(fig)

st.markdown("---")
