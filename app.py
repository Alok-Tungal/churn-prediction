import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Patch custom functions if needed
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Streamlit config
st.set_page_config(page_title="📊 Churn Predictor", layout="wide")
sns.set(style='whitegrid')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model
@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        return pickle.load(f)

# ✅ Manually define model columns to avoid file dependency
model_columns = [
    'tenure', 'MonthlyCharges', 'Contract_Month-to-month',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Load model and data
data = load_data()
model = load_model()

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights", "📄 Raw Data"])

# ========== 🏠 Prediction ==========
if page == "🏠 Churn Prediction":
    st.title("🔮 Predict Churn Risk")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])

    # Prepare input
    user_input = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'Contract': [contract],
        'PaymentMethod': [payment]
    })

    # One-hot encode
    user_encoded = pd.get_dummies(user_input)

    # Ensure all expected model columns exist
    for col in model_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    user_encoded = user_encoded[model_columns]  # correct order

    if st.button("🔍 Predict Churn"):
        try:
            prediction = model.predict(user_encoded)[0]
            probability = model.predict_proba(user_encoded)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# ========== 📈 Insights ==========
elif page == "📈 Insights":
    st.title("📊 Churn Analysis")

    st.subheader("Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    st.pyplot(fig)

    st.subheader("Churn by Contract")
    churn_by_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.bar(churn_by_contract.index, churn_by_contract.values, color='orange')
    st.pyplot(fig)

    st.subheader("Churn by Payment Method")
    churn_by_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.barh(churn_by_payment.index, churn_by_payment.values, color='skyblue')
    st.pyplot(fig)

# ========== 📄 Raw Data ==========
elif page == "📄 Raw Data":
    st.title("📄 Raw Data")
    st.dataframe(data)
