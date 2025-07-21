import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Patch (in case model uses custom funcs)
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

data = load_data()
model = load_model()

# ✅ MANUALLY define feature names used while training
model_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'SeniorCitizen_No', 'SeniorCitizen_Yes',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# =============== 🏠 Prediction Page =====================
if page == "🏠 Churn Prediction":
    st.title("🔮 Predict Telecom Churn")
    st.markdown("Enter customer details below to check churn probability.")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
        Partner = st.selectbox("Partner", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.slider("Tenure (months)", 0, 100, 12)
        PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
        MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])

    with col2:
        InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        PaymentMethod = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    # Build Input DataFrame
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }])

    # Encode input
    input_encoded = pd.get_dummies(input_data)

    # Add any missing columns expected by model
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match model
    input_encoded = input_encoded[model_features]

    if st.button("🔍 Predict Churn"):
        try:
            pred = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0][1] * 100

            if pred == 1:
                st.error(f"⚠️ Likely to churn (Probability: {prob:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - prob:.1f}%)")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# =============== 📈 Insights =====================
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Visual Insights")

    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.bar_label(bars)
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    bars = ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

# =============== 📄 Raw Data =====================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")

