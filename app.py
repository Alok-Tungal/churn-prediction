import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Patch custom function (for safety)
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout and settings
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load
data = load_data()
model = load_model()

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ===================== 🏠 MAIN: Churn Prediction =====================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Enter customer details to predict churn likelihood:")

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
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    # Predict button
    if st.button("🔍 Predict Churn"):
        try:
            # Raw input DataFrame
            input_dict = {
                'gender': gender,
                'SeniorCitizen': SeniorCitizen,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
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
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges
            }
            input_df = pd.DataFrame([input_dict])

            # Encode categorical columns using one-hot encoding
            input_encoded = pd.get_dummies(input_df)

            # Align with model's expected input features
            expected_cols = model.feature_names_in_
            for col in expected_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0  # Add missing with 0

            input_encoded = input_encoded[expected_cols]  # Reorder columns

            # Predict
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# ===================== 📈 Insights & Graphs =====================
elif page == "📈 Insights & Graphs":
    st.title("📊 Churn Analysis")

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
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("### 🧠 Key Insights")
    st.markdown("""
    - Month-to-month contracts have highest churn rate.
    - Electronic check users churn more.
    - Customers with short tenure are more likely to churn.
    """)

# ===================== 📄 Raw Data =====================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset Preview")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")
