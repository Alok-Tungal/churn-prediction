import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Dummy patch if your model uses custom preprocessing
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

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:  # Updated model file name
        return pickle.load(f)

# Load resources
data = load_data()
model = load_model()

# Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ======================= 🏠 Churn Prediction =======================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Enter customer details to predict churn likelihood:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
            Partner = st.selectbox('Partner', ['Yes', 'No'])
            Dependents = st.selectbox('Dependents', ['Yes', 'No'])
            tenure = st.slider('Tenure (months)', 0, 100, 12)
            PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
            MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
            InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        with col2:
            DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
            StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
            StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
            Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
            PaymentMethod = st.selectbox('Payment Method', [
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
            MonthlyCharges = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
            TotalCharges = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            input_data = pd.DataFrame([{
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
            }])

            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] * 100

                if prediction == 1:
                    st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
                else:
                    st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")

# ======================= 📈 Insights Tab =======================
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Insights & Visualizations")

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

    st.markdown("### 🧠 Key Business Insights")
    st.markdown("""
    - Month-to-month contracts have the highest churn.
    - Customers paying with electronic checks churn more.
    - Fiber optic users churn more than DSL users.
    """)

# ======================= 📄 Raw Data =======================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset Preview")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")
