import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 🔧 Patch if custom functions were used during model training
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# --- Streamlit Page Config ---
st.set_page_config(page_title="📊 Churn Predictor", layout="wide")
sns.set(style='whitegrid')

# --- Load data for visualizations ---
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# --- Load model ---
@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# --- Load model columns to match feature count ---
@st.cache_resource
def load_model_columns():
    with open('model_columns.pkl', 'rb') as f:
        return pickle.load(f)

# --- Initialize Resources ---
data = load_data()
model = load_model()
model_columns = load_model_columns()

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights", "📄 Raw Data"])

# ========== 🏠 Churn Prediction ==========
if page == "🏠 Churn Prediction":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter a few customer details:")

    # 👇 Only Top Features Based on EDA & Feature Importance
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])

    # Input base
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'Contract': contract,
        'PaymentMethod': payment,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure correct column order
    input_encoded = input_encoded[model_columns]

    # Predict
    if st.button("🔍 Predict Churn"):
        try:
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# ========== 📈 Insights ==========
elif page == "📈 Insights":
    st.title("📊 Churn Insights")

    st.subheader("📌 Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    st.pyplot(fig)

    st.subheader("🧾 Churn by Contract Type")
    churn_by_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.bar(churn_by_contract.index, churn_by_contract.values, color='orange')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_by_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.barh(churn_by_payment.index, churn_by_payment.values, color='skyblue')
    st.pyplot(fig)

    st.markdown("### 📌 Key Takeaways:")
    st.markdown("""
    - Month-to-month contracts have the highest churn.
    - Electronic check users churn more.
    - Long-tenure customers tend to stay.
    """)

# ========== 📄 Raw Data ==========
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")
