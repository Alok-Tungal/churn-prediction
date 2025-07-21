import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Patching custom function if needed
def ordinal_encode_func(df):
    return df

import sys
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout settings
st.set_page_config(page_title="📊 Telecom Churn Prediction", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model and scaler
@st.cache_resource
def load_advanced_model():
    with open('churn_pred.pkl', 'rb') as f:
        model, scaler = pickle.load(f)

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

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ===================== 🏠 Churn Prediction Page =====================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Train and test your model by entering customer details:")

    # Input UI
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

    # Prepare input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total],
        f'Contract_{contract}': [1],
        f'PaymentMethod_{payment}': [1],
        f'InternetService_{internet}': [1]
    })

    # Add missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    if st.button("🔍 Predict Churn"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100
        if pred == 1:
            st.error(f"⚠️ Likely to churn (Probability: {prob:.1f}%)")
        else:
            st.success(f"✅ Not likely to churn (Probability: {100 - prob:.1f}%)")

        # Feature importance
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

# ===================== 📈 Insights Page =====================
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Insights and Visualizations")

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

    st.markdown("### ✏️ Key Insights")
    st.markdown("""
    - Month-to-month customers show highest churn.
    - Electronic checks are associated with higher churn.
    - Long-tenure customers show lower churn.
    """)

# ===================== 📄 Raw Data Page =====================
elif page == "📄 Raw Data":
    st.title("📄 Raw Data Preview")
    st.dataframe(data)
    st.markdown(f"**Total Records:** {len(data)}")
