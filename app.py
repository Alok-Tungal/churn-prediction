import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Streamlit config
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load Model
@st.cache_resource
def load_model():
    with open('churn_pred.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load resources
data = load_data()
model = load_model()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ======================= 🏠 MAIN: Churn Prediction =======================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Enter customer details to predict churn likelihood.")

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

    # Raw input for model pipeline
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total],
        'Contract': [contract],
        'PaymentMethod': [payment],
        'InternetService': [internet]
    })

    # Prediction
    if st.button("🔍 Predict Churn"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

            # Feature Importance (if supported)
            if hasattr(model, 'named_steps') and hasattr(model.named_steps[list(model.named_steps)[-1]], 'feature_importances_'):
                st.subheader("📊 Feature Importance (Top 5)")
                final_model = model.named_steps[list(model.named_steps)[-1]]
                try:
                    importances = final_model.feature_importances_
                    st.info("Feature importance is based on the final model.")
                    # You may need the actual column names post-preprocessing for accurate mapping
                except:
                    st.warning("Feature importances not available.")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

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
    - Customers with fiber optic internet churn more.
    """)

# ======================= 📄 Raw Data =======================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset Preview")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")

