import streamlit as st
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# --- TITLE ---
st.title("📉 Customer Churn Prediction App")
st.markdown("**Visualize, Analyze & Predict Customer Churn in One Place**")

# --- SIDEBAR UPLOAD ---
st.sidebar.header("🔍 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- LOAD DATA ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Ensure numeric conversion
    for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df.dropna(subset=['MonthlyCharges', 'TotalCharges', 'tenure'], inplace=True)

    # Convert to integer for visualization
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(int)
    df['TotalCharges'] = df['TotalCharges'].astype(int)
    df['tenure'] = df['tenure'].astype(int)

    return df

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")  # Make sure this file is in your project folder

# --- MAIN BODY ---
if uploaded_file:
    df = load_data(uploaded_file)

    # --- FILTER OPTIONS ---
    st.sidebar.subheader("Filter Data")
    if 'gender' in df.columns:
        gender = st.sidebar.multiselect("Select Gender", df['gender'].unique(), default=df['gender'].unique())
        df = df[df['gender'].isin(gender)]

    # --- DATA OVERVIEW ---
    st.markdown("### 📊 Exploratory Data Analysis")
    st.write("#### Preview of Dataset")
    st.dataframe(df.head())

    # --- CHURN PIE CHART ---
    if 'Churn' in df.columns:
        churn_fig = px.pie(df, names='Churn', title='Churn Distribution', hole=0.4)
        st.plotly_chart(churn_fig, use_container_width=True)

    # --- SCATTER PLOT ---
    st.subheader("📈 Tenure vs Monthly Charges")
    fig2 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn' if 'Churn' in df.columns else None,
                      size='TotalCharges', title='Tenure vs Monthly Charges')
    st.plotly_chart(fig2, use_container_width=True)

    # --- HEATMAP ---
    st.subheader("🔗 Correlation Heatmap")
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # --- PREDICTION ---
    st.markdown("### 🧠 Churn Prediction (Trained Model)")
    model = load_model()

    if st.checkbox("Show Prediction Form"):
        with st.form("prediction_form"):
            st.write("Enter customer details:")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

            submit = st.form_submit_button("Predict Churn")

        if submit:
            input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                                      columns=["tenure", "MonthlyCharges", "TotalCharges"])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.success(f"Prediction: **{'Churn' if prediction == 1 else 'No Churn'}**")
            st.info(f"Churn Probability: `{probability:.2%}`")

else:
    st.info("👈 Upload a dataset from the sidebar to begin.")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Alok Tungal's Churn App")



# model = joblib.load("C:/Users/Dell/Downloads/Weather Data/Weather Data/2021/streamlit_application/churn_model.pkl")

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import joblib

# import pandas as pd

# # Replace 'your_data.csv' with the actual file path
# df = pd.read_csv("Churn_data")

# # Make a copy of the DataFrame for processing
# data = df.copy()

# # Drop customerID as it's not useful for prediction
# data.drop('customerID', axis=1, inplace=True)

# # Convert TotalCharges to numeric (it might have spaces or invalid entries)
# data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# # Drop rows with missing values (after conversion)
# data.dropna(inplace=True)

# # Encode target column
# data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# # Encode categorical variables
# label_encoders = {}
# for col in data.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     label_encoders[col] = le

# # Features and target
# X = data.drop('Churn', axis=1)
# y = data['Churn']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model
# model_path = "/mnt/data/churn_model.pkl"
# joblib.dump(model, model_path)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, model.predict(X_test))
# model_path, accuracy
