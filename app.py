# import streamlit as st

# # Set page config
# st.set_page_config(page_title="Alok Mahadev Tungal - Resume", layout="wide")

# # Header
# col1, col2, col3 = st.columns([1, 2, 1])  # [left, center, right]
# with col2:
#     st.title("Alok Mahadev Tungal")
#     with 
#     st.subheader("Data Analyst Trainee")
#     st.write("📍 Bengaluru, Karnataka | 📧 aloktungal39@gmail.com | 📞 +91 7483471059")
#     st.write("[LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207) | [GitHub](https://github.com/Alok-Tungal)")

# st.markdown("---")

# # Education
# st.header("🎓 Education")
# st.write("**B.E in Electronics and Communication Engineering**")
# st.write("KLE College of Engineering and Technology (2022 – 2025), Chikodi, Karnataka")

# # Internships
# st.header("💼 Internships")
# st.write("**Jetking Belagavi** (Nov 2023 – Dec 2023)")
# st.markdown("""
# - Assembled, troubleshot, and maintained computer systems.  
# - Diagnosed hardware issues, performed upgrades and developed expertise in hardware and networking.
# """)

# st.write("**Innomatics Research Lab, Bangalore** (Feb 2025 – May 2025)")
# st.markdown("""
# - Cleaned, transformed, and interpreted data using Python and Pandas.  
# - Applied web scraping with Selenium and BeautifulSoup.  
# - Built SQL queries and created Power BI dashboards for business reporting.
# """)

# # Projects
# st.header("📊 Projects")

# st.write("**IMDB MOVIES - Exploratory Data Analysis (18 Apr – 30 Apr 2025)**")
# st.markdown("""
# - Scraped 700 rows of movie data using Selenium and WebChromium.  
# - Cleaned and engineered features like genre, vote count, runtime.  
# - Visualized insights using Matplotlib and Seaborn.  
# - Derived trends in ratings, genre popularity, and release year.
# """)

# st.write("**IMDB MOVIES - Power BI Dashboard (03 Jun – 10 Jun 2025)**")
# st.markdown("""
# - Imported, cleaned and modeled data in Power BI.  
# - Built calculated columns and DAX measures.  
# - Created interactive dashboards to visualize trends in ratings, genres, and years.
# """)

# # Skills
# st.header("🛠️ Technical Skills")
# st.markdown("""
# - **Programming:** Python (Core, OOP, Exception Handling, Performance Tuning)  
# - **Libraries:** Pandas, NumPy, Matplotlib  
# - **Database:** SQL (MySQL)  
# - **Tools:** Power BI, Selenium, BeautifulSoup  
# """)

# # Soft Skills
# st.header("🤝 Soft Skills")
# st.markdown("""
# - Analytical Thinking  
# - Clear Communication  
# - Problem Solving
# """)

# # Certificates
# st.header("📜 Certificates")
# st.write("✅ Python for Data Science – NPTEL Jan-Feb 2024")
# st.write("✅ Cyber Security and Privacy – NPTEL Jul-Oct 2023")

# # Footer
# st.markdown("---")
# st.caption("Built with ❤️ by Alok Mahadev Tungal using Streamlit")



# import streamlit as st
# import os

# # =================== Page Config =====================
# st.set_page_config(page_title="Alok Mahadev Tungal - Resume", layout="wide")

# # =================== Sidebar =========================
# with st.sidebar:
#     if os.path.exists("alok_profile.jpg"):
#         st.image("alok_profile.jpg", width=180)
#     else:
#         st.image("https://via.placeholder.com/180", caption="Alok Tungal", width=180)

#     st.title("Alok Mahadev Tungal")
#     st.write("📍 Bengaluru, Karnataka")
#     st.write("📧 aloktungal39@gmail.com")
#     st.write("📞 +91 7483471059")

#     st.markdown("### 🔗 Connect")
#     st.markdown("[LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[GitHub](https://github.com/Alok-Tungal)")

#     # Download resume
#     if os.path.exists("Alok_Tungal_Resume.pdf"):
#         with open("Alok_Tungal_Resume.pdf", "rb") as pdf_file:
#             PDFbyte = pdf_file.read()
#         st.download_button(label="📥 Download Resume",
#                            data=PDFbyte,
#                            file_name="Alok_Tungal_Resume.pdf",
#                            mime='application/octet-stream')

# # =================== Main Content =====================
# st.title("💼 Resume of Alok Mahadev Tungal")
# st.markdown("---")

# # Education
# col1, col2 = st.columns([2, 5])
# with col1:
#     st.header("🎓 Education")
# with col2:
#     st.write("**B.E in Electronics & Communication Engineering** (2022 – 2025)")
#     st.write("KLE College of Engineering and Technology, Chikodi")

# # Internships
# st.markdown("---")
# st.header("💼 Internships")

# st.markdown("#### 🛠 Jetking Belagavi *(Nov 2023 – Dec 2023)*")
# st.markdown("""
# - Assembled, troubleshot, and maintained computer systems  
# - Diagnosed hardware issues and performed upgrades  
# - Gained hands-on experience in networking and diagnostics
# """)

# st.markdown("#### 📊 Innomatics Research Lab *(Feb 2025 – May 2025)*")
# st.markdown("""
# - Cleaned and transformed datasets using Python and Pandas  
# - Web scraped structured data using Selenium & BeautifulSoup  
# - Built SQL queries and designed dashboards in Power BI
# """)

# # Projects
# st.markdown("---")
# st.header("📈 Projects")

# st.markdown("#### 🎬 IMDB Movies – EDA")
# st.markdown("""
# - Scraped 700+ movie entries using Selenium and Chromium  
# - Cleaned and engineered features like genre, votes, runtime  
# - Used Pandas, NumPy, Matplotlib & Seaborn for EDA  
# - Uncovered insights on genres, ratings, and release years
# """)

# st.markdown("#### 📊 IMDB Movies – Power BI")
# st.markdown("""
# - Loaded and transformed IMDb data using Power Query  
# - Created calculated columns and DAX measures  
# - Built interactive dashboards with filters and visual insights  
# - Highlighted genre popularity, rating patterns, and year trends
# """)

# # Skills
# st.markdown("---")
# st.header("🧠 Skills")

# st.markdown("""
# - **Languages:** Python (Core, OOP, Exception Handling)  
# - **Libraries:** Pandas, NumPy, Matplotlib  
# - **Database:** SQL (MySQL)  
# - **Tools:** Power BI, Selenium, BeautifulSoup  
# - **Soft Skills:** Analytical Thinking, Communication, Problem Solving
# """)

# # Certifications
# st.markdown("---")
# st.header("📜 Certifications")
# st.write("✅ Python for Data Science – *NPTEL Jan–Feb 2024*")
# st.write("✅ Cyber Security and Privacy – *NPTEL Jul–Oct 2023*")

# # Footer
# st.markdown("---")
# st.caption("🚀 Created by Alok Mahadev Tungal | Made with ❤️ in Streamlit")


# import streamlit as st

# st.title("My Name ")
# st.title(":red[Royal Alok]:sunglasses:")

# st.header ("Banglore City",divider=True)

# multi=''' If you have a help Alok will,\n
#             help you'''
# st.markdown(multi)

# st.badge("Alok")
# st.divider()


# st.html(
#     "<p><span style='text-decoration: line-through double blue;'>Grapes</span>!</p>"
# )



# import streamlit as st

# st.title("This is a title")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")

# import streamlit as st

# st.caption("Hi I am Alok.")
# st.caption("The BMW Car Lover  :sunglasses:")

# import streamlit as st

# code = '''def hello():
#     print("Hello, Streamlit!")'''
# st.code(code, language="python")

# code='lambda x:x*x'
# st.code(code,language='python')

# import streamlit as st
# import os
# st.set_page_config(page_title="Alok Mahadev Tungal",layout='wide')

# with st.container():
#     if os.path.exists("E:\OneDrive\Pictures\my pic.M.jpg"):
#         st.image("E:\OneDrive\Pictures\my pic.M.jpg", width=180)
#     else:
#         st.image("https://via.placeholder.com/180", caption="Alok Tungal", width=180)

#     st.title("Alok Mahadev Tungal")
#     st.write("📍 Bengaluru, Karnataka")
#     st.write("📧 aloktungal39@gmail.com")
#     st.write("📞 +91 7483471059")

#     st.markdown("### 🔗 Connect")
#     st.markdown("[LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[GitHub](https://github.com/Alok-Tungal)")

# Download resume

import streamlit as st
import os
st.set_page_config(page_title='Alok Mahadev Tungal')

# with st.container():
#     if os.path.exists("E:\OneDrive\Pictures\my pic.M.jpg"):
#         st.image("E:\OneDrive\Pictures\my pic.M.jpg", width=180)
#     else:
#         st.image("https://via.placeholder.com/180", caption="Alok Tungal", width=180)
# st.title("Alok Mahadev Tungal")
# st.write("**Data Scientist Train**")
# st.write("📍 Bengaluru,Karntaka")
# st.write("📞 7483471059")
# st.write("📩 aloktungal39@gmail.com")

# st.markdown("### 🔗 connect")
# st.markdown("[Linkedin](https://linkedin.com/in/alok-tungal-8a5547207)")
# st.markdown("[Github](https://github.com/Alok-Tungal)")



# import streamlit as st

# Create three columns
# col1, col2, col3 = st.columns([1, 2, 1])  # [left, center, right]

# Write content in the center column
# with col2:
#     st.title("Alok Mahadev Tungal")
#     st.write("**Data Scientist Trainee**")
#     st.write("📍 Bengaluru, Karnataka")
#     st.write("📞 7483471059")
#     st.write("📩 aloktungal39@gmail.com")

#     st.markdown("### 🔗 Connect")
#     st.markdown("[LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[GitHub](https://github.com/Alok-Tungal)")


# import streamlit as st
# from PIL import Image

# # Set page configuration
# st.set_page_config(page_title="Alok Tungal | Resume", layout="wide")

# # Load your photo
# photo = Image.open("E:\OneDrive\Pictures\my pic.M.jpg")

# # --- Layout ---
# col1, col2 = st.columns([1, 2.5])

# # LEFT SIDE - PHOTO
# with col1:
#     st.image(photo, width=200, caption="Alok Mahadev Tungal")
#     st.markdown("### 📍 Bengaluru, Karnataka")
#     st.markdown("📧 aloktungal39@gmail.com")
#     st.markdown("📞 +91 74834 71059")
#     st.markdown("[🔗 LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[💻 GitHub](https://github.com/Alok-Tungal)")

# # RIGHT SIDE - DETAILS
# with col2:
#     st.title("Alok Mahadev Tungal")
#     st.subheader("📊 Data Analyst Trainee")

#     # --- Education ---
#     st.markdown("### 🎓 Education")
#     st.markdown("**B.E in Electronics and Communication**")
#     st.markdown("KLE College of Engineering and Technology | 2022 – 2025 | Chikodi, Karnataka")

#     # --- Projects ---
#     st.markdown("### 💼 Projects")
    
#     st.markdown("#### 1. 🎬 IMDB MOVIES - EDA (Apr 2025)")
#     st.markdown("""
#     - Scraped 700 rows using Selenium & WebChromium.
#     - Cleaned with Pandas/NumPy, feature engineering.
#     - Visualization using Matplotlib and Seaborn.
#     - Extracted trends in ratings, genres, and release years.
#     """)

#     st.markdown("#### 2. 🛒 Grocery Store Management - SQL (May 2025)")
#     st.markdown("""
#     - Developed inventory & billing system using MySQL and Python.
#     - Implemented product handling, billing logic, and discount management.
#     """)

#     st.markdown("#### 3. 📈 IMDB MOVIES - Power BI (Jun 2025)")
#     st.markdown("""
#     - Loaded data in Power BI, created interactive dashboards.
#     - Analyzed genres, ratings, and yearly trends.
#     """)

#     # --- Internships ---
#     st.markdown("### 👨‍💻 Internships")
    
#     st.markdown("#### 🔧 Jetking Belagavi (Nov–Dec 2023)")
#     st.markdown("""
#     - Built and diagnosed computer systems.
#     - Worked on networking and hardware installations.
#     """)

#     st.markdown("#### 🧪 Innomatics Research Lab (Feb–May 2025)")
#     st.markdown("""
#     - Hands-on Data Analysis with Python & Pandas.
#     - Web scraping using Selenium and BeautifulSoup.
#     - Dashboards using Power BI.
#     """)

#     # --- Skills ---
#     st.markdown("### 🧠 Skills")
#     st.markdown("**Languages:** Python, SQL")
#     st.markdown("**Tools:** Pandas, NumPy, Power BI, Web Scraping, Statistics")
#     st.markdown("**Visualization:** Matplotlib, Seaborn")

#     # --- Certifications ---
#     st.markdown("### 📜 Certificates")
#     st.markdown("- Python for Data Science (NPTEL Jan–Feb 2024)")
#     st.markdown("- Cyber Security and Privacy (NPTEL Jul–Oct 2023)")


# import streamlit as st
# from PIL import Image

# # Set page config
# st.set_page_config(page_title="Alok Tungal | Resume", layout="wide")

# # Load profile photo
# photo = Image.open("E:\OneDrive\Pictures\my pic.M.jpg")  # Make sure this image is in the same directory

# # Create layout with 2 columns — main content and image at the edge
# left, right = st.columns([4, 1])  # Wider left for content, narrow right for photo

# # --- RIGHT SIDE PHOTO ---
# with right:
#     st.image(photo, width=180, caption="Alok Mahadev Tungal")

# # --- LEFT SIDE RESUME CONTENT ---
# with left:
#     st.title("Alok Mahadev Tungal")
#     st.subheader("📊 Data Analyst Trainee")

#     # Contact
#     st.markdown("📧 aloktungal39@gmail.com")
#     st.markdown("📞 +91 74834 71059")
#     st.markdown("📍 Bengaluru, Karnataka")
#     st.markdown("[🔗 LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[💻 GitHub](https://github.com/Alok-Tungal)")

#     # Education
#     st.markdown("### 🎓 Education")
#     st.markdown("**B.E in Electronics and Communication**")
#     st.markdown("KLE College of Engineering and Technology, Chikodi (2022 – 2025)")

#     # Projects
#     st.markdown("### 💼 Projects")

#     st.markdown("#### 🎬 IMDB MOVIES - EDA")
#     st.markdown("""
#     - Scraped 700+ rows using Selenium & WebChromium  
#     - Cleaned data using Pandas and NumPy  
#     - Engineered features: genre, votes, runtime  
#     - Visualized trends using Matplotlib & Seaborn  
#     - Extracted insights on genres, ratings, and release years  
#     """)

#     st.markdown("#### 🛒 Grocery Store Management - SQL")
#     st.markdown("""
#     - Inventory tracking & billing system using MySQL  
#     - Stock updates, price calc, and discount logic  
#     - Handled real retail scenarios with Python & SQL  
#     """)

#     st.markdown("#### 📈 IMDB MOVIES - Power BI")
#     st.markdown("""
#     - Imported IMDb data into Power BI  
#     - Cleaned and modeled using Power Query  
#     - Built dashboards with DAX, slicers, genre insights  
#     """)

#     # Internships
#     st.markdown("### 👨‍💻 Internships")

#     st.markdown("#### 🔧 Jetking Belagavi (Nov–Dec 2023)")
#     st.markdown("""
#     - Built & diagnosed computer systems  
#     - Worked on networking & hardware upgrades  
#     """)

#     st.markdown("#### 🧪 Innomatics Research Lab (Feb–May 2025)")
#     st.markdown("""
#     - Data Analysis with Python & Pandas  
#     - Web scraping using Selenium + BeautifulSoup  
#     - Dashboarding in Power BI, SQL data queries  
#     """)

#     # Skills
#     st.markdown("### 🧠 Skills")
#     st.markdown("""
#     - **Languages:** Python, SQL  
#     - **Tools:** Pandas, NumPy, Power BI, Statistics, Web Scraping  
#     - **Visualization:** Matplotlib, Seaborn  
#     """)

#     # Certifications
#     st.markdown("### 📜 Certifications")
#     st.markdown("""
#     - Python for Data Science — *NPTEL Jan–Feb 2024*  
#     - Cyber Security & Privacy — *NPTEL Jul–Oct 2023*  
#     ""DETAILS
# with col2:
#     st.title("Alok Mahadev Tungal")
#     st.subheader("📊 Data Analyst Trainee")

#     # --- Education ---
#     st.markdown("### 🎓 Education")
#     st.markdown("**B.E in Electronics and Communication**")
#     st.markdown("KLE College of Engineering and Technology | 2022 – 2025 | Chikodi, Karnataka")

#     # --- Projects ---
#     st.markdown("### 💼 Projects")
    
#     st.markdown("#### 1. 🎬 IMDB MOVIES - EDA (Apr 2025)")
#     st.markdown("""
#     - Scraped 700 rows using Selenium & WebChromium.
#     - Cleaned with Pandas/NumPy, feature engineering.
#     - Visualization using Matplotlib and Seaborn.
#     - Extracted trends in ratings, genres, and release years.
#     """)

#     st.markdown("#### 2. 🛒 Grocery Store Management - SQL (May 2025)")
#     st.markdown("""
#     - Developed inventory & billing system using MySQL and Python.
#     - Implemented product handling, billing logic, and discount management.
#     """)

#     st.markdown("#### 3. 📈 IMDB MOVIES - Power BI (Jun 2025)")
#     st.markdown("""
#     - Loaded data in Power BI, created interactive dashboards.
#     - Analyzed genres, ratings, and yearly trends.
#     """)

#     # --- Internships ---
#     st.markdown("### 👨‍💻 Internships")
    
#     st.markdown("#### 🔧 Jetking Belagavi (Nov–Dec 2023)")
#     st.markdown("""
#     - Built and diagnosed computer systems.
#     - Worked on networking and hardware installations.
#     """)

#     st.markdown("#### 🧪 Innomatics Research Lab (Feb–May 2025)")
#     st.markdown("""
#     - Hands-on Data Analysis with Python & Pandas.
#     - Web scraping using Selenium and BeautifulSoup.
#     - Dashboards using Power BI.
#     """)

#     # --- Skills ---
#     st.markdown("### 🧠 Skills")
#     st.markdown("**Languages:** Python, SQL")
#     st.markdown("**Tools:** Pandas, NumPy, Power BI, Web Scraping, Statistics")
#     st.markdown("**Visualization:** Matplotlib, Seaborn")

#     # --- Certifications ---
#     st.markdown("### 📜 Certificates")
#     st.markdown("- Python for Data Science (NPTEL Jan–Feb 2024)")
#     st.markdown("- Cyber Security and Privacy (NPTEL Jul–Oct 2023)")


# import streamlit as st
# from PIL import Image

# # Set page config
# st.set_page_config(page_title="Alok Tungal | Resume", layout="wide")

# # Load profile photo
# photo = Image.open("E:\OneDrive\Pictures\my pic.M.jpg")  # Make sure this image is in the same directory

# # Create layout with 2 columns — main content and image at the edge
# left, right = st.columns([4, 1])  # Wider left for content, narrow right for photo

# # --- RIGHT SIDE PHOTO ---
# with right:
#     st.image(photo, width=180, caption="Alok Mahadev Tungal")

# # --- LEFT SIDE RESUME CONTENT ---
# with left:
#     st.title("Alok Mahadev Tungal")
#     st.subheader("📊 Data Analyst Trainee")

#     # Contact
#     st.markdown("📧 aloktungal39@gmail.com")
#     st.markdown("📞 +91 74834 71059")
#     st.markdown("📍 Bengaluru, Karnataka")
#     st.markdown("[🔗 LinkedIn](https://linkedin.com/in/alok-tungal-8a5547207)")
#     st.markdown("[💻 GitHub](https://github.com/Alok-Tungal)")

#     # Education
#     st.markdown("### 🎓 Education")
#     st.markdown("**B.E in Electronics and Communication**")
#     st.markdown("KLE College of Engineering and Technology, Chikodi (2022 – 2025)")

#     # Projects
#     st.markdown("### 💼 Projects")

#     st.markdown("#### 🎬 IMDB MOVIES - EDA")
#     st.markdown("""
#     - Scraped 700+ rows using Selenium & WebChromium  
#     - Cleaned data using Pandas and NumPy  
#     - Engineered features: genre, votes, runtime  
#     - Visualized trends using Matplotlib & Seaborn  
#     - Extracted insights on genres, ratings, and release years  
#     """)

#     st.markdown("#### 🛒 Grocery Store Management - SQL")
#     st.markdown("""
#     - Inventory tracking & billing system using MySQL  
#     - Stock updates, price calc, and discount logic  
#     - Handled real retail scenarios with Python & SQL  
#     """)

#     st.markdown("#### 📈 IMDB MOVIES - Power BI")
#     st.markdown("""
#     - Imported IMDb data into Power BI  
#     - Cleaned and modeled using Power Query  
#     - Built dashboards with DAX, slicers, genre insights  
#     """)

#     # Internships
#     st.markdown("### 👨‍💻 Internships")

#     st.markdown("#### 🔧 Jetking Belagavi (Nov–Dec 2023)")
#     st.markdown("""
#     - Built & diagnosed computer systems  
#     - Worked on networking & hardware upgrades  
#     """)

#     st.markdown("#### 🧪 Innomatics Research Lab (Feb–May 2025)")
#     st.markdown("""
#     - Data Analysis with Python & Pandas  
#     - Web scraping using Selenium + BeautifulSoup  
#     - Dashboarding in Power BI, SQL data queries  
#     """)

#     # Skills
#     st.markdown("### 🧠 Skills")
#     st.markdown("""
#     - **Languages:** Python, SQL  
#     - **Tools:** Pandas, NumPy, Power BI, Statistics, Web Scraping  
#     - **Visualization:** Matplotlib, Seaborn  
#     """)

#     # Certifications
#     st.markdown("### 📜 Certifications")
#     st.markdown("""
#     - Python for Data Science — *NPTEL Jan–Feb 2024*  
#     - Cyber Security & Privacy — *NPTEL Jul–Oct 2023*  
#     """)

# C:\Users\Dell\Downloads\Weather Data\Weather Data\2021\streamlit_application\streamlit_env\Scripts\activate
# pip install plotly


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# # --- TITLE ---
# st.title("📉 Customer Churn Prediction App")
# st.markdown("**Visualize, Analyze & Predict Customer Churn in One Place**")

# # --- SIDEBAR ---
# st.sidebar.header("🔍 Upload Your Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# # --- LOAD DATA ---
# @st.cache_data
# def load_data(file):
#     df = pd.read_csv(file)
#     return df

# if uploaded_file:
#     df = load_data(uploaded_file)
    
#     # --- FILTER ---
#     st.sidebar.subheader("Filter Data")
#     gender = st.sidebar.multiselect("Select Gender", df['gender'].unique(), default=df['gender'].unique())
#     df = df[df['gender'].isin(gender)]

#     st.markdown("### 📊 Exploratory Data Analysis")

#     # --- CHURN DISTRIBUTION ---
#     churn_fig = px.pie(df, names='Churn', title='Churn Distribution', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
#     st.plotly_chart(churn_fig, use_container_width=True)

#     # --- CUSTOMER TENURE ---
#     st.subheader("📈 Tenure vs Monthly Charges")
#     fig2 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', size='TotalCharges', title='Tenure vs Monthly Charges')
#     st.plotly_chart(fig2, use_container_width=True)

#     # --- CORRELATION HEATMAP ---
#     st.subheader("🔗 Correlation Heatmap")
#     num_df = df.select_dtypes(include='number')
#     corr = num_df.corr()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
#     st.pyplot(fig)

#     # --- PREDICTION SECTION ---
#     st.markdown("### 🧠 Churn Prediction (using saved model)")
    
#     if st.checkbox("Show Prediction Form"):
#         with st.form("prediction_form"):
#             st.write("Enter customer details:")
#             tenure = st.slider("Tenure (months)", 0, 72, 12)
#             monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
#             total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
#             submit = st.form_submit_button("Predict Churn")

#         if submit:
#             # Example model input format
#             model_input = pd.DataFrame([[tenure, monthly_charges, total_charges]],
#                                        columns=["tenure", "MonthlyCharges", "TotalCharges"])
#             model = joblib.load("churn_model.pkl")  # your trained model
#             pred = model.predict(model_input)[0]
#             st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")

# else:
#     st.info("👈 Upload a dataset from the sidebar to begin.")

# # --- FOOTER ---
# st.markdown("---")
# st.caption("Built with ❤️ using Streamlit | Alok Tungal's Churn App")




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
