import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Loan Predictor", page_icon="🏦")

# --- BACKGROUND IMAGE & COLORS ---
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://unsplash.com");
            background-attachment: fixed;
            background-size: cover;
        }}
        .stTextInput, .stNumberInput, .stSelectbox {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()

# --- 1. LOAD & CLEAN DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    # Missing data ko handle karna
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'train.csv' file nahi mili! Please check karein ki file same folder mein hai.")
    st.stop()

# --- 2. TRAINING THE AI MODEL ---
df = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 
         'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']]

# Convert text columns to numbers
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- 3. WEBSITE USER INTERFACE ---
st.title("🏦 Smart Bank Loan Predictor")
st.markdown("### *Fill in your details and let AI check your approval status.1`*")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married?", ["No", "Yes"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["No", "Yes"])

with col2:
    income = st.number_input("Monthly Income ($)", value=3000)
    co_income = st.number_input("Co-Applicant Income ($)", value=0)
    loan_amt = st.number_input("Loan Amount (In Thousands)", value=120)
    credit = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])

# Format inputs for model
gender_n = 1 if gender == "Male" else 0
married_n = 1 if married == "Yes" else 0
edu_n = 0 if education == "Graduate" else 1
self_n = 1 if self_employed == "Yes" else 0
credit_n = 1.0 if "Good" in credit else 0.0

if st.button("🚀 Check Loan Status"):
    user_data = [[gender_n, married_n, edu_n, self_n, income, co_income, loan_amt, credit_n]]
    result = model.predict(user_data)
    
    if result == 1:
        st.success("✅ Congratulations! There is a good chance that your loan will be approved.")
        st.balloons()
    else:
        st.error("⚠️ We're sorry, but there is a risk that your loan may be rejected.")
