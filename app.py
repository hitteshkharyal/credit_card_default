import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Credit Card Default Detection",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- HEADER SECTION ----------
st.markdown(
    """
    <div style='background-color:#0047AB;padding:15px;border-radius:10px;'>
        <h2 style='color:white;text-align:center;'>💳 Credit Card Default Detection System</h2>
        <p style='color:white;text-align:center;'>Predict the likelihood of a customer defaulting on their next payment using Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD MODEL ----------
model = joblib.load("model/trained_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------- SIDEBAR ----------
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About Project", "Developer Info"])

if page == "Home":
    st.header("Enter Customer Details for Prediction")

    col1, col2 = st.columns(2)

    with col1:
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", 10000, 1000000, 200000)
        sex = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level (1=Graduate, 2=University, 3=HighSchool, 4=Others)", [1, 2, 3, 4])
        marriage = st.selectbox("Marital Status (1=Married, 2=Single, 3=Others)", [1, 2, 3])
        age = st.slider("Age", 18, 75, 30)

    with col2:
        pay_0 = st.selectbox("Last Payment Status (PAY_0)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        bill_amt1 = st.number_input("Last Month Bill Amount", 0, 1000000, 50000)
        pay_amt1 = st.number_input("Last Month Payment Amount", 0, 1000000, 20000)

    st.markdown("---")

    if st.button("🔮 Predict Default"):
        # Arrange data
        input_data = np.array([[limit_bal, 1 if sex == "Male" else 2, education, marriage,
                                age, pay_0, 0, 0, 0, 0, 0,
                                bill_amt1, 0, 0, 0, 0, 0,
                                pay_amt1, 0, 0, 0, 0, 0, 0]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ The customer is **likely to DEFAULT** on the next payment.")
        else:
            st.success("✅ The customer is **not likely to default**.")

elif page == "About Project":
    st.header("📘 About the Project")
    st.write("""
    This Credit Card Default Detection System uses **Machine Learning** to predict the likelihood of a customer failing to make the next month’s credit card payment.

    ### 🧠 Model Details
    - Algorithm: Random Forest Classifier
    - Dataset: Default of Credit Card Clients (Kaggle)
    - Accuracy: ~80%
    - Features used include demographic and payment history information.

    ### ⚙️ Process Flow
    1. Data Preprocessing (scaling, cleaning)
    2. Model Training and Testing
    3. Streamlit UI for user input and prediction
    """)

elif page == "Developer Info":
    st.header("👨‍💻 Developer Information")
    st.write("""
    **Project by:** Hitesh Kumar  
    **Technology Used:** Python, Streamlit, Scikit-Learn  
    **IDE:** PyCharm  
    **Dataset:** Kaggle – Default of Credit Card Clients  
    """)

# ---------- FOOTER ----------
st.markdown(
    """
    <br><hr>
    <div style='text-align:center;'>
        <p style='color:gray;'> Credit Card Default Detection |</p>
    </div>
    """,
    unsafe_allow_html=True
)
