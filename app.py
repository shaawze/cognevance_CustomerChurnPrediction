
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
bundle = joblib.load("final_churn_model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the service.")

st.divider()

st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (Months)", min_value=0, max_value=72, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button("Predict Customer Churn")

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probabilities = model.predict_proba(customer)[0]
    yes_index = list(model.classes_).index("Yes")
    churn_probability = probabilities[yes_index]

    prediction = (
        "Yes" if churn_probability >= threshold else "No"
    )

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2%}"
    )

    if prediction == "Yes":
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

    st.caption(f"Decision threshold: {threshold:.0%}")
