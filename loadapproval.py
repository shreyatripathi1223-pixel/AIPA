import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Setup the Page
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details below to check loan eligibility.")

# 2. Re-create the model (matching your notebook's logic)
# In a production app, you would normally load a pre-trained 'model.pkl' file.
def train_model():
    # Sample data matching your screenshot's structure
    data = {
        'Age': [33, 21, 57, 50, 21, 43, 30, 23, 41],
        'Income': [95457, 21105, 88970, 92604, 50622, 37833, 62531, 72585, 44272],
        'CreditScore': [349, 650, 569, 709, 305, 575, 506, 614, 536],
        'LoanApproved': [0, 0, 0, 1, 0, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    X = df[["Age", "Income", "CreditScore"]]
    y = df["LoanApproved"]
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

model = train_model()

# 3. Create Input Widgets in a Sidebar or Form
with st.form("prediction_form"):
    st.subheader("Applicant Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)
    
    submit = st.form_submit_button("Predict Approval Status")

# 4. Handle Prediction
if submit:
    # Prepare input for the model
    user_input = pd.DataFrame([[age, income, credit_score]], 
                              columns=["Age", "Income", "CreditScore"])
    
    prediction = model.predict(user_input)
    
    st.divider()
    if prediction[0] == 1:
        st.success("✅ Congratulations! The loan is likely to be **APPROVED**.")
    else:
        st.error("❌ Sorry, the loan application is likely to be **REJECTED**.")

# 5. Show training data (Optional)
if st.checkbox("Show Training Data Sample"):
    st.write("This model was trained on data similar to your Jupyter notebook:")
    st.table(pd.DataFrame({
        'Age': [33, 21, 57], 'Income': [95457, 21105, 88970], 
        'CreditScore': [349, 650, 569], 'Approved': [0, 0, 0]
    }))