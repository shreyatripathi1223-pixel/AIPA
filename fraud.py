import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up the app title
st.title("Fraud Detection System")

# 1. Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("fraud_rf.csv")
    return df

try:
    df = load_data()
    st.write("### Dataset Preview", df.head())

    # 2. Train the model (Mirroring your Notebook logic)
    X = df[["Amount", "Time", "LocationRisk"]]
    y = df["Fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Display accuracy
    accuracy = model.score(X_test, y_test)
    st.sidebar.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    # 3. User Input for Prediction
    st.write("### Predict New Transaction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0, value=5000)
    with col2:
        time = st.number_input("Time", min_value=0, value=10)
    with col3:
        location_risk = st.number_input("Location Risk (0-5)", min_value=0, max_value=5, value=2)

    # Prediction Button
    if st.button("Check for Fraud"):
        input_data = [[amount, time, location_risk]]
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("Warning: This transaction is likely FRAUDULENT!")
        else:
            st.success("This transaction appears to be LEGITIMATE.")

except FileNotFoundError:
    st.error("Dataset 'fraud_rf.csv' not found. Please ensure the file is in the same directory.")