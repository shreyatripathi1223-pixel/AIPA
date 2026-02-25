import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import base64

# --- Background Setup ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# --- Original Application Code ---
st.title("Student PCA Dataset Analysis")

# Load the dataset
try:
    df = pd.read_csv("student_pca_dataset.csv")
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'student_pca_dataset.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

st.subheader("Original DataFrame (first 5 rows)")
st.dataframe(df.head())
st.write(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")

# Perform PCA
st.subheader("Principal Component Analysis (PCA)")
n_components = 2
pca = PCA(n_components=n_components)
# Assuming numeric data is pre-processed or only numeric columns are used
model = pca.fit_transform(df)

# Create a new DataFrame for PCA results
pca_df = pd.DataFrame(model, columns=["A1", "A2"])
st.write(f"PCA results reduced to {n_components} components.")
st.dataframe(pca_df.head())

# Optional: Display explained variance
st.subheader("Explained Variance Ratio")
st.write(pca.explained_variance_ratio_)
st.write(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")