import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(layout='wide', page_title="Food Sentiment Analysis")

# --- Custom CSS for the Heading ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');
    
    .main-heading {
        font-family: 'Poppins', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ff4e50 0%, #f9d423 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .underline {
        width: 150px;
        height: 6px;
        background: #ff4e50;
        margin: 10px auto 30px auto;
        border-radius: 10px;
    }
    </style>
    
    <h1 class="main-heading">Food Sentiment Analysis</h1>
    <div class="underline"></div>
    """, unsafe_allow_html=True)

# --- Logic ---
def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower() # Thoda fix kiya: [^a-zA-Z] better hai

# Model loading (ensure file exists)
try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("Model file 'Sentiment_model.pkl' nahi mili!")

# --- Sidebar ---
# st.sidebar.image("Flag_of_India.jpg") # Make sure image is in same folder
st.sidebar.title("🇮🇳 About Project")
st.sidebar.info("Prediction of Sentiment (Negative or Positive) for food reviews using Machine Learning.")

st.sidebar.title("📞 Contact Us")
st.sidebar.write("Phone: +91 999999999")

st.sidebar.title("👥 About Us")
st.sidebar.write("We are a group of AI Engineers at **DUCAT**.")

# --- Main UI ---
st.write("### 📝 Enter Review")
sample = st.text_input("Customer review yahan likhein...", placeholder="Example: The pizza was delicious!")

if st.button("Predict"):
    if sample:
        # Prediction logic
        pred = model.predict([sample])
        prob = model.predict_proba([sample])
        
        col1, col2 = st.columns(2)
        with col1:
            if pred[0] == 0:
                st.error(f"### Result: Neg 👎")
            else:
                st.success(f"### Result: Pos 👍")
        with col2:
            confidence = prob[0][0] if pred[0] == 0 else prob[0][1]
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
    else:
        st.warning("Please enter a review first!")

st.divider()

# --- Bulk Prediction ---
st.write("### 📂 Bulk Prediction")
file = st.file_uploader("Select CSV or TXT file", type=["csv", "txt"])

if file:
    df = pd.read_csv(file, names=["Review"])
    if st.button("Run Bulk Prediction", key="b2"):
        corpus = df.Review
        pred = model.predict(corpus)
        prob = np.max(model.predict_proba(corpus), axis=1)
        
        df['Sentiment'] = pred
        df['Confidence'] = prob
        df['Sentiment'] = df['Sentiment'].map({0: 'Dislike 👎', 1: 'Like 👍'})
        
        st.dataframe(df, use_container_width=True)
