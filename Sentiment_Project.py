import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(layout='wide', page_title="Food Sentiment Analysis")

# --- Custom CSS for Dynamic Banner ---
# Har baar refresh hone par 'source.unsplash.com' se nayi food image aayegi
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');
    
    .hero-section {
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                          url('https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=1500&q=80');
        background-size: cover;
        background-position: center;
        padding: 60px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 40px;
    }

    /* Random Image Trick: URL mein timestamp add karne se image change hoti hai */
    .hero-section {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                          url('https://loremflickr.com/1200/400/food,cooking');
    }

    .main-heading {
        font-family: 'Poppins', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
        color: #ffffff;
        text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
    }
    
    .sub-text {
        font-size: 1.5rem;
        opacity: 0.9;
    }

    /* Style for buttons and inputs */
    .stButton>button {
        background-color: #ff4e50 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 25px !important;
    }
    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🍔 Food Sentiment Analysis</h1>
        <p class="sub-text">AI based Taste & Review Recognition</p>
    </div>
    """, unsafe_allow_html=True)

# --- Logic ---
def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower()

# Model Loading
try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("Model file 'Sentiment_model.pkl' nahi mili! Please check the file name.")

# --- Sidebar ---
st.sidebar.title("🇮🇳 Project Dashboard")
st.sidebar.info("Analyze food reviews instantly.")

st.sidebar.header("📞 Contact")
st.sidebar.write("Support: +91 999999999")

# --- UI Layout ---
col_main, col_spacer = st.columns([2, 1])

with col_main:
    st.write("### 💬 Real-time Prediction")
    sample = st.text_input("Customer ka review yahan type karein...", placeholder="e.g. The pasta was amazing and fresh!")
    
    if st.button("Analyze Sentiment"):
        if sample:
            pred = model.predict([sample])
            prob = model.predict_proba([sample])
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred[0] == 0:
                    st.error("### Result: Negative 👎")
                    st.write("Customer is unhappy with the food.")
                else:
                    st.success("### Result: Positive 👍")
                    st.write("Great! Customer liked the food.")
            with res_col2:
                confidence = prob[0][0] if pred[0] == 0 else prob[0][1]
                st.metric("Confidence", f"{confidence*100:.2f}%")
        else:
            st.warning("Kuch toh likhiye!")

st.divider()

# --- Bulk Prediction ---
st.write("### 📂 Bulk Analysis (CSV/TXT)")
file = st.file_uploader("Upload review file", type=["csv", "txt"])

if file:
    df = pd.read_csv(file, names=["Review"])
    if st.button("Run Bulk Prediction", key="b2"):
        corpus = df.Review
        pred = model.predict(corpus)
        prob = np.max(model.predict_proba(corpus), axis=1)
        
        df['Result'] = pred
        df['Confidence'] = prob
        df['Result'] = df['Result'].map({0: 'Dislike 👎', 1: 'Like 👍'})
        
        st.dataframe(df, use_container_width=True)
