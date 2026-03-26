import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(layout='wide', page_title="Food Sentiment Analysis", page_icon="🍔")

# --- Custom CSS for Dynamic Banner & About Section ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');
    
    /* Dynamic Hero Section with Random Food Image */
    .hero-section {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                          url('https://loremflickr.com/1600/500/food,restaurant,pizza');
        background-size: cover;
        background-position: center;
        padding: 80px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        border-bottom: 5px solid #ff4e50;
    }

    .main-heading {
        font-family: 'Poppins', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 20px rgba(0,0,0,0.7);
    }
    
    .sub-text {
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.9;
        letter-spacing: 1px;
    }

    /* About Section Styling */
    .about-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #f9d423;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #ff4e50, #f9d423) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 12px !important;
        transition: 0.3s;
    }
    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🍲 Food Sentiment Analysis</h1>
        <p class="sub-text">Smart AI for Restaurant Reviews & Customer Feedback</p>
    </div>
    """, unsafe_allow_html=True)

# --- Logic ---
def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower()

# Model Loading
try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("Error: 'Sentiment_model.pkl' file missing!")

# --- Layout: Two Columns (Prediction & About) ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔍 Real-time Sentiment Predictor")
    sample = st.text_area("Review Paste Karein:", placeholder="Example: The biryani was spicy and authentic, loved it!", height=150)
    
    if st.button("Analyze Now"):
        if sample:
            pred = model.predict([sample])
            prob = model.predict_proba([sample])
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred[0] == 0:
                    st.error("### Result: Negative 👎")
                else:
                    st.success("### Result: Positive 👍")
            with res_col2:
                confidence = prob[0][0] if pred[0] == 0 else prob[0][1]
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
        else:
            st.warning("Please enter some text to analyze.")

with col_right:
    st.markdown(f"""
        <div class="about-card">
            <h3>📖 About This Project</h3>
            <p>Yeh project Machine Learning ka use karke customer reviews ko analyze karta hai. 
            Iska maksad restaurants ko yeh samjhana hai ki unka khana logon ko kaisa lag raha hai.</p>
            <hr>
            <strong>Features:</strong>
            <ul>
                <li>NLP (Natural Language Processing)</li>
                <li>Real-time Probability Score</li>
                <li>Bulk Data Processing</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar style contact info inside main page
    st.info("💡 **Tip:** Clear text results in higher confidence scores.")

st.divider()

# --- Bulk Prediction ---
st.write("### 📂 Bulk Analysis (Upload CSV)")
file = st.file_uploader("", type=["csv"])

if file:
    df = pd.read_csv(file, names=["Review"])
    if st.button("Process Bulk Data", key="bulk_btn"):
        # Assuming the model can handle a list/Series of strings
        df['Result'] = model.predict(df['Review'])
        # Map values
        df['Sentiment'] = df['Result'].map({0: 'Dislike 👎', 1: 'Like 👍'})
        st.dataframe(df[['Review', 'Sentiment']], use_container_width=True)
