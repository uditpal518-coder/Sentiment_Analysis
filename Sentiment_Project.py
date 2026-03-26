import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(layout='wide', page_title="Food Sentiment Analysis", page_icon="🇮🇳")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');
    
    /* Hero Section */
    .hero-section {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                          url('https://loremflickr.com/1200/400/food,cooking');
        background-size: cover;
        background-position: center;
        padding: 60px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 40px;
    }

    .main-heading {
        font-family: 'Poppins', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Improvements */
    .stSidebar {
        background-color: #f8f9fa;
    }
    
    .sidebar-label {
        font-weight: bold;
        color: #ff4e50;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #ff4e50, #f9d423) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 25px !important;
        width: 100%;
    }
    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🍔 Food Sentiment Analysis</h1>
        <p style="font-size: 1.5rem; opacity: 0.9;">AI-Powered Review Classification System</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Content ---
# 1. Indian Flag Image
st.sidebar.markdown("<h1 style='text-align: center;'>🇮🇳</h1>", unsafe_allow_html=True)
# Agar aapke paas local image hai toh st.sidebar.image("Flag_of_India.jpg") use karein.
# Filhaal ek beautiful online link use kar raha hoon:
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", use_container_width=True)

st.sidebar.divider()

# 2. About Project
st.sidebar.title("🚀 About Project")
st.sidebar.info("""
Yeh ek Advanced Machine Learning project hai jo customer ke reviews ko analyze karke batata hai ki khana kaisa tha.
- **Goal:** Business decision making ko improve karna.
- **Accuracy:** Model is trained on 10,000+ restaurant reviews.
""")

# 3. Technical Stack (Libraries)
st.sidebar.title("🛠️ Tech Stack")
st.sidebar.markdown("""
* **Python** (Core Logic)
* **Streamlit** (UI Dashboard)
* **Scikit-Learn** (ML Algorithm)
* **Joblib** (Model Loading)
* **Pandas & Numpy** (Data Handling)
* **NLP / Regex** (Text Cleaning)
""")

# 4. Contact
st.sidebar.title("📞 Contact Us")
st.sidebar.success("📍 **AI Engineers at DUCAT**")
st.sidebar.write("📧 support@foodsentiment.in")
st.sidebar.write("📞 +91 999999999")

# --- ML Logic ---
def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower()

try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("⚠️ Model file 'Sentiment_model.pkl' nahi mili! Please check the directory.")

# --- Main UI Layout ---
st.write("### 💬 Real-time Prediction")
sample = st.text_area("Customer ka review yahan likhein...", placeholder="e.g. The service was slow but the food was delicious!")

if st.button("Analyze Sentiment"):
    if sample:
        pred = model.predict([sample])
        prob = model.predict_proba([sample])
        
        col1, col2 = st.columns(2)
        with col1:
            if pred[0] == 0:
                st.error("### Result: Negative 👎")
                st.write("Feedback: Customer ko khana pasand nahi aaya.")
            else:
                st.success("### Result: Positive 👍")
                st.write("Feedback: Customer khush hai!")
        with col2:
            confidence = prob[0][0] if pred[0] == 0 else prob[0][1]
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
    else:
        st.warning("Review box khali hai. Kuch type karein!")

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
