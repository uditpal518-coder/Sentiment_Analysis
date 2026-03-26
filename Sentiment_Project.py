import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(layout='wide', page_title="Food Sentiment Analysis", page_icon="🇮🇳")

# --- Custom Responsive CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');
    
    /* Universal Box Sizing */
    * { font-family: 'Poppins', sans-serif; }

    /* Hero Section - Fully Responsive */
    .hero-section {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                          url('https://loremflickr.com/1200/400/food,cooking');
        background-size: cover;
        background-position: center;
        padding: clamp(30px, 8vw, 80px); /* Dynamic padding */
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        width: 100%;
    }

    .main-heading {
        font-size: clamp(2rem, 10vw, 4.5rem); /* Text scales with screen size */
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
    }
    
    .sub-text {
        font-size: clamp(1rem, 3vw, 1.5rem);
        opacity: 0.9;
        margin-top: 10px;
    }

    /* Sidebar Image Scaling */
    [data-testid="stSidebar"] img {
        max-width: 150px;
        margin: 0 auto;
        display: block;
        border-radius: 10px;
    }

    /* Responsive Button */
    .stButton>button {
        background: linear-gradient(to right, #ff4e50, #f9d423) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
        width: 100%;
        transition: 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 78, 80, 0.3);
    }

    /* Dataframe responsive fix */
    .stDataFrame {
        width: 100% !important;
    }
    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🍔 Food Sentiment Analysis</h1>
        <p class="sub-text">AI-Powered Review Classification System</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🇮🇳</h1>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg")
    
    st.divider()
    
    st.title("🚀 About Project")
    st.info("Customer reviews analyze karke business growth mein madad karta hai.")
    
    st.title("🛠️ Tech Stack")
    st.markdown("- **ML:** Scikit-Learn\n- **UI:** Streamlit\n- **Data:** Pandas/Numpy")
    
    st.title("📞 Contact Us")
    st.success("📍 AI Engineers @ DUCAT")

# --- ML Logic ---
def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower()

try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("⚠️ Model 'Sentiment_model.pkl' not found!")

# --- Responsive Grid Layout ---
# Mobile par columns ek ke niche ek aa jayenge
col_main, col_spacer = st.columns([3, 1])

with col_main:
    st.write("### 💬 Real-time Prediction")
    sample = st.text_area("Customer review yahan likhein...", placeholder="e.g. Delicious food!", height=100)
    
    if st.button("Analyze Sentiment"):
        if sample:
            pred = model.predict([sample])
            prob = model.predict_proba([sample])
            
            # Nested columns for result cards
            c1, c2 = st.columns(2)
            with c1:
                if pred[0] == 0:
                    st.error("### Neg 👎")
                else:
                    st.success("### Pos 👍")
            with c2:
                confidence = prob[0][0] if pred[0] == 0 else prob[0][1]
                st.metric("Confidence", f"{confidence*100:.1f}%")
        else:
            st.warning("Please type something!")

st.divider()

# --- Bulk Analysis ---
st.write("### 📂 Bulk Analysis (CSV)")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file, names=["Review"])
    if st.button("Process Bulk File"):
        df['Result'] = model.predict(df['Review'])
        df['Sentiment'] = df['Result'].map({0: 'Dislike 👎', 1: 'Like 👍'})
        st.dataframe(df[['Review', 'Sentiment']], use_container_width=True)
