import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

st.set_page_config(layout='wide', page_title="Food Sentiment Analysis", page_icon="🍔")

st.markdown("""
    <style>
        
    /*.stApp{background-image: url("https://www.rivaicmimarlik.com/upload/images/sayfalar/2022/restoran-ic-mimari-tasarimi-44902-5037952778.jpg");
        background-size: cover;
        background-position: center;
        padding: clamp(30px, 8vw, 80px);
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        width: 100%;
    }*/

    .hero-section {
        background-image: url("https://media.smallbiztrends.com/2021/05/fast-food-restaurants.png");
        background-size: cover;
        background-position: center;
        padding: clamp(30px, 8vw, 80px);
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        width: 100%;
    }

    .main-heading {
        font-size: clamp(2rem, 10vw, 4.5rem);
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

    </style>
    
    <div class="hero-section">
        <h1 class="main-heading">🍔 Food Sentiment Analysis</h1>
        <p class="sub-text">AI-Powered Review Classification System</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("restaurant.jpg")
    
    st.divider()
    
    st.title("🚀 About Project")
    st.info("Yeh system Natural Language Processing ka use karke food reviews ko automatically analyze karta hai.")
    
    st.title("🛠️ Technical Stack")
    st.write("**Streamlit:** Framework for creating the interactive web interface.")
    st.write("**Scikit-Learn:** Used for training and implementing the Sentiment Model.")
    st.write("**Pandas:** For data manipulation and reading CSV files.")
    st.write("**Numpy:** For mathematical operations and array processing.")
    st.write("**Joblib:** To load the pre-trained Machine Learning model.")
    st.write("**Regex (re):** For cleaning and preprocessing the text data.")
    
    st.divider()
    
    st.title("📞 Contact Us")
    st.success("📍 **AI Engineers @ DUCAT**")
    st.write("📧 **Email:** uditpal518@gmail.com")
    st.write("📱 **Phone:** +91 99999-88888")
    st.write("🌐 **Website:** www.ducatindia.com")

def mycleaning(doc):
    return re.sub("[^a-zA-Z]"," ",doc).lower()

try:
    model = joblib.load("Sentiment_model.pkl")
except:
    st.error("⚠️ Model 'Sentiment_model.pkl' not found!")

col_main, col_spacer = st.columns([3, 1])

with col_main:
    st.write("### 💬 Real-time Prediction")
    sample = st.text_area("Customer review...", placeholder="e.g. The food was absolutely delicious!", height=100)
    
    if st.button("Analyze Sentiment"):
        if sample:
            pred = model.predict([sample])
            prob = model.predict_proba([sample])
            
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

st.write("### 📂 Bulk Analysis (CSV & TXT)")
data_source = st.radio("Choose Data Source:",["Upload your own file","Use Sample File"])
if data_source == "Upload your own file":
    file = st.file_uploader("Upload CSV & TEXT", type=["csv","txt"])
    if file:
        df = pd.read_csv(file, names=["Review"])
elif data_source == "Use Sample File":
    with open("reviews.txt", "br") as file:
        df = pd.read_csv(file, names=["Review"])

if df is not None:  
    if st.button("Process Bulk File"):
        df['Result'] = model.predict(df['Review'])
        df['Confidence']= np.max(model.predict_proba(df['Review']),axis=1)
        df['Sentiment'] = df['Result'].map({0: 'Dislike 👎', 1: 'Like 👍'})
        st.dataframe(df[['Review', 'Sentiment','Confidence']], use_container_width=True)
