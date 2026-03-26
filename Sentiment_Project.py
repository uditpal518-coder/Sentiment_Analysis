import  streamlit as st
import joblib
import re
import pandas as pd
import numpy as np


def mycleaning(doc):
    return re.sub("^a-zA-Z","",doc).lower()

model=joblib.load("Sentiment_model.pkl")

st.set_page_config(layout='wide')
st.title("Food Sentiment Analysis")

st.sidebar.image("Flag_of_india.jpg")
st.sidebar.title("About Project")
st.sidebar.write("Prediction of Sentiment Negative or Positive for a food review")

st.sidebar.title("Contact Us")
st.sidebar.write("999999999")

st.sidebar.title("About us")
st.sidebar.write("We are a group of AI Engineers at DUCAT")

st.write("\n")
st.write("### Enter Review")
sample=st.text_input("")
if st.button("Predict"):
    pred=model.predict([sample])
    prob=model.predict_proba([sample])
    if pred[0]==0:
        st.write("Neg👎")
        st.write(f"Confidence Score : {prob[0][0]:.2f}")
    else:
        st.write("Pos👍")
        st.write(f"Confidence Score : {prob[0][1]:.2f}")


st.write("### Bulk Prediction")
file=st.file_uploader("select file",type=["csv","txt"])
if file:
    df=pd.read_csv(file,names=["Review"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("Predict",key="b2"):
        corpus=df.Review
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df['Sentiment']=pred
        df['Confidence']=prob
        df['Sentiment']=df['Sentiment'].map({0:'Dislike👎',1:'Like👍'})
        placeholder.dataframe(df)


