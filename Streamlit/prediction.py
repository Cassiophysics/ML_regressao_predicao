import joblib
def predict(data):
    xgbr2 = joblib.load('/Streamlit/xgbr2_model.sav')
    return xgbr2.predict(data)

#import streamlit as st
#import os
#path_to_find = os.listdir()
#st.title(path_to_find)

#import joblib
#def predict(data):
#    xgbr2 = joblib.load(open('saved_steps.pkl', 'rb'))
#    return xgbr2.predict(data)