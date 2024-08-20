import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

import shap
from streamlit_shap import st_shap

import streamlit as st
from streamlit import components

# Load the training data
X_train = pd.read_csv('db/X_train.csv', index_col="Unnamed: 0")
y_train = pd.read_csv('db/y_train.csv')


# Load the model from the .pkl file
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

st.title("Are you confused? 🧠")

# 'VideoID', 'Attention', 'Mediation', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'age', 'gender'

VideoID = st.select_slider('Video Id', [0,1,2,3,4,5,6,7,8,9])
Attention = st.slider('Attention Level', 0,100)
Mediation = st.slider('Mediation', 0,100)
Delta = st.slider('Delta', 0, 3964663)
Theta = st.slider('Theta', 0, 3007802)
Alpha1 = st.slider('Alpha1', 0, 1369955)
Alpha2 = st.slider('Alpha2', 0, 1016913)
Beta1 = st.slider('Beta1', 0, 1067778)
Beta2 = st.slider('Beta2', 0, 1645369)
Gamma1 = st.slider('Gamma1', 0, 1972506)
Gamma2 = st.slider('Gamma2', 0, 1348117)
age = st.slider('Age', 0,90)
gender = st.selectbox('Gender', [0, 1])



class_names=['Not-Confused', 'Confused']

def predict():
    input = np.array([[VideoID, Attention, Mediation, Delta, Theta, Alpha1, Alpha2, Beta1, Beta2, Gamma1, Gamma2, age, gender]])
    # Make a prediction
    prediction = loaded_model.predict(input)
    if prediction[0] == 0:
        st.success("You are focused. Keep it up. 🔥")
    else:
        st.error('You seem Confuse. Try to focus more. 😐')
    
    
    # SHAP
    st.subheader('SHAP Explanation')
    shap_explainer = shap.Explainer(loaded_model, X_train)   # this is the shap part, it will change according to the models.
    shap_values = shap_explainer(input)
    st_shap(shap.plots.waterfall(shap_values[0]), height=500, width=700)
    

    # Initialize the LIME explainer
    explainer = LimeTabularExplainer(
        X_train.values, 
        feature_names=X_train.columns.values.tolist(),
        class_names=class_names, 
        verbose=True, 
        mode='classification'
    )

    # Explain the instance
    exp = explainer.explain_instance(input[0], loaded_model.predict_proba, num_features=10)
    
    html_lime = exp.as_html()
    st.subheader('Lime Explanation')
    components.v1.html(html_lime, width=700, height=500, scrolling=True)
    
    
st.button('Predict', on_click=predict)