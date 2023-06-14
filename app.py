import streamlit as st
import pickle
import numpy as np


st.set_page_config(
    page_title="Regression Basic Deployment",
    page_icon="🚀"
)

st.title('Simple Regression Example : Model Deployment')

if 'model' not in st.session_state:
    model = pickle.load(open('data/model.sav', 'rb'))
    st.session_state['model'] = model

area_input = st.number_input('Insert Area of the House 🏠')
bedroom_input = st.number_input('Insert How many bedrooms in the House 🛏️')
age_input = st.number_input('Insert Age of the House 🔢')

if st.button('Model Predict'):
    data = np.array([area_input, bedroom_input, age_input]).reshape(1, -1)
    result = st.session_state['model'].predict(data)
    st.write(f'Prediction model : ${result[0]}')
else:
    st.write('Please input the feature above to start modelling')