# import neccessary files

import streamlit as st
import pickle

# loading model
with open('rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
st.title("Heart Disease Prediction")
st.write("Enter patient details below to predict the likelihood of heart disease.")



# getting users input
age =  st.number_input('Patients age', min_value=1, max_value=100, step=1, key='age')
sex = st.selectbox("Patient's sex", ['Male', 'Female'], key='sex')
cp = st.selectbox('Chest pain experience by patient', ['Typical Angina','Atypical Angina','Non-Anginal Pain', 'Asymptomatic'], key='cp')
bp = st.slider('Drag to select systolic blood pressure', min_value=90, max_value=300, key='bp')
chol = st.slider('Drag to select cholesterol level', min_value= 50, max_value = 400, key='chol')
fbs = st.selectbox('is fasting blood sugar level above 120mg/dl?', ['Yes', 'No'], key='fbs')
restecg = st.selectbox('Resting electrocardiographic results', ['Normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'], key='restecg')
hr = st.slider("Drag to select patient's heart rate", min_value=50, max_value=180, key='hr')
exang  = st.selectbox('Exercise-induced angina', ['Yes', 'No'], key='exang')
oldpeak = st.slider('Drag to select ST depression induced by exercise relative to rest', min_value=0.0, max_value =  10.0, step=0.1, key='oldpeak')
slope = st.selectbox('What is the Slope of the peak exercise ST segment?', ['Upsloping', 'Flat', 'Downsloping'], key='slope')
ca = st.selectbox('Number of major vessels colored by fluoroscopy', [0, 1, 3, 4], key='ca')
thal =  st.selectbox('Type of Thalassemia', ['Normal', 'Fixed defect', 'Reversible defect'], key='thal')
 
# mapping some variables
sex_mapping = {'Female':0, 'Male':1}
cp_mapping = {'Typical Angina':0,'Atypical Angina':1,'Non-Anginal Pain':2, 'Asymptomatic':3}
yes_no_mapping = {'No':0, 'Yes': 1}
restecg_mapping = {'Normal':0, 'ST-T wave abnormality':1, 'left ventricular hypertrophy':2}
slope_mapping = {'Upsloping':0, 'Flat':1, 'Downsloping':2}
thal_mapping = {'Normal':1, 'Fixed defect':2, 'Reversible defect':3}

#storing inputes
data = [
    age, 
    sex_mapping[sex], 
    cp_mapping[cp],
    bp, 
    chol, 
    yes_no_mapping[fbs], 
    restecg_mapping[restecg],
    hr,
    yes_no_mapping[exang],
    oldpeak,
    slope_mapping[slope],
    ca,
    thal_mapping[thal]
    ]

#predict when button is click
if st.button('Predict'):
    prediction =  loaded_model.predict([data])

    if int(prediction[0]) == 0:
        st.write('This patient is likely not having heart disease')
    else:
        st.write('This patient is likely having heart disease')
