import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

st.title("Iris Flower Classification")
st.write("Enter the values for the flower's features:")

# input form
st.write("id:")
iris_id = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, key="id")
st.write("Sepal Length (cm):")
sepal_length = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, key="sepal_length")

st.write("Sepal Width (cm):")
sepal_width = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, key="sepal_width")

st.write("Petal Length (cm):")
petal_length = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, key="petal_length")

st.write("Petal Width (cm):")
petal_width = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, key="petal_width")
# predict button
if st.button("Predict"):
    sample = [iris_id, sepal_length, sepal_width, petal_length, petal_width]
    prediction = loaded_model.predict([sample])
    species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    st.write("Prediction:", species_mapping[int(prediction[0])])




