import streamlit as st

st.title("Welcome to the Heart Disease Prediction App")
st.write('Heart disease refers to any problem affecting the heart, such as coronary artery disease, arrhythmia, and heart failure')
st.divider()
st.write("This app provides an interactive way to explore heart disease data and make predictions using machine learning models.")

    
st.subheader("Features:")
st.markdown("- **EDA Page**: Explore and visualize heart disease data of the datasets.")
st.markdown("- **Prediction Page**: Input patient data and get a prediction on heart disease risk.")
    
st.image("https://www.cdc.gov/heart-disease/media/images/2024/10/Heart-Disease-Facts.jpg", width=600)