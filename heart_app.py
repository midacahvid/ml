
import streamlit as st

# Define the pages
home = st.Page("home.py", title="Home", icon="🎈")
eda = st.Page("eda.py", title="Exploratory data analysis", icon="❄️")
model = st.Page("model.py", title="Heart Disease Prediction", icon="🎉")

# Set up navigation
pg = st.navigation([home, eda, model])

# Run the selected page
pg.run()











