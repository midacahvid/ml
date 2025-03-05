import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

df = pd.read_csv('clean_heart.csv')

st.title("Exploratory Data Analysis (EDA)")
st.write("This section will contain various visualizations and insights about the heart disease dataset.")
    
# Placeholder for EDA content
st.info("EDA features coming soon!")
col1, col2, col3 = st.columns(3)

col1.metric("No patient's", value=df.shape[0])
col2.metric("Count of features", value=len(df.columns)-1)
col3.metric('Mean age', value=round(df['age'].mean(), 2))
style_metric_cards()


# male and female
sex, labels= st.columns(2, border=True)
fig1 = px.pie(df, names=df['sex'].map({0: 'Female', 1: 'Male'}), title='Male and Female')
sex.plotly_chart(fig1)

# count of labels
fig2 = px.pie(df, names=df['target'].map({0: 'No Disease', 1: 'Heart Disease'}), title='Count of Target')
labels.plotly_chart(fig2)


gender_col, cp_col= st.columns(2, border=True)
# Heart Disease by Gender
gender_target = df.groupby('sex')['target'].sum().reset_index()
fig3 = px.bar(gender_target, x=gender_target['sex'].map({0:'Female', 1:'Male'}), y='target', title='Heart Disease by Gender'
)
fig3.update_xaxes(title_text='Gender')
fig3.update_yaxes(title_text='Count')
gender_col.plotly_chart(fig3)
# chest pain
chest_pain =  df['cp'].value_counts(ascending=False).reset_index()
fig4= px.bar(chest_pain, x=chest_pain['cp'].map({0:'Typical Angina', 1:'Atypical Angina', 2:'Non-Anginal Pain', 3:'Asymptomatic'}), y='count', title='Chest Pain Type')
fig4.update_xaxes(title_text='Chest Pain Type')
fig4.update_yaxes(title_text='Count')
cp_col.plotly_chart(fig4)


# correlation of features
corr = df.corr()
fig5 = px.imshow(corr, title='Correlation Matrix')
st.plotly_chart(fig5)