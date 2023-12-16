import streamlit as st
import pickle
import pandas as pd
model = pickle.load(open('irismodel.pkl','rb'))
st.title('iris data prediction')
sl = st.number_input('sepal length')
sw = st.number_input('sepal width')
pl = st.number_input('petal length')
pw = st.number_input('petal width')
bt = st.button('predict')
if bt:
    result = model.predict([[sl,sw,pl,pw]]) 
    st.write(f'The predicted value is {result[0]}')


