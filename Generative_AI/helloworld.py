import streamlit as st
import pandas as pd 
st.title("Hello World") ###To run app type --  streamlit run helloworld.py
st.write("EXAMPLE")
df=pd.DataFrame({
    'first column' : [1,2,3,4],
    'second column' : [10,20,30,40]}
)
st.write("Here is the dataframe")
st.write(df)
st.line_chart(df)