import streamlit as st

st.write("Have some relationship problems? Ask our panel of experts!")
x = st.slider('x')  
st.write(x, 'squared is', x * x)
st.text_input("Your name", key="name")

