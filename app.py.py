import streamlit as st
age = st.slider("Select your age", 0, 100, 30)
st.write(f"You are {age} years old!")