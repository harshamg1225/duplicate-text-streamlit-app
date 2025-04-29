import streamlit as st
import helper
import pickle
import numpy as np

try:
    rm = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title('Duplicate Question Pairs Detector')
st.write("Enter two questions to check if they are duplicates.")

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

st.write("Q1:", q1)
st.write("Q2:", q2)

if st.button('Check Duplicate'):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        try:
            query = helper.query_point_creator(q1, q2)  # Make sure this returns the correct format
            
            result = rm.predict(np.array(query).reshape(1, -1))[0]

            if result:
                st.success('Result: Duplicate ✅')
            else:
                st.info('Result: Not Duplicate ❌')

        except Exception as e:
            st.error(f"An error occurred: {e}")

