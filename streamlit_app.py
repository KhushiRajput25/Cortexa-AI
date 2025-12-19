import streamlit as st
import pandas as pd
import joblib

# Load your model (Make sure the name matches your .pkl file exactly)
model = joblib.load('predictor_model.pkl')

st.title("🧠 Cortexa AI: Neurological Risk assessment")

# Creating the Inputs
mood = st.slider("Rate your mood (1 is low, 10 is high)", 1, 10, 5)
sleep = st.number_input("Hours of sleep last night", 0, 15, 7)
activity = st.slider("Physical Activity Level", 1, 10, 5)
diet = st.slider("Diet Quality", 1, 10, 5)

if st.button("Analyze My Risk"):
    # This converts your inputs into the format the AI understands
    data = pd.DataFrame([[mood, sleep, activity, diet]], 
                        columns=['mood', 'sleep', 'activity', 'diet'])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Result: High Neurological Risk Detected. Please consult a doctor.")
    else:
        st.success("Result: Low Neurological Risk Detected. Keep up the healthy lifestyle!")
