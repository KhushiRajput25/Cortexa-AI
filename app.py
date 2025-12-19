from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
app = Flask(__name__)
model = None 

# List of features the model was trained on. CRUCIAL for matching inputs.
MODEL_FEATURES = ['mood', 'sleep', 'activity', 'diet_moderate', 'diet_poor']

# --- 1. LOAD MODEL (Run once at startup) ---
try:
    with open('predictor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model 'predictor_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("ERROR: predictor_model.pkl not found. Run train_model.py first.")

# --- 2. THE SCORING/PREPROCESSING FUNCTION ---
def preprocess_input(form_data):
    # Initialize the input array with zeros
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)

    # Convert numeric inputs (direct scoring)
    input_df['mood'] = int(form_data['mood'])
    input_df['sleep'] = int(form_data['sleep'])
    input_df['activity'] = int(form_data['activity'])

    # Handle Categorical Input (Diet Scoring / One-Hot Encoding)
    diet = form_data['diet']
    if diet == 'moderate':
        input_df['diet_moderate'] = 1
    elif diet == 'poor':
        input_df['diet_poor'] = 1

    return input_df.values.flatten() 

# --- 3. FLASK ROUTES ---

# Route for the Home/Input Page
@app.route('/')
def home():
    # Looks for index.html in the 'templates' folder
    return render_template('index.html') 

# Route for Prediction (Handles POST request from the form)
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not available. Server error.", 500

    form_data = request.form
    
    # Preprocess/Score the user data
    model_input = preprocess_input(form_data)
    final_features = np.array(model_input).reshape(1, -1)
    
    # Make the prediction and get probability
    prediction = model.predict(final_features)
    risk_confidence = model.predict_proba(final_features)[0][prediction[0]] * 100 
    
    # --- RESULT AND ANALYSIS (Minimal Logic) ---
    risk_status = "AT RISK" if prediction[0] == 1 else "LOW RISK INDICATION"
    
    if risk_status == "AT RISK":
        description = "Prediction is linked to factors commonly associated with chronic stress, sleep disruption, or poor nutrient intake."
        recommendation = "IMMEDIATE ACTION: Focus on improving sleep hygiene, ensuring 7-9 hours of consistent sleep, and incorporating daily stress reduction techniques."
    else:
        description = "Your current lifestyle shows a low correlation with common neurological risk factors based on this model."
        recommendation = "Continue monitoring mood and sleep. Maintain current consistency in physical activity and diet."
        
    # Render a simple result page with CORRECTED NAME
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Cortexa AI Result</title>
        <link rel="stylesheet" href="{url_for('static', filename='style.css')}">
        <style>
            .result-box {{ 
                margin-top: 20px; 
                padding: 20px; 
                border-radius: 8px; 
                text-align: center;
                background-color: { '#f8d7da' if prediction[0] == 1 else '#d4edda' }; 
                border: 1px solid { '#f5c6cb' if prediction[0] == 1 else '#c3e6cb' };
                color: { '#721c24' if prediction[0] == 1 else '#155724' };
            }}
            .recommendation {{ margin-top: 15px; font-style: italic; color: #555; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cortexa AI Prediction</h1>
            <div class="result-box">
                <h2>{risk_status}</h2>
                <p>Confidence: <strong>{risk_confidence:.2f}%</strong></p>
            </div>

            <h2>Minimal Analysis</h2>
            <p>{description}</p>
            
            <h2>Recommendation</h2>
            <p class="recommendation">{recommendation}</p>
            
            <a href="{url_for('home')}"><button>Analyze New Metrics</button></a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)