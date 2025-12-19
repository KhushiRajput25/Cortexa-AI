import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# 1. CREATE MINIMAL DATASET (Simulated Training Data)
# Mood: 1-10, Sleep: 4-12 hrs, Activity: 0-7 days/week
# Risk: 0 = No Risk, 1 = At Risk
data = {
    'mood': [8, 9, 7, 3, 5, 2, 8, 7, 4, 6, 9, 3],
    'sleep': [7, 8, 6, 5, 4, 9, 7, 5, 10, 6, 7, 4],
    'activity': [5, 6, 4, 1, 2, 0, 5, 3, 6, 2, 5, 1],
    'diet': ['healthy', 'healthy', 'moderate', 'poor', 'poor', 'poor', 'healthy', 'moderate', 'healthy', 'moderate', 'healthy', 'poor'],
    'risk': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1] 
}
df = pd.DataFrame(data)

# 2. FEATURE ENGINEERING (Scoring / One-Hot Encoding for 'diet')
df_processed = pd.get_dummies(df, columns=['diet'], drop_first=True)
X = df_processed[['mood', 'sleep', 'activity', 'diet_moderate', 'diet_poor']]
y = df_processed['risk']

# 3. TRAIN THE LOGISTIC REGRESSION MODEL
model = LogisticRegression(solver='liblinear') # Using liblinear for small datasets
model.fit(X, y)

# 4. SAVE THE MODEL (Serialization)
with open('predictor_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Minimal Logistic Regression model trained and saved as 'predictor_model.pkl'.")