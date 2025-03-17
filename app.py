from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('random_forest_olympics_model.joblib',mmap_mode=None)

# List of expected features (Example: Fill in your actual feature names)
sport_columns = [
    'Sport_Alpine Skiing', 'Sport_Alpinism', 'Sport_Archery',
    'Sport_Art Competitions', 'Sport_Athletics', 'Sport_Badminton',
    'Sport_Baseball', 'Sport_Basketball', 'Sport_Basque Pelota',
    'Sport_Beach Volleyball', 'Sport_Biathlon', 'Sport_Bobsleigh',
    'Sport_Boxing', 'Sport_Canoeing', 'Sport_Cricket', 'Sport_Croquet',
    'Sport_Cross Country Skiing', 'Sport_Curling', 'Sport_Cycling',
    'Sport_Diving', 'Sport_Equestrianism', 'Sport_Fencing',
    'Sport_Figure Skating', 'Sport_Football', 'Sport_Freestyle Skiing',
    'Sport_Golf', 'Sport_Gymnastics', 'Sport_Handball', 'Sport_Hockey',
    'Sport_Ice Hockey', 'Sport_Jeu De Paume', 'Sport_Judo',
    'Sport_Lacrosse', 'Sport_Luge', 'Sport_Military Ski Patrol',
    'Sport_Modern Pentathlon', 'Sport_Motorboating',
    'Sport_Nordic Combined', 'Sport_Polo', 'Sport_Racquets',
    'Sport_Rhythmic Gymnastics', 'Sport_Roque', 'Sport_Rowing',
    'Sport_Rugby', 'Sport_Rugby Sevens', 'Sport_Sailing', 'Sport_Shooting',
    'Sport_Short Track Speed Skating', 'Sport_Skeleton',
    'Sport_Ski Jumping', 'Sport_Snowboarding', 'Sport_Softball',
    'Sport_Speed Skating', 'Sport_Swimming', 'Sport_Synchronized Swimming',
    'Sport_Table Tennis', 'Sport_Taekwondo', 'Sport_Tennis',
    'Sport_Trampolining', 'Sport_Triathlon', 'Sport_Tug-Of-War',
    'Sport_Volleyball', 'Sport_Water Polo', 'Sport_Weightlifting',
    'Sport_Wrestling'
]

# Helper function to create sample input
def create_sample_from_input(data):
    # Extract inputs from JSON request
    age = data.get('age')
    height = data.get('height')
    weight = data.get('weight')
    bmi = data.get('bmi')
    sex = data.get('sex')
    year = data.get('year')
    season = data.get('season')
    sport_name = data.get('sport_name')

    # One-hot encode sport
    one_hot_sport = [1 if col == f"Sport_{sport_name}" else 0 for col in sport_columns]

    if sum(one_hot_sport) == 0:
        return None, f"Sport '{sport_name}' not found in available sports."

    # Arrange features in correct order
    sample_input = np.array([[sex, age, height, weight, year, season, bmi] + one_hot_sport])

    return sample_input, None

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Create input
    sample_input, error = create_sample_from_input(data)

    if error:
        return jsonify({"error": error}), 400

    # Predict
    prediction = model.predict(sample_input)

    # Map numerical prediction to medal label
    medal_mapping = {0: "No Medal", 1: "Bronze", 2: "Silver", 3: "Gold"}
    prediction_value = prediction[0]  # Make sure this is a number!
    
    predicted_label = medal_mapping.get(prediction_value, "Unknown")
    
    return jsonify({
    "prediction": prediction_value
})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
