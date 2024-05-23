from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

filepath = os.path.join(os.path.dirname(__file__),'models','rf_pipeline.pkl')

app = Flask(__name__)


# Load your trained model (replace 'model.pkl' with your model file)
with open(filepath, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return jsonify({message :"Server Started"})

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    temperature = float(data['Temperature'])
    humidity = float(data['Humidity'])
    moisture = float(data['Moisture'])
    soil_type = data['Soil Type']
    crop_type = data['Crop Type']
    nitrogen = float(data['Nitrogen'])
    potassium = float(data['Potassium'])
    phosphorous = float(data['Phosphorous'])
    
    # Encode categorical variables
    soil_type_encoded = encode_soil_type(soil_type)
    crop_type_encoded = encode_crop_type(crop_type)
    
    # Create DataFrame for model input
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]],
                              columns=['Temperature', 'Humidity', 'Moisture', 'soil_type_encoded', 'crop_type_encoded', 'Nitrogen', 'Potassium', 'Phosphorous'])
    
    # Predict using the model
    prediction = model.predict(input_data)[0]
    return jsonify({message :"Prediction Successful", prediction : prediction})

def encode_soil_type(soil_type):
    # Replace with your actual encoding logic or mapping
    soil_types = ['Black', 'Clayey', 'Loamy', 'Red','Sandy']
    return soil_types.index(soil_type)

def encode_crop_type(crop_type):
    # Replace with your actual encoding logic or mapping
    crops = ['Brinjal', 'Chili', 'Tomato']
    return crops.index(crop_type)

if __name__ == "__main__":
    app.run(debug=True)
