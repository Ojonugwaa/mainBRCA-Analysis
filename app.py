from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model for storing user data
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    features = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)

# Define the model path
model_path = 'logistic_regression_model.pkl'
scaler_path = 'scaler.pkl'

# Load the model (you can move this inside main if you prefer delayed loading)
#with open(model_path, 'rb') as f:
    #model = pickle.load(f)

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

@app.route('/list_files', methods=['GET'])
def list_files():
    # List all files and directories in the current directory
    files = os.listdir(os.getcwd())
    return jsonify({'files': files})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the incoming JSON data
    data = request.get_json()

    name = data['patient_name']
    features = [
        float(data['feature1']),
        float(data['feature2']),
        float(data['feature3']),
        float(data['feature4']),
        float(data['feature5']),
        float(data['feature6']),
        float(data['feature7']),
        float(data['feature8']),
        float(data['feature9']),
        float(data['feature10'])
    ]

    # Convert features into a numpy array and scale them using the scaler
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)  # Apply the same scaling as during training

    # Make prediction
    prediction = model.predict(scaled_features)
    result = 'Malignant' if prediction[0] == 0 else 'Benign'

    # Store the result in the database
    new_prediction = Prediction(name=name, features=str(features), result=result)
    db.session.add(new_prediction)
    db.session.commit()

    # Return JSON response
    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Check model file existence and inspect contents
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    else:
        print(f"Model file found at {model_path}, size: {os.path.getsize(model_path)} bytes")
        with open(model_path, 'rb') as f:
            head = f.read(100)
            print(f"First 100 bytes of model file:\n{head}")

    # Initialize database and run app
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)