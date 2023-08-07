# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the Decision Tree model from a pickle file
with open('decision_tree1.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the LabelEncoder and StandardScaler objects from pickle files
with open('label_encoder1.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# Helper function to preprocess the user input
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data], columns=input_data.keys())

    # Perform label encoding for categorical columns
   
    df=df.apply(label_encoder.fit_transform) 

    # Standardize numerical columns using the scaler
   
    df= standard_scaler.transform(df)

    return df

# Define the route and view for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        location = request.form['location']
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        area = float(request.form['area'])
        carpet_area = float(request.form['carpet_area'])
        ownership = request.form['ownership']
        sale_type = request.form['sale_type']
        age_of_the_property = request.form['age_of_the_Property']
        plot_type = request.form['plot_type']
        floor = request.form['floor']
        furnishing = request.form['Furnishing']
        status = request.form['Status']
        facing = request.form['Facing']

        # Preprocess the user input
        input_data = {
            'location': location,
            'bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'area': area,
            'carpet_area': carpet_area,
            'ownership': ownership,
            'sale_type': sale_type,
            'age_of_the_property': age_of_the_property,
            'plot_type': plot_type,
            'floor': floor,
            'Furnishing': furnishing,
            'Status': status,
            'Facing': facing
        }
        input_df = preprocess_input(input_data)

        # Make prediction using the model
        predicted_price = model.predict(input_df)[0]

        return render_template('index.html', predicted_price=predicted_price)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
