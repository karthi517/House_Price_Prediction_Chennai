import numpy as np
from flask import Flask, request, jsonify, render_template,request
import pickle
import pandas as pd
# Create flask app
app = Flask(__name__)
#model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

# Load the Decision Tree model from a pickle file
with open('decision_tree.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the LabelEncoder and StandardScaler objects from pickle files
with open('label_encoders.pkl', 'rb') as file:
    loaded_label_encoders = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# Helper function to preprocess the user input
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data], columns=input_data.keys())

    # Perform label encoding for categorical columns
    columns=['location','ownership','sale_type', 'age_of_the_property', 'plot_type','Furnishing',
       'Status', 'Facing']
    for column in columns:
          
        df[column] = loaded_label_encoders [column].transform(df[column])

    # Standardize numerical columns using the scaler
   
    df= standard_scaler.transform(df)

    return df

# Define the route and view for the web application
@app.route('/next_page', methods=['GET', 'POST'])
def next_page():
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

        return render_template('next_page.html', predicted_price=predicted_price)

    return render_template('next_page.html')


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port='8080')


