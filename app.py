from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Define the route and view for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Load the Decision Tree model from a pickle file
        with open('decision_tree1.pkl', 'rb') as file:
            model = pickle.load(file)

        # Load the LabelEncoder and StandardScaler objects from pickle files
        with open('label_encoder1.pkl', 'rb') as file:
            label_encoder = pickle.load(file)

        with open('scaler1.pkl', 'rb') as file:
            standard_scaler = pickle.load(file)

        # Get user inputs from the form
        location = request.form['location']
        # ... (rest of the code remains the same)
        # ... (preprocessing, prediction, and rendering)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
