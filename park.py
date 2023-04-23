import json
import numpy as np
import pandas as pd
import xgboost as xgb

# Load the XGBoost model from file
model = xgb.Booster()
model.load_model('parkinsons_model.xgb')

# Define the function that will handle the post request
def predict_from_post(request):
    # Retrieve the input data from the post
    input_data = request.form # Assuming the input data is sent as form data
    # Convert the input data into the appropriate format
    age = float(input_data['age'])
    gender = input_data['gender']
    tremors = float(input_data['tremors'])
    rigidity = float(input_data['rigidity'])
    bradykinesia = float(input_data['bradykinesia'])
    sleep_disorders = int(input_data['sleep_disorders'])
    depression = int(input_data['depression'])
    cognitive_impairment = int(input_data['cognitive_impairment'])
    family_history = int(input_data['family_history'])
    input_array = np.array([age, gender, tremors, rigidity, bradykinesia, sleep_disorders, depression, cognitive_impairment, family_history])
    input_dataframe = pd.DataFrame([input_array], columns=['age', 'gender', 'tremors', 'rigidity', 'bradykinesia', 'sleep_disorders', 'depression', 'cognitive_impairment', 'family_history'])
    # Make the prediction
    prediction = model.predict(xgb.DMatrix(input_dataframe))[0]
    # Return the prediction
    if prediction >= 0.5:
        result = 'Likely to have Parkinson\'s disease'
    else:
        result = 'Not likely to have Parkinson\'s disease'
    return {'result': result, 'prediction': prediction}
