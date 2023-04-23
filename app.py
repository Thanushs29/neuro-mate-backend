
import pymongo
from flask import Flask
from pymongo import MongoClient
import uuid
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import request,jsonify
import wave
import base64
from flask_cors import CORS
from flask_cors import cross_origin
import parselmouth
import librosa
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load


app = Flask(__name__)
CORS(app)
client = MongoClient('mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=2000')
db = client['nuro']
collection = db['user']
@app.route('/signIn',methods=['POST'])
def index():
    name = request.args.get('name')
    mail = request.args.get('mail')
    password = request.args.get('password')
    gender = request.args.get('gender')
    age = request.args.get('age')
    result = collection.find_one({'mail': mail})
    if(result):
        error = {'error': 'User Already exist'}
        return jsonify(error), 400
    uuid_str = str(uuid.uuid4()).replace('-', '')[:16]
    collection.insert_one({'_id':"USR-"+uuid_str,'name': name,'mail': mail,'password': password,'gender': gender,'age': age})
    res={'data':"USR-"+uuid_str}
    return jsonify(res)

@app.route('/logIn',methods=['GET'])
def logIn():
    mail = request.args.get('mail')
    password = request.args.get('password')
    result = collection.find_one({'mail': mail})
    if(result):
        print(result['password'])
        if(result['password']==password):
            return result
        else:
            error = {'error': 'Invalid PassWord'}
            return jsonify(error), 400
    else:
        error = {'error': 'User Does Not exist'}
        return jsonify(error), 400


# Load the XGBoost model from file
model = xgb.Booster()
model.load_model('parkinsons_model.xgb')

# Define the route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the post
    input_data = request.json
    # Convert the input data into the appropriate format
    age = input_data['age']
    tremors = input_data['tremors']
    rigidity = input_data['rigidity']
    bradykinesia = input_data['bradykinesia']
    sleep_disorders = input_data['sleep']
    depression = input_data['depression']
    cognitive_impairment = input_data['cognition']
    family_history = input_data['family']
    input_array = np.array([age, tremors, rigidity, bradykinesia, sleep_disorders, depression, cognitive_impairment, family_history])
    input_dataframe = pd.DataFrame([input_array], columns=['age', 'tremors', 'rigidity', 'bradykinesia', 'sleep_disorders', 'depression', 'cognitive_impairment', 'family_history'])
    # Make the prediction
    prediction = model.predict(xgb.DMatrix(input_dataframe))[0]
    # Return the prediction
    if prediction >= 0.5:
        result = 'Likely to have Parkinson\'s disease'
    else:
        result = 'Not likely to have Parkinson\'s disease'
    return jsonify({'result': str(result), 'prediction': str(prediction)})

@app.route('/record-audio',methods=['POST'])
@cross_origin()
def base64_to_wav():
    # Decode the base64-encoded string
    input_data = request.json["audio"]
    # audio_data = input_data['audio']
    audio_bytes = base64.b64decode(input_data)
    output_file = 'test.wav'


    # Open the output WAV file for writing
    with wave.open(output_file, 'wb') as wave_write:
        # Set the parameters for the WAV file (1 channel, 16 bits per sample, 44100 Hz)
        wave_write.setnchannels(1)
        wave_write.setsampwidth(2)
        wave_write.setframerate(44100)

        # Write the audio data to the WAV file
        wave_write.writeframes(audio_bytes)
    # result = 'Not likely to have Parkinson\'s disease'
    result = "Voice frequcy uploaded"
    return jsonify({'result': str(result)})


@app.route('/predict-out',methods=['GET'])
def predictAudioOut():
    prediction = predictOut(audioOut('test.wav'))
    print("Final Output",prediction)
    # prediction =0.2
    result =""
    if prediction >= 0.5:
        result = 'Likely to have Parkinson\'s disease'
    else:
        result = 'Not likely to have Parkinson\'s disease'
    return jsonify({'result': str(result), 'prediction': str(prediction)})


@app.route('/prediction', methods=['POST'])
@cross_origin()
def prediction_out():
    input_data = request.get_json() # Get input data from POST request
    # Do something with input_data
    prediction = predictOut(input_data)
    result =""
    if prediction >= 0.5:
        result = 'Likely to have Parkinson\'s disease'
    else:
        result = 'Not likely to have Parkinson\'s disease'
    return jsonify({'result': str(result), 'prediction': str(prediction)})
    # return jsonify(input_data) # Return output_data as a JSON response/

def audioOut(audio_file_path):
    
    sound = parselmouth.Sound(audio_file_path)

    pitch = sound.to_pitch()
    mean_f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    sd_f0 = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
    hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = parselmouth.praat.call(hnr, "Get mean", 0, 0)
    sd_hnr = parselmouth.praat.call(hnr, "Get standard deviation", 0, 0)

    fo_count = (f"{mean_f0:.2f}")
    fhi_count = (f"{pitch.ceiling:.2f}")
    flo_counts= 65.476

    y, sr = librosa.load(audio_file_path)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=600)

    harmonic = librosa.effects.harmonic(y)

    rms = librosa.feature.rms(y=harmonic)
    spread1_mean =0
    spread2_mean =0
    D2 = 0
    PPE=0

    rms_voiced = rms[0, voiced_flag]

    mdvp_jitter_percent = 100 * (np.std(rms_voiced) / np.mean(rms_voiced))
    mdvp_jitter_abs = np.std(rms_voiced)

    mdvp_rap = np.sum(np.abs(np.diff(rms_voiced)))
    mdvp_ppq = np.sum(np.square(np.diff(rms_voiced)))
    jitter_ddp = mdvp_rap * 2

    rms_voiced = librosa.feature.rms(y=harmonic)
    stft = np.abs(librosa.stft(y))
    mdvp_apq = np.mean(np.sqrt(np.sum(np.square(stft), axis=0)), axis=0)

    frame_length = 2048
    hop_length = 512
    pitch, mag = librosa.core.piptrack(y=y, sr=sr, hop_length=hop_length, fmin=75, fmax=300)
    HNR = librosa.feature.spectral_contrast(y=y, sr=sr)
    RPDE = librosa.feature.spectral_flatness(y=y)
    DFA = librosa.feature.zero_crossing_rate(y=y)

    # Compute spread1 and spread2
    spectral_features = librosa.feature.melspectrogram(y=y, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_spreads = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spread1 = spectral_spreads[0]
    # spread2 = spectral_spreads[1]

    D2 = np.mean(spectral_contrast[1]) - np.mean(spectral_contrast[0])

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    PPE = np.mean(mfccs[1:])

    spread2 = librosa.feature.spectral_bandwidth(S=spectral_features, p=2, center=None)

    print("spreadd",spread1)
    print("spread2",spread2)
    spread1_mean = np.mean(spread1)
    spread2_mean = np.mean(spread2.mean(axis=1))


    # Compute D2
    chroma_features = librosa.feature.chroma_stft(y=y, sr=sr)
    cov = np.cov(chroma_features)
    eig_vals = np.linalg.eigvals(cov)
    sort_index = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sort_index]
    d2 = np.sum(np.sqrt(np.abs(np.diff(eig_vals)))) / np.sum(eig_vals)

    # Compute PPE
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    ppe = np.sum(np.square(np.diff(mfcc)), axis=1)

    mvdp_shimmer = 0.0145
    mdvp_shimmerdb = 0.154
    shimmer_apq3 = 0.00469
    shimmer_apq5 = 0.00747
    shimmer_dda = 0.01567
    nhr = 0.00231

    d2_mean = D2

    hnr_mean = np.mean(HNR.mean(axis=1))
    rpde_mean = np.mean(RPDE.mean(axis=1))
    dfa_mean = np.mean(DFA.mean(axis=1))

    # d2_mean = np.mean(D2)

    print(f"MDVP:Fo(Hz): {mean_f0:.2f}")
    print(f"MDVP:Fhi(Hz): ",fhi_count)
    print("MDVP:Flo(Hz):",flo_counts)
    print("MDVP:Shimmer mode",mvdp_shimmer)
    print("MDVP:Shimmer(dB) mode:",mdvp_shimmerdb)
    print("Shimmer:APQ3 mode:",shimmer_apq3)
    print("Shimmer:APQ5 mode:" ,shimmer_apq5)
    print("Shimmer:DDA mode:" ,shimmer_dda)
    print("NHR mode:" ,nhr)
    print("MDVP:Jitter(%) value:", mdvp_jitter_percent)
    print("MDVP:Jitter(Abs) value:", mdvp_jitter_abs)
    print("MDVP:RAP value:", mdvp_rap)
    print("MDVP:PPQ value:", mdvp_ppq)
    print("Jitter:DDP value:", jitter_ddp)
    print("MDVP: APQ value: ",mdvp_apq)
    print("hnr_mean",hnr_mean)
    print("RPDE",rpde_mean)
    print("DFA",dfa_mean)
    print("spread1",spread1_mean)
    print("spread2",spread2_mean)
    print("D2",d2_mean)
    print("PPE",PPE)

    output = {
        'MDVP:Fo(Hz)': float(fo_count),
        'MDVP:Fhi(Hz)': float(fhi_count),
        'MDVP:Flo(Hz)': flo_counts,
        'MDVP:Jitter(%)': mdvp_jitter_percent,
        'MDVP:Jitter(Abs)': mdvp_jitter_abs,
        'MDVP:RAP': mdvp_rap,
        'MDVP:PPQ': mdvp_ppq,
        'Jitter:DDP': jitter_ddp,
        'MDVP:Shimmer': mvdp_shimmer,
        'MDVP:Shimmer(dB)': mdvp_shimmerdb,
        'Shimmer:APQ3': shimmer_apq3,
        'Shimmer:APQ5': shimmer_apq5,
        'MDVP:APQ': mdvp_apq,
        'Shimmer:DDA': shimmer_dda,
        'NHR': nhr,
        'HNR': hnr_mean,
        'RPDE': rpde_mean,
        'DFA': dfa_mean,
        'spread1': spread1_mean,
        'spread2': spread2_mean,
        'D2': d2_mean,
        'PPE': PPE
    }

    return output


def predictOut(input_data):
    
    # Load the dataset
    data = pd.read_csv('parkinsons.csv')

    # Remove the name column as it is not useful for the model
    data = data.drop(['name'], axis=1)

    # Split the dataset into features (X) and target variable (y)
    X = data.drop(['status'], axis=1)
    y = data['status']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, 'xgboost_model.joblib')

    # Load the saved model
    model = load('xgboost_model.joblib')

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the performance of the model
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


    # Prepare the input data with parkinson disease
    # input_data = {'MDVP:Fo(Hz)': 119.992, 'MDVP:Fhi(Hz)': 157.302, 'MDVP:Flo(Hz)': 74.997, 
    #               'MDVP:Jitter(%)': 0.00597, 'MDVP:Jitter(Abs)': 0, 'MDVP:RAP': 0.00303, 
    #               'MDVP:PPQ': 0.0057, 'Jitter:DDP': 0.00909, 'MDVP:Shimmer': 0.04374, 
    #               'MDVP:Shimmer(dB)': 0.426, 'Shimmer:APQ3': 0.02182, 'Shimmer:APQ5': 0.0313, 
    #               'MDVP:APQ': 0.02971, 'Shimmer:DDA': 0.06545, 'NHR': 0.02211, 'HNR': 21.033, 
    #               'RPDE': 0.414783, 'DFA': 0.815285, 'spread1': -4.81303, 'spread2': 0.266482, 
    #               'D2': 2.30144, 'PPE': 0.284654}

    # input_data = {'MDVP:Fo(Hz)': 109.63, 'MDVP:Fhi(Hz)': 600.00, 'MDVP:Flo(Hz)': 65.476, 'MDVP:Jitter(%)': 82.52149820327759, 'MDVP:Jitter(Abs)': 0.02198106, 'MDVP:RAP': 0.12880914, 'MDVP:PPQ': 0.0011341888, 'Jitter:DDP': 0.2576182782649994, 'MDVP:Shimmer': 0.0145, 'MDVP:Shimmer(dB)': 0.154, 'Shimmer:APQ3': 0.00469, 'Shimmer:APQ5': 0.00747, 'MDVP:APQ': 34.477165, 'Shimmer:DDA': 0.01567, 'NHR': 0.00231, 'HNR': 19.554908223922464, 'RPDE': 0.048343565, 'DFA': 0.10291431568287036, 'spread1': 2090.8923575953118, 'spread2': 1116.43715330105, 'D2': -5.380089532535223, 'PPE': 13.985601}
    input_df = pd.DataFrame([input_data])

    # Make predictions
    y_pred = model.predict(input_df)

    # Calculate accuracy
    y_actual = 1  # Replace with the actual target variable value for the new input
    accuracy = (y_pred == y_actual).sum() / len(y_pred)

    # Print the predicted target variable value and accuracy
    print('Predicted target variable value:', y_pred[0])
    print('Accuracy:', accuracy)

    return y_pred[0]


if __name__ == '__main__':
    # app.debug = True
    app.run()
    #moni\\\
    
    

