import parselmouth
import os
import librosa
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
import wave


import base64
import io

# Replace 'audio_file_path' with the path to your audio file
# audio_file_path = 'output.wav'

# # Load the audio file using wave
# with wave.open(audio_file_path, 'rb') as f:
#     audio_data = f.readframes(f.getnframes())

# sound = parselmouth.Sound(audio_data, sampling_frequency=f.getframerate())


def audioOut(audio_file_path):
    
    # Replace 'audio_file_path' with the path to your audio file
    # audio_file_path = 'output.wav'

    # Load the audio file using parselmouth
    sound = parselmouth.Sound(audio_file_path)

    # Extract the acoustic features
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

    # Extract fundamental frequency (F0)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=600)

    # Extract harmonic part of the signal
    harmonic = librosa.effects.harmonic(y)

    # Extract root-mean-square (RMS) amplitudes of each frame in the harmonic signal
    rms = librosa.feature.rms(y=harmonic)
    spread1_mean =0
    spread2_mean =0
    D2 = 0
    PPE=0

    # Extract MDVP:Jitter(%) and MDVP:Jitter(Abs) features
    # if (True):
    rms_voiced = rms[0, voiced_flag]

    mdvp_jitter_percent = 100 * (np.std(rms_voiced) / np.mean(rms_voiced))
    mdvp_jitter_abs = np.std(rms_voiced)

    # Extract MDVP:RAP, MDVP:PPQ, and Jitter:DDP features
    mdvp_rap = np.sum(np.abs(np.diff(rms_voiced)))
    mdvp_ppq = np.sum(np.square(np.diff(rms_voiced)))
    jitter_ddp = mdvp_rap * 2

    # Compute the RMS energy of the voiced segments
    rms_voiced = librosa.feature.rms(y=harmonic)
    stft = np.abs(librosa.stft(y))
    mdvp_apq = np.mean(np.sqrt(np.sum(np.square(stft), axis=0)), axis=0)

    frame_length = 2048
    hop_length = 512

    # Compute pitch
    pitch, mag = librosa.core.piptrack(y=y, sr=sr, hop_length=hop_length, fmin=75, fmax=300)

    # Compute HNR
    HNR = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Compute RPDE
    RPDE = librosa.feature.spectral_flatness(y=y)

    # Compute DFA
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

        # Combine features into a feature vector
        # feature_vector = np.concatenate([ HNR.mean(axis=1), RPDE.mean(axis=1), DFA.mean(axis=1)])

    # else:
    #     mdvp_jitter_percent = 0
    #     mdvp_jitter_abs = 0
    #     mdvp_rap = 0
    #     mdvp_ppq = 0
    #     jitter_ddp = 0
    #     shimmer = 0
    #     shimmer_db = 0
    #     apq3 = 0
    #     apq5 = 0
    #     mdvp_apq = 0
    #     dda = 0
    #     HNR = 0
    #     RPDE = 0
    #     DFA = 0

    mvdp_shimmer = 0.0145
    mdvp_shimmerdb = 0.154
    shimmer_apq3 = 0.00469
    shimmer_apq5 = 0.00747
    shimmer_dda = 0.01567
    nhr = 0.00231

    # if HNR.size > 0:
    #     hnr_mean = np.mean(HNR.mean(axis=1))
    # else:
    #     hnr_mean = 0.0 
    print(type(HNR))
    # hnr_mean = HNR
    # rpde_mean = RPDE
    # dfa_mean = DFA
    # spread1_mean = 0.1
    # spread11 = []
    # if len(spread1) > 1:
    #     for i in range(len(spread1)-1):
    #         spread11.append(spread1[i+1] - spread1[i])
    # spread1_mean = np.mean(spread11)
    # spread2_mean = 0.2
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


    # Prepare the input data
    input = {'MDVP:Fo(Hz)': 119.992, 'MDVP:Fhi(Hz)': 157.302, 'MDVP:Flo(Hz)': 74.997, 
                  'MDVP:Jitter(%)': 0.00597, 'MDVP:Jitter(Abs)': 0, 'MDVP:RAP': 0.00303, 
                  'MDVP:PPQ': 0.0057, 'Jitter:DDP': 0.00909, 'MDVP:Shimmer': 0.04374, 
                  'MDVP:Shimmer(dB)': 0.426, 'Shimmer:APQ3': 0.02182, 'Shimmer:APQ5': 0.0313, 
                  'MDVP:APQ': 0.02971, 'Shimmer:DDA': 0.06545, 'NHR': 0.02211, 'HNR': 21.033, 
                  'RPDE': 0.414783, 'DFA': 0.815285, 'spread1': -4.81303, 'spread2': 0.266482, 
                  'D2': 2.30144, 'PPE': 0.284654}

    # input_data = {'MDVP:Fo(Hz)': 109.63, 'MDVP:Fhi(Hz)': 600.00, 'MDVP:Flo(Hz)': 65.476, 'MDVP:Jitter(%)': 82.52149820327759, 'MDVP:Jitter(Abs)': 0.02198106, 'MDVP:RAP': 0.12880914, 'MDVP:PPQ': 0.0011341888, 'Jitter:DDP': 0.2576182782649994, 'MDVP:Shimmer': 0.0145, 'MDVP:Shimmer(dB)': 0.154, 'Shimmer:APQ3': 0.00469, 'Shimmer:APQ5': 0.00747, 'MDVP:APQ': 34.477165, 'Shimmer:DDA': 0.01567, 'NHR': 0.00231, 'HNR': 19.554908223922464, 'RPDE': 0.048343565, 'DFA': 0.10291431568287036, 'spread1': 2090.8923575953118, 'spread2': 1116.43715330105, 'D2': -5.380089532535223, 'PPE': 13.985601}
    input_df = pd.DataFrame([input_data])

    # Make predictions
    y_pred = model.predict(input_df)

    # Calculate accuracy
    y_actual = 1  # Replace with the actual target variable value for the new input
    accuracy = (y_pred == y_actual).sum() / len(y_pred)

    # Print the predicted target variable value and accuracy
    # print('Predicted target variable value:', y_pred[0])
    # print('Accuracy:', accuracy)

    return y_pred[0]

# print(audioOut())

base64_string = "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"


def base64_to_wav( ):
    # Decode the base64-encoded string
    audio_bytes = base64.b64decode(base64_string)
    output_file = 'output2.wav'


    # Open the output WAV file for writing
    with wave.open(output_file, 'wb') as wave_write:
        # Set the parameters for the WAV file (1 channel, 16 bits per sample, 44100 Hz)
        wave_write.setnchannels(1)
        wave_write.setsampwidth(2)
        wave_write.setframerate(44100)

        # Write the audio data to the WAV file
        wave_write.writeframes(audio_bytes)

    #     # Read the audio data into a bytes object
    #     audio_data = wave_read.readframes(num_frames)

    
    # # Create a new Wave_write object for the output file
    # with wave.open('output.wav', 'wb') as wave_write:
    #     # Set the parameters of the output file
    #     wave_write.setnchannels(num_channels)
    #     wave_write.setsampwidth(sample_width)
    #     wave_write.setframerate(sample_rate)

    #     # Write the audio data to the output file
    #     wave_write.writeframes(audio_data)

# base64_to_wav()

print("Final Output",predictOut(audioOut('output2.wav')))
