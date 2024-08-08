seed_value = 42
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import opensmile

path = 'shemo'  
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]


def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]


def opensmile_Functionals():
    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
        sampling_rate=16000, resample=True,
    )
    features = []
    emotions = []
    for file in os.listdir(path):
        if emo_labels[get_emotion_label(file)] != 'fear':
            df = feature_extractor.process_file(f'{path}/{file}')
            features.append(df)
            emotions.append(get_emotion_label(file))
    features = np.array(features).squeeze()
    emotions = np.array(emotions)
    return features, emotions



if __name__ == "__main__":
    features, emotions = opensmile_Functionals()
    np.save('features.npy', features)  
    np.save('emotions.npy', emotions)  

import os
import numpy as np
import librosa
seed_value = 42
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Define paths and labels
path = 'shemo'  # Path to your audio files
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]

def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]

def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)

    # Compute features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
    
    # Combine all features
    return np.concatenate([mfccs, chroma, mel, contrast, tonnetz])

def opensmile_Functionals():
    features = []
    emotions = []
    for file in os.listdir(path):
        if emo_labels[get_emotion_label(file)] != 'fear':
            file_path = os.path.join(path, file)
            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            emotions.append(get_emotion_label(file))
    features = np.array(features)
    emotions = np.array(emotions)
    return features, emotions

if __name__ == "__main__":
    features, emotions = opensmile_Functionals(path)
    np.save('features.npy', features)
    np.save('emotions.npy', emotions)
