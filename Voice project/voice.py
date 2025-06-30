import streamlit as st
import librosa
import numpy as np
import joblib
import io
import scipy.stats

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("voice_gender_model.pkl")

model = load_model()

# Top 15 features based on importance
feature_names = [
    "mfcc_13_std", "mfcc_13_mean", "mfcc_12_std", "mfcc_12_mean",
    "mfcc_11_std", "mfcc_11_mean", "mfcc_10_std", "mfcc_10_mean",
    "mfcc_9_std", "mfcc_9_mean", "mfcc_8_std", "mfcc_8_mean",
    "mfcc_7_std", "mfcc_7_mean", "mfcc_6_std"
]

# Feature extraction function for top 15 features only
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # Build the feature vector in the correct order
    features_dict = {
        f"mfcc_{i+1}_mean": mfcc_means[i] for i in range(13)
    }
    features_dict.update({
        f"mfcc_{i+1}_std": mfcc_stds[i] for i in range(13)
    })

    # Extract only the selected top 15 features
    features = np.array([features_dict[name] for name in feature_names])
    return features

# Streamlit UI
st.title("🎤 Voice Gender Recognition (Top 15 Features)")
st.write("Upload a voice recording (WAV, MP3, etc.) to predict the speaker's gender.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])

if uploaded_file is not None:
    try:
        audio_data, sr = librosa.load(uploaded_file, sr=None, res_type="kaiser_fast")
        features = extract_features(audio_data, sr)

        feature_info = dict(zip(feature_names, features))

        st.subheader("🔍 Extracted Features (Top 15)")
        st.json(feature_info)

        prediction = model.predict([features])[0]
        gender = "Male" if prediction == 1 else "Female"
        st.success(f"Predicted Gender: **{gender}**")

        st.audio(uploaded_file)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
