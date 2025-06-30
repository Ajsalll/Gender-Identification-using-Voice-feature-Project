# 🎤 Voice Gender Identification System

A machine learning-based web application that identifies the gender (Male/Female) of a speaker using voice recordings. Built using Python, Librosa for feature extraction, Scikit-learn for model training, and Streamlit for deployment.

---

## 📌 Project Overview

This project demonstrates how speech signals can be analyzed and classified based on gender using Mel Frequency Cepstral Coefficients (MFCCs). The system allows users to upload an audio file, extracts acoustic features, and predicts the speaker's gender in real time.

---

## 🧠 Features

- 🎙️ Upload audio files (`.wav`, `.mp3`, `.ogg`, `.m4a`)
- 📊 Extract top 15 MFCC-based statistical features
- 🤖 Predict gender using a trained Random Forest model
- 💡 Interactive UI using Streamlit
- 🔒 Local and lightweight — no data is stored

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Librosa, NumPy, Pandas, Scikit-learn, Joblib  
- **Web Framework:** Streamlit  
- **Model:** Random Forest Classifier

---

## 🧪 How It Works

1. **Audio Preprocessing:** The uploaded file is loaded and resampled using Librosa.
2. **Feature Extraction:** Top 15 MFCC features (means and standard deviations) are extracted.
3. **Prediction:** The features are passed into a pre-trained Random Forest model.
4. **Output:** Gender is predicted and displayed with extracted feature values.

---

## 🚀 Getting Started

### Prerequisites

Make sure the following packages are installed:
```bash
pip install streamlit librosa scikit-learn numpy pandas joblib
