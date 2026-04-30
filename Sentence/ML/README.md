# Prototype Speech Recognizer

This project implements a basic speech recognition pipeline that converts `.wav` audio files into text using machine learning. It is built as a single notebook that demonstrates the full workflow from data preparation to prediction.

---

## Project Overview

The notebook processes audio input, extracts features, trains machine learning models, and predicts the corresponding text output. It also includes data augmentation to simulate real-world conditions.

---

## Features

* Synthetic speech data generation using text-to-speech
* Audio augmentation (noise, pitch shift, speed changes)
* Feature extraction using MFCC and related features
* Training and evaluation of multiple ML models
* Automatic selection of the best model
* Prediction with confidence scores
* Basic audio visualization

---

## Tech Stack

* Python
* NumPy, Pandas
* Librosa
* Scikit-learn
* Matplotlib, Seaborn
* Pyttsx3

---

## Project Structure

```id="x3p9zk"
speech-project(1).ipynb   # Complete implementation (data, training, prediction)
```

---

## How It Works

### 1. Data Generation

Text sentences are converted into audio using a text-to-speech engine.

### 2. Data Augmentation

To improve robustness, the following transformations are applied:

* Noise addition
* Pitch shifting
* Speed variation

### 3. Feature Extraction

Audio is converted into numerical features using:

* MFCC (Mel Frequency Cepstral Coefficients)
* Delta features
* Statistical summaries

### 4. Model Training

The notebook trains and compares:

* Support Vector Machine
* Random Forest
* Gradient Boosting

The best-performing model is selected.

### 5. Prediction

The trained model predicts text from an input audio file along with a confidence score.

Example:

```python id="p3k2x1"
result = recognize_speech("audio.wav")

print(result["transcript"])
print(result["confidence"])
```

---

## Installation

Install required dependencies:

```id="j7m4q2"
pip install numpy pandas librosa scikit-learn matplotlib seaborn pyttsx3
```

For Linux systems:

```id="o9r1t6"
apt-get install espeak-ng
```

---

## Output

* Predicted transcript
* Confidence score

---

## Future Improvements

* Add deep learning models (CNN, RNN, Transformers)
* Support real-time microphone input
* Use larger real-world datasets
* Improve accuracy and scalability

---
