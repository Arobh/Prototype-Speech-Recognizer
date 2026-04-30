# Prototype Speech Recognizer

A complete end-to-end Automatic Speech Recognition (ASR) system built as part of the Advanced Topics in Speech Processing (CS60116) course at IIT Kharagpur.

This project explores the evolution of speech recognition — from classical ML to deep learning and real-time transcription using Whisper.

---

## Overview

This project is organized into 4 phases:

1. Word-Level Recognition  
   - MFCC feature extraction  
   - Classical ML models (SVM, Random Forest, XGBoost)  
   - CNN on MFCC spectrograms  

2. Sentence-Level Classification  
   - Synthetic dataset using TTS  
   - 120-D MFCC features  
   - ML classifiers  

3. End-to-End ASR  
   - Conv1D + BiLSTM  
   - CTC Loss  
   - LibriSpeech dataset  

4. Real-Time Web App  
   - Whisper-based transcription  
   - FastAPI backend  
   - Browser interface  

---

## Project Structure

```
├── PPT/
│   └── SpeechRecognizer.pptx
├── Prototype_Speech_Recognizer/
│   ├── PROJECT_REPORT.md
│   ├── README.md
│   ├── app.js
│   ├── server.py
│   ├── index.html
│   ├── styles.css
│   ├── requirements.txt
│   └── scripts (.bat files)
├── Report/
│   └── atsp_project.pdf
├── Sentence/
│   ├── DL/
│   │   ├── model (.h5)
│   │   └── README.md
├── ML/
│   ├── speech-project(1).ipynb
│   └── README.md
├── Word/
│   ├── notebooks/
│   │   ├── 01_dataset_exploration.ipynb
│   │   ├── 02_feature_extraction.ipynb
│   │   ├── 03_model_training.ipynb
│   │   ├── 04_cnn_features.ipynb
│   │   ├── 05_cnn_model.ipynb
│   │   └── 06_prediction_demo.ipynb
│   └── README.md
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Arobh/Prototype-Speech-Recognizer.git
cd Prototype-Speech-Recognizer
```

Install dependencies:

```
pip install -r Prototype_Speech_Recognizer/requirements.txt
```

---

## How to Run

### Run Web App (Whisper)

```
cd Prototype_Speech_Recognizer
python server.py
```

Open browser:
```
http://localhost:8000
```

---

### Run Notebooks

Open Jupyter:

```
jupyter notebook
```

Navigate to:
- Word/notebooks/
- ML/
- Sentence/DL/

---

## Results

- Word-Level: SVM, XGBoost best performers  
- CNN improved accuracy using MFCC images  

- Sentence-Level: ~88–95% accuracy (SVM)  

- End-to-End Model:  
  - WER ≈ 0.77  

- Whisper:  
  - WER ≈ 0.31 (base)  
  - Near 0% on clean speech  

---

## Evaluation Metric

WER = (S + D + I) / N  

---

## Technologies Used

- Python  
- librosa  
- scikit-learn  
- TensorFlow / Keras  
- XGBoost  
- FastAPI  
- Whisper  
- NumPy, Pandas  

---

## Challenges

- Large dataset processing  
- MFCC loses temporal info  
- TTS vs real speech mismatch  
- CTC instability  

---

## Future Work

- Real speech datasets  
- Conformer models  
- Whisper fine-tuning  
- Streaming ASR  
- Indian language support  

---

## Authors

Aarobh Kumar  
Apeksha S. Gulhane  

Under Prof. K. Sreenivasa Rao  
IIT Kharagpur  

---

## License

Academic use only.
