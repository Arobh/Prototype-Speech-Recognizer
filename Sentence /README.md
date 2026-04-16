# Prototype Speech Recognizer

This project implements a basic Automatic Speech Recognition (ASR) system using deep learning (CNN + LSTM + CTC) and compares its performance with the Whisper model.

---

## Dataset

The project uses the LibriSpeech dataset from OpenSLR:

- Training set: train-clean-100  
- Test set: test-clean  

Link: https://www.openslr.org/12/

---

## Features

- MFCC feature extraction from audio  
- CNN + LSTM based acoustic model  
- CTC loss for sequence prediction  
- Evaluation using Word Error Rate (WER) and Character Error Rate (CER)  
- Comparison with Whisper model  
- Microphone input support for live testing  

---

## Model Architecture

- Conv1D layer with Batch Normalization  
- Bidirectional LSTM layers  
- Dense layer with Softmax output  

---

## Results

| Model     | WER  | CER  |
|----------|------|------|
| My Model | 0.77 | ~0.5 |
| Whisper  | 0.31 | ~0.1 |

---

## How to Run

1. Download and extract the dataset  
2. Run preprocessing to extract MFCC features  
3. Train the model  
4. Evaluate on the test dataset  
5. Run inference using audio files or microphone input  

---

## Save and Load Model

Save the trained model:

```python
model.save("my_asr_model.h5")
