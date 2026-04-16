# 📊 Dataset: Speech Commands v0.02

This project uses the **Speech Commands Dataset v0.02**, a collection of short audio clips of spoken words designed for training speech recognition and word prediction models.

---

##  Dataset Overview

- Total Files: ~105,000+
- Audio Format: `.wav`
- Duration: ~1 second per clip
- Sample Rate: 16 kHz
- Language: English
- Type: Single-word utterances

The dataset contains recordings of different people speaking simple command words.

---

##  Classes (Words)

### Core Words (20 commands)
yes, no, up, down, left, right, on, off, stop, go,
zero, one, two, three, four, five, six, seven, eight, nine

### Auxiliary Words (10 words)
bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow

---

## 📦 Dataset Structure

speech_commands/
│── yes/
│   ├── 0a7c2a8d_nohash_0.wav
│── no/
│── up/
│── down/
│── ...
│── _background_noise_/
│── validation_list.txt
│── testing_list.txt

- Each folder represents a **label (word)**
- Each file contains a **spoken version of that word**

---

##  Data Splitting

- Training set (majority)
- Validation set (~10%)
- Test set (~10%)

Files are assigned using a **hash-based splitting method** to ensure consistency.

---

##  Data Collection

- Collected via crowdsourcing
- Speakers recorded words in real-world environments
- Designed for **robust speech recognition in noisy conditions**

---

##  Preprocessing

- Converted to 16-bit PCM WAV
- Sample rate: 16,000 Hz
- Trimmed to 1-second clips
- Organized by word labels

---

##  Source

Original dataset:
http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Research Paper:
https://arxiv.org/abs/1804.03209

---

##  License

Creative Commons Attribution 4.0 (CC BY 4.0)

---

##  Usage in This Project

This dataset is used to:

- Train a speech recognition / word prediction model
- Classify spoken commands into predefined labels
- Evaluate model performance using WER (Word Error Rate)

---

##  Credits

Dataset created by:
Pete Warden (Google AI)
