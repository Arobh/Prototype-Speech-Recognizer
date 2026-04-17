# Prototype Speech Recognizer

This project is a prototype automatic speech recognition system for the course **Advanced Topics in Speech Processing**. The system takes spoken input from a microphone and converts it into readable text.

## Objective

The objective of this project is to demonstrate a basic speech-to-text pipeline where a user's voice is captured, processed by the Whisper automatic speech recognition model, and returned as text.

## Problem Statement

Human speech is a continuous acoustic signal. A speech recognizer must identify meaningful linguistic units from this signal and map them to written text. This prototype focuses on the final user-facing behavior of an automatic speech recognition system: when a person speaks, the system returns the recognized text.

## Methodology

The prototype uses **Whisper**, a deep learning automatic speech recognition model. The browser records the user's voice and sends the audio to a local Python backend, where Whisper performs transcription.

The speech-processing flow is:

1. Audio input is captured from the user's microphone.
2. A live waveform is drawn from the microphone signal.
3. A live spectrogram is drawn from the microphone signal.
4. The browser stores the recording as an audio blob.
5. The audio recording is sent to the Python backend.
6. The backend loads the selected Whisper model.
7. Whisper converts the speech signal into text.
8. Processing time is measured for the transcription.
9. Final recognized text is displayed in the browser.
10. The recognized text can be compared with a reference transcript using Word Error Rate.

## Speech Processing Concepts Demonstrated

- **Speech acquisition:** microphone input is used as the audio source.
- **Waveform visualization:** the time-domain audio signal is displayed while recording.
- **Spectrogram visualization:** frequency energy is displayed over time while recording.
- **Uploaded audio feature extraction:** uploaded samples are analyzed for spectrogram and MFCC features.
- **Noise testing:** uploaded samples show RMS, estimated noise floor, and estimated SNR.
- **Automatic speech recognition:** spoken words are converted into text using Whisper.
- **Deep learning ASR model:** Whisper is used as the recognition model.
- **Language selection:** recognition can be configured for selected languages or automatic detection.
- **Word Error Rate:** recognized text can be evaluated against a reference transcript.
- **Dataset testing:** uploaded audio samples can be transcribed and evaluated.
- **Human-computer interaction:** the interface provides start, stop, clear, and copy controls.

## Model Implemented

This project implements **Whisper ASR** through the `faster-whisper` Python package.

Available model options in the interface:

- `tiny`: fastest, lower accuracy.
- `base`: balanced speed and accuracy.
- `small`: better accuracy, slower than `base`.

The backend runs the model on CPU using int8 computation so it can work on a normal laptop without requiring a GPU.

## Features

- Start and stop microphone recording.
- Sends recorded speech to a local Whisper backend.
- Shows a live audio waveform during microphone recording.
- Shows a live spectrogram during microphone recording.
- Shows spectrogram and MFCC feature maps for uploaded audio.
- Measures Whisper processing time.
- Compares `tiny`, `base`, and `small` Whisper models on the same uploaded audio.
- Compares Whisper against a Vosk offline ASR baseline.
- Supports batch testing with multiple uploaded audio files.
- Adds optional white-noise augmentation at selected SNR levels.
- Records speaker, environment, microphone, and noise metadata.
- Gives a simple noise test summary using estimated SNR.
- Draws WER and processing-time charts by model.
- Shows word-level substitutions, deletions, and insertions.
- Exports dataset experiment results as CSV.
- Saves final recognized text.
- Supports multiple recognition languages.
- Calculates Word Error Rate against an expected transcript.
- Supports uploaded audio sample testing for dataset-style evaluation.
- Copies recognized text to the clipboard.
- Gives microphone and browser support feedback.

## How to Run

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Dependencies have already been installed in `.venv` for this local copy.

Run the Whisper backend:

```powershell
.\.venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 5500
```

If you are not using a virtual environment, run:

```powershell
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 5500
```

Then open:

```text
http://127.0.0.1:5500
```

You can also double-click `run_whisper_app.bat` after installing the dependencies.

If port `5500` is already busy, `run_whisper_app.bat` automatically chooses the next free port between `5500` and `5510` and prints the URL.

If you manually run uvicorn and see this error:

```text
[WinError 10048] only one usage of each socket address is normally permitted
```

It means another server is already using port `5500`. Either open the already-running app at `http://127.0.0.1:5500`, run `stop_server_5500.bat`, or start uvicorn on another port:

```powershell
.\.venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 5501
```

Use Google Chrome or Microsoft Edge for best microphone recording support. The first run may take time because the Whisper model has to be downloaded.

## Optional Vosk Baseline Setup

The Vosk Python package is included in `requirements.txt`, but Vosk also needs a local acoustic model. To download the small English model, run:

```powershell
.\download_vosk_model.bat
```

Then restart the app. The **Compare Vosk** button will compare the currently selected Whisper model against `vosk-model-small-en-us-0.15`.

## Expected Output

When the user says:

```text
speech processing is an important field
```

The application should display text similar to:

```text
speech processing is an important field
```

Recognition accuracy may vary depending on microphone quality, background noise, pronunciation, selected language, and selected Whisper model size.

## Word Error Rate Evaluation

The project includes a Word Error Rate evaluator. Enter the expected transcript, record speech, and click **Evaluate**. The app calculates:

```text
WER = word edits / total reference words
```

The word edits are computed using Levenshtein distance over words. Lower WER means better recognition accuracy.

## Dataset Audio Testing

The dataset test section lets you upload an audio file and paste its expected transcript. The app sends the audio file to Whisper, displays the model output, and calculates WER for that sample. This is useful for testing multiple speech samples from a small dataset.

You can upload multiple audio files at once. **Run Dataset Test** processes all selected files with the currently selected model. **Compare Models** processes all selected files with `tiny`, `base`, and `small` so speed and accuracy can be compared.

The **Noise augmentation** option adds synthetic white noise before transcription and feature extraction. Supported target SNR levels are `20 dB`, `10 dB`, and `5 dB`.

The dataset section also records experiment metadata:

| Field | Purpose |
| --- | --- |
| Speaker ID | Identifies the speaker or sample group. |
| Environment | Labels quiet, moderate-noise, or high-noise conditions. |
| Microphone | Records the recording device used. |
| Noise type | Describes background noise such as fan, traffic, or classroom. |

The result table summarizes:

| Speaker | Environment | Noise | Model | WER | Time | SNR | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| speaker_01 | quiet | fan | base | 8.33% | 2.41s | 18.7 dB | sample.wav |

Click **Download CSV** to export dataset experiment rows. The CSV includes metadata, reference text, recognized text, WER, processing time, audio duration, sample rate, RMS, noise floor, and estimated SNR.

The app also draws:

- **WER by Model:** average WER for each tested model.
- **Processing Time by Model:** average transcription time for each tested model.

## Error Analysis

For each evaluated transcript, the app shows:

- **Substitutions:** reference words recognized as different words.
- **Deletions:** reference words missed by the recognizer.
- **Insertions:** extra recognized words that were not in the reference.

These values are exported in the CSV as `substitutions`, `deletions`, and `insertions`.

## Feature Visualization

Uploaded audio is analyzed by the Python backend. The app displays:

- **Uploaded spectrogram:** frequency energy over time.
- **MFCC feature map:** compact cepstral features commonly used in speech processing.
- **Noise summary:** duration, RMS level, and estimated SNR.

## Limitations

- The current prototype performs transcription after the user stops recording, not continuously word by word.
- The first run requires downloading the selected Whisper model.
- Recognition accuracy can decrease in noisy environments.
- The noise estimate is approximate and is based on frame-level energy statistics.
- Noise augmentation uses synthetic white noise, not real environmental noise recordings.
- Vosk comparison requires downloading a local Vosk model first.
- Custom ASR model training is outside the current scope.

## Future Enhancements

- Add noise reduction before recognition.
- Save recognized transcripts to a file.
- Add speaker-independent and multilingual testing.
- Add real-noise datasets such as classroom, traffic, and fan recordings.
- Add Hindi/Tamil/Telugu Vosk models for non-English baseline testing.

## Conclusion

The project successfully demonstrates a working Whisper-based speech recognition prototype. When a person speaks into the microphone, the application records the speech input, sends it to the Whisper ASR backend, and returns the corresponding text output on the screen.
