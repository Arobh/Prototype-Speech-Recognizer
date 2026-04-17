from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
import json
import os
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - used for setup guidance at runtime.
    av = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - used for setup guidance at runtime.
    WhisperModel = None

try:
    from vosk import KaldiRecognizer, Model as VoskModel
except ImportError:  # pragma: no cover - used for setup guidance at runtime.
    KaldiRecognizer = None
    VoskModel = None


ROOT = Path(__file__).resolve().parent
ALLOWED_MODELS = {"tiny", "base", "small"}
DEFAULT_VOSK_MODEL = ROOT / "models" / "vosk-model-small-en-us-0.15"

app = FastAPI(title="Prototype Speech Recognizer")
loaded_models: dict[str, WhisperModel] = {}
loaded_vosk_model: VoskModel | None = None


def get_model(model_size: str) -> WhisperModel:
    if WhisperModel is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "faster-whisper is not installed. Run: "
                "pip install -r requirements.txt"
            ),
        )

    if model_size not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported Whisper model.")

    if model_size not in loaded_models:
        loaded_models[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    return loaded_models[model_size]


def get_vosk_model() -> VoskModel:
    global loaded_vosk_model

    if VoskModel is None or KaldiRecognizer is None:
        raise HTTPException(
            status_code=500,
            detail="vosk is not installed. Run: pip install -r requirements.txt",
        )

    model_path = Path(os.environ.get("VOSK_MODEL_PATH", DEFAULT_VOSK_MODEL))

    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Vosk model not found. Run download_vosk_model.bat or set "
                "VOSK_MODEL_PATH to a local Vosk model folder."
            ),
        )

    if loaded_vosk_model is None:
        loaded_vosk_model = VoskModel(str(model_path))

    return loaded_vosk_model


def save_upload(audio: UploadFile, content: bytes) -> Path:
    suffix = Path(audio.filename or "speech.webm").suffix or ".webm"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_path = Path(temp_audio.name)
        temp_audio.write(content)

    return temp_path


def write_wav(audio: np.ndarray, sample_rate: int) -> Path:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = Path(temp_audio.name)

    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return temp_path


def decode_audio(path: Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    if av is None:
        raise HTTPException(
            status_code=500,
            detail="PyAV is not installed. Run: pip install -r requirements.txt",
        )

    samples = []
    sample_rate = target_rate

    try:
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                sample_rate = frame.sample_rate or sample_rate
                array = frame.to_ndarray().astype(np.float32)

                if array.ndim > 1:
                    array = array.mean(axis=0)

                samples.append(array.reshape(-1))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    if not samples:
        return np.zeros(0, dtype=np.float32), target_rate

    audio = np.concatenate(samples)
    peak = float(np.max(np.abs(audio))) or 1.0

    if peak > 1.0:
        audio = audio / peak

    if sample_rate != target_rate and len(audio) > 1:
        duration = len(audio) / sample_rate
        target_length = max(1, int(duration * target_rate))
        source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
        sample_rate = target_rate

    return audio.astype(np.float32), sample_rate


def add_white_noise(audio: np.ndarray, target_snr_db: float | None) -> np.ndarray:
    if target_snr_db is None or not len(audio):
        return audio

    signal_power = float(np.mean(audio**2))

    if signal_power <= 0:
        return audio

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, len(audio)).astype(np.float32)
    noise_power = float(np.mean(noise**2)) or 1.0
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(audio + scaled_noise, -1.0, 1.0).astype(np.float32)


def downsample_matrix(matrix: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    if matrix.size == 0:
        return matrix

    row_indices = np.linspace(0, matrix.shape[0] - 1, min(max_rows, matrix.shape[0])).astype(int)
    col_indices = np.linspace(0, matrix.shape[1] - 1, min(max_cols, matrix.shape[1])).astype(int)
    return matrix[np.ix_(row_indices, col_indices)]


def stft_power(audio: np.ndarray, frame_length: int = 512, hop_length: int = 160) -> np.ndarray:
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    frame_count = 1 + max(0, (len(audio) - frame_length) // hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    frames = np.stack(
        [
            audio[index * hop_length : index * hop_length + frame_length] * window
            for index in range(frame_count)
        ]
    )
    spectrum = np.fft.rfft(frames, axis=1)
    return (np.abs(spectrum) ** 2).T


def normalize_matrix(matrix: np.ndarray) -> list[list[float]]:
    if matrix.size == 0:
        return []

    matrix = matrix.astype(np.float32)
    low = float(np.percentile(matrix, 5))
    high = float(np.percentile(matrix, 95))

    if high <= low:
        return np.zeros_like(matrix).tolist()

    matrix = np.clip((matrix - low) / (high - low), 0.0, 1.0)
    return matrix.tolist()


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filter_bank(sample_rate: int, fft_size: int, filter_count: int = 26) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2]))[0], filter_count + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((filter_count, fft_size // 2 + 1), dtype=np.float32)

    for index in range(1, filter_count + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]

        if center > left:
            filters[index - 1, left:center] = (np.arange(left, center) - left) / (center - left)

        if right > center:
            filters[index - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filters


def dct_type_two(values: np.ndarray, coefficient_count: int = 13) -> np.ndarray:
    filter_count = values.shape[0]
    indices = np.arange(filter_count)
    coefficients = []

    for coefficient in range(coefficient_count):
        basis = np.cos(np.pi * coefficient * (2 * indices + 1) / (2 * filter_count))
        coefficients.append(np.dot(basis, values))

    return np.array(coefficients, dtype=np.float32)


def compute_mfcc(power: np.ndarray, sample_rate: int) -> np.ndarray:
    filters = mel_filter_bank(sample_rate, 512)
    mel_energy = np.maximum(filters @ power, 1e-10)
    log_mel = np.log(mel_energy)
    return np.stack([dct_type_two(log_mel[:, frame]) for frame in range(log_mel.shape[1])], axis=1)


def analyze_audio_file(path: Path, noise_snr_db: float | None = None) -> dict[str, object]:
    audio, sample_rate = decode_audio(path)
    audio = add_white_noise(audio, noise_snr_db)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0

    frame_length = 512
    hop_length = 160
    if len(audio) >= frame_length:
        frame_rms = np.array(
            [
                np.sqrt(np.mean(audio[index : index + frame_length] ** 2))
                for index in range(0, len(audio) - frame_length + 1, hop_length)
            ]
        )
    else:
        frame_rms = np.array([rms])

    noise_floor = float(np.percentile(frame_rms, 10)) if frame_rms.size else 0.0
    speech_level = float(np.percentile(frame_rms, 90)) if frame_rms.size else 0.0
    snr_db = 20 * np.log10((speech_level + 1e-9) / (noise_floor + 1e-9))
    power = stft_power(audio, frame_length=frame_length, hop_length=hop_length)
    log_power = 10 * np.log10(power + 1e-10)
    mfcc = compute_mfcc(power, sample_rate)

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "rms": round(rms, 6),
        "noise_floor": round(noise_floor, 6),
        "estimated_snr_db": round(float(snr_db), 2),
        "noise_augmentation_snr_db": noise_snr_db,
        "spectrogram": normalize_matrix(downsample_matrix(log_power, 96, 160)),
        "mfcc": normalize_matrix(downsample_matrix(mfcc, 13, 160)),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    model_size: str = Form("base"),
    noise_snr_db: str = Form(""),
) -> dict[str, object]:
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Upload an audio recording.")

    model = get_model(model_size)
    temp_path = save_upload(audio, await audio.read())
    transcription_path = temp_path
    augmented_path: Path | None = None

    try:
        noise_value = float(noise_snr_db) if noise_snr_db else None

        if noise_value is not None:
            decoded_audio, sample_rate = decode_audio(temp_path)
            augmented_path = write_wav(add_white_noise(decoded_audio, noise_value), sample_rate)
            transcription_path = augmented_path

        start_time = perf_counter()
        segments, info = model.transcribe(
            str(transcription_path),
            language=language or None,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        processing_time = perf_counter() - start_time

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "model": model_size,
            "noise_augmentation_snr_db": noise_value,
            "processing_time_seconds": round(processing_time, 3),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if augmented_path:
            augmented_path.unlink(missing_ok=True)

        temp_path.unlink(missing_ok=True)


@app.post("/transcribe-vosk")
async def transcribe_vosk(
    audio: UploadFile = File(...),
    noise_snr_db: str = Form(""),
) -> dict[str, object]:
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Upload an audio recording.")

    model = get_vosk_model()
    temp_path = save_upload(audio, await audio.read())
    wav_path: Path | None = None

    try:
        noise_value = float(noise_snr_db) if noise_snr_db else None
        decoded_audio, sample_rate = decode_audio(temp_path)
        decoded_audio = add_white_noise(decoded_audio, noise_value)
        wav_path = write_wav(decoded_audio, sample_rate)

        start_time = perf_counter()
        recognizer = KaldiRecognizer(model, sample_rate)

        with wave.open(str(wav_path), "rb") as wav_file:
            while True:
                data = wav_file.readframes(4000)

                if not data:
                    break

                recognizer.AcceptWaveform(data)

        result = json.loads(recognizer.FinalResult())
        processing_time = perf_counter() - start_time

        return {
            "text": result.get("text", "").strip(),
            "language": "en",
            "language_probability": None,
            "model": "vosk-small-en-us-0.15",
            "engine": "vosk",
            "noise_augmentation_snr_db": noise_value,
            "processing_time_seconds": round(processing_time, 3),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if wav_path:
            wav_path.unlink(missing_ok=True)

        temp_path.unlink(missing_ok=True)


@app.post("/analyze-audio")
async def analyze_audio(
    audio: UploadFile = File(...),
    noise_snr_db: str = Form(""),
) -> dict[str, object]:
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Upload an audio recording.")

    temp_path = save_upload(audio, await audio.read())

    try:
        noise_value = float(noise_snr_db) if noise_snr_db else None
        return analyze_audio_file(temp_path, noise_value)
    finally:
        temp_path.unlink(missing_ok=True)


app.mount("/", StaticFiles(directory=ROOT), name="static")
