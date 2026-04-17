const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
const copyButton = document.querySelector("#copy-button");
const evaluateButton = document.querySelector("#evaluate-button");
const datasetButton = document.querySelector("#dataset-button");
const compareModelsButton = document.querySelector("#compare-models-button");
const compareVoskButton = document.querySelector("#compare-vosk-button");
const exportCsvButton = document.querySelector("#export-csv-button");
const languageSelect = document.querySelector("#language-select");
const modelSelect = document.querySelector("#model-select");
const statusPill = document.querySelector("#status-pill");
const supportNote = document.querySelector("#support-note");
const finalTranscript = document.querySelector("#final-transcript");
const interimTranscript = document.querySelector("#interim-transcript");
const placeholder = document.querySelector("#placeholder");
const meter = document.querySelector(".meter");
const waveformCanvas = document.querySelector("#waveform-canvas");
const spectrogramCanvas = document.querySelector("#spectrogram-canvas");
const referenceText = document.querySelector("#reference-text");
const werResult = document.querySelector("#wer-result");
const datasetAudio = document.querySelector("#dataset-audio");
const datasetReference = document.querySelector("#dataset-reference");
const datasetTranscript = document.querySelector("#dataset-transcript");
const datasetResult = document.querySelector("#dataset-result");
const datasetTime = document.querySelector("#dataset-time");
const noiseResult = document.querySelector("#noise-result");
const substitutionOutput = document.querySelector("#substitution-output");
const deletionOutput = document.querySelector("#deletion-output");
const insertionOutput = document.querySelector("#insertion-output");
const speakerId = document.querySelector("#speaker-id");
const environmentSelect = document.querySelector("#environment-select");
const microphoneType = document.querySelector("#microphone-type");
const noiseLabel = document.querySelector("#noise-label");
const noiseAugmentationSelect = document.querySelector("#noise-augmentation-select");
const uploadedSpectrogramCanvas = document.querySelector("#uploaded-spectrogram-canvas");
const mfccCanvas = document.querySelector("#mfcc-canvas");
const werChartCanvas = document.querySelector("#wer-chart-canvas");
const timeChartCanvas = document.querySelector("#time-chart-canvas");
const resultsTableBody = document.querySelector("#results-table-body");

const waveformContext = waveformCanvas.getContext("2d");
const spectrogramContext = spectrogramCanvas.getContext("2d");
const uploadedSpectrogramContext = uploadedSpectrogramCanvas.getContext("2d");
const mfccContext = mfccCanvas.getContext("2d");
const werChartContext = werChartCanvas.getContext("2d");
const timeChartContext = timeChartCanvas.getContext("2d");

let mediaRecorder;
let audioStream;
let audioChunks = [];
let finalText = "";
let audioContext;
let analyser;
let waveformFrame;
let spectrogramFrame;
let datasetResults = [];

function setStatus(text, isListening = false) {
  statusPill.textContent = text;
  statusPill.classList.toggle("listening", isListening);
  meter.classList.toggle("active", isListening);
}

function syncTranscript() {
  finalTranscript.textContent = finalText;
  placeholder.hidden = Boolean(finalText || interimTranscript.textContent);
}

function setRecordingState(isRecording) {
  startButton.disabled = isRecording;
  stopButton.disabled = !isRecording;
  languageSelect.disabled = isRecording;
  modelSelect.disabled = isRecording;
}

function getBestMimeType() {
  const supportedTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];

  return supportedTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function normalizeText(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function calculateWer(reference, hypothesis) {
  const referenceWords = normalizeText(reference).split(" ").filter(Boolean);
  const hypothesisWords = normalizeText(hypothesis).split(" ").filter(Boolean);
  const rows = referenceWords.length + 1;
  const cols = hypothesisWords.length + 1;
  const distances = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let row = 0; row < rows; row += 1) {
    distances[row][0] = row;
  }

  for (let col = 0; col < cols; col += 1) {
    distances[0][col] = col;
  }

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const substitutionCost =
        referenceWords[row - 1] === hypothesisWords[col - 1] ? 0 : 1;

      distances[row][col] = Math.min(
        distances[row - 1][col] + 1,
        distances[row][col - 1] + 1,
        distances[row - 1][col - 1] + substitutionCost,
      );
    }
  }

  const edits = distances[referenceWords.length][hypothesisWords.length];
  const wordCount = referenceWords.length;

  return {
    edits,
    wordCount,
    wer: wordCount === 0 ? 0 : edits / wordCount,
  };
}

function analyzeWordErrors(reference, hypothesis) {
  const referenceWords = normalizeText(reference).split(" ").filter(Boolean);
  const hypothesisWords = normalizeText(hypothesis).split(" ").filter(Boolean);
  const rows = referenceWords.length + 1;
  const cols = hypothesisWords.length + 1;
  const distances = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let row = 0; row < rows; row += 1) {
    distances[row][0] = row;
  }

  for (let col = 0; col < cols; col += 1) {
    distances[0][col] = col;
  }

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const substitutionCost =
        referenceWords[row - 1] === hypothesisWords[col - 1] ? 0 : 1;

      distances[row][col] = Math.min(
        distances[row - 1][col] + 1,
        distances[row][col - 1] + 1,
        distances[row - 1][col - 1] + substitutionCost,
      );
    }
  }

  const substitutions = [];
  const deletions = [];
  const insertions = [];
  let row = referenceWords.length;
  let col = hypothesisWords.length;

  while (row > 0 || col > 0) {
    if (
      row > 0 &&
      col > 0 &&
      referenceWords[row - 1] === hypothesisWords[col - 1] &&
      distances[row][col] === distances[row - 1][col - 1]
    ) {
      row -= 1;
      col -= 1;
    } else if (
      row > 0 &&
      col > 0 &&
      distances[row][col] === distances[row - 1][col - 1] + 1
    ) {
      substitutions.push(`${referenceWords[row - 1]} -> ${hypothesisWords[col - 1]}`);
      row -= 1;
      col -= 1;
    } else if (row > 0 && distances[row][col] === distances[row - 1][col] + 1) {
      deletions.push(referenceWords[row - 1]);
      row -= 1;
    } else {
      insertions.push(hypothesisWords[col - 1]);
      col -= 1;
    }
  }

  return {
    substitutions: substitutions.reverse(),
    deletions: deletions.reverse(),
    insertions: insertions.reverse(),
  };
}

function renderErrorAnalysis(reference, hypothesis) {
  if (!reference.trim() || !hypothesis.trim()) {
    substitutionOutput.textContent = "None yet";
    deletionOutput.textContent = "None yet";
    insertionOutput.textContent = "None yet";
    return {
      substitutions: [],
      deletions: [],
      insertions: [],
    };
  }

  const errors = analyzeWordErrors(reference, hypothesis);
  substitutionOutput.textContent = errors.substitutions.join(", ") || "None";
  deletionOutput.textContent = errors.deletions.join(", ") || "None";
  insertionOutput.textContent = errors.insertions.join(", ") || "None";
  return errors;
}

function renderWer(target, reference, hypothesis) {
  if (!reference.trim()) {
    target.textContent = "Add the expected transcript first.";
    return null;
  }

  if (!hypothesis.trim()) {
    target.textContent = "No recognized text is available for evaluation.";
    return null;
  }

  const result = calculateWer(reference, hypothesis);
  const percent = (result.wer * 100).toFixed(2);
  target.textContent = `WER: ${percent}% (${result.edits} word edits / ${result.wordCount} reference words)`;
  return result;
}

function resizeWaveformCanvas() {
  const pixelRatio = window.devicePixelRatio || 1;
  const width = waveformCanvas.clientWidth;
  const height = waveformCanvas.clientHeight;
  waveformCanvas.width = Math.max(1, Math.floor(width * pixelRatio));
  waveformCanvas.height = Math.max(1, Math.floor(height * pixelRatio));
  waveformContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function resizeSpectrogramCanvas() {
  const width = spectrogramCanvas.clientWidth;
  const height = spectrogramCanvas.clientHeight;
  spectrogramCanvas.width = Math.max(1, Math.floor(width));
  spectrogramCanvas.height = Math.max(1, Math.floor(height));
  spectrogramContext.setTransform(1, 0, 0, 1, 0, 0);
}

function drawFlatWaveform() {
  resizeWaveformCanvas();
  const width = waveformCanvas.clientWidth;
  const height = waveformCanvas.clientHeight;

  waveformContext.clearRect(0, 0, width, height);
  waveformContext.strokeStyle = "#9ca3af";
  waveformContext.lineWidth = 2;
  waveformContext.beginPath();
  waveformContext.moveTo(0, height / 2);
  waveformContext.lineTo(width, height / 2);
  waveformContext.stroke();
}

function drawEmptySpectrogram() {
  resizeSpectrogramCanvas();
  const width = spectrogramCanvas.width;
  const height = spectrogramCanvas.height;

  spectrogramContext.fillStyle = "#121820";
  spectrogramContext.fillRect(0, 0, width, height);
  spectrogramContext.fillStyle = "rgba(255, 255, 255, 0.62)";
  spectrogramContext.font = "13px sans-serif";
  spectrogramContext.fillText("Frequency energy appears here while recording", 14, 24);
}

function getSpectrogramColor(value) {
  const normalized = value / 255;
  const hue = 220 - normalized * 170;
  const lightness = 12 + normalized * 56;
  return `hsl(${hue}, 86%, ${lightness}%)`;
}

function getHeatmapColor(value) {
  const hue = 230 - value * 185;
  const lightness = 13 + value * 58;
  return `hsl(${hue}, 86%, ${lightness}%)`;
}

function drawFeaturePlaceholder(canvas, context, text) {
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight));
  context.fillStyle = "#121820";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(255, 255, 255, 0.62)";
  context.font = "13px sans-serif";
  context.fillText(text, 14, 24);
}

function drawMatrix(canvas, context, matrix, emptyText) {
  if (!matrix?.length || !matrix[0]?.length) {
    drawFeaturePlaceholder(canvas, context, emptyText);
    return;
  }

  canvas.width = Math.max(1, Math.floor(canvas.clientWidth));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight));

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellWidth = canvas.width / cols;
  const cellHeight = canvas.height / rows;

  context.fillStyle = "#121820";
  context.fillRect(0, 0, canvas.width, canvas.height);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = Math.max(0, Math.min(1, matrix[row][col]));
      context.fillStyle = getHeatmapColor(value);
      context.fillRect(
        col * cellWidth,
        (rows - row - 1) * cellHeight,
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }
}

function getNoiseAugmentationValue() {
  return noiseAugmentationSelect.value;
}

function getNoiseAugmentationLabel() {
  return getNoiseAugmentationValue()
    ? `${getNoiseAugmentationValue()} dB`
    : "none";
}

function escapeCsv(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function downloadCsv() {
  if (!datasetResults.length) {
    datasetResult.textContent = "Run at least one dataset test before exporting.";
    return;
  }

  const headers = [
    "timestamp",
    "file_name",
    "speaker_id",
    "environment",
    "microphone",
    "noise_type",
    "language",
    "model",
    "engine",
    "reference_text",
    "recognized_text",
    "wer_percent",
    "word_edits",
    "reference_words",
    "processing_time_seconds",
    "duration_seconds",
    "sample_rate",
    "rms",
    "noise_floor",
    "estimated_snr_db",
    "noise_augmentation_snr_db",
    "substitutions",
    "deletions",
    "insertions",
  ];
  const rows = datasetResults.map((result) =>
    headers.map((header) => escapeCsv(result[header])).join(","),
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = `speech-recognizer-results-${new Date()
    .toISOString()
    .slice(0, 19)
    .replaceAll(":", "-")}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function drawChartPlaceholder(canvas, context, text) {
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight));
  context.fillStyle = "#fbfcfc";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#6b7280";
  context.font = "13px sans-serif";
  context.fillText(text, 14, 24);
}

function averageByModel(metric) {
  const groups = new Map();

  for (const result of datasetResults) {
    const value = Number(result[metric]);

    if (Number.isFinite(value)) {
      const values = groups.get(result.model) || [];
      values.push(value);
      groups.set(result.model, values);
    }
  }

  return [...groups.entries()].map(([model, values]) => ({
    label: model,
    value: values.reduce((sum, value) => sum + value, 0) / values.length,
  }));
}

function drawBarChart(canvas, context, values, title, suffix) {
  if (!values.length) {
    drawChartPlaceholder(canvas, context, title);
    return;
  }

  canvas.width = Math.max(1, Math.floor(canvas.clientWidth));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight));

  const width = canvas.width;
  const height = canvas.height;
  const padding = 34;
  const maxValue = Math.max(...values.map((item) => item.value), 1);
  const barWidth = (width - padding * 2) / values.length;

  context.fillStyle = "#fbfcfc";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(28, 31, 36, 0.16)";
  context.beginPath();
  context.moveTo(padding, 16);
  context.lineTo(padding, height - padding);
  context.lineTo(width - 12, height - padding);
  context.stroke();

  values.forEach((item, index) => {
    const barHeight = ((height - padding * 2) * item.value) / maxValue;
    const x = padding + index * barWidth + barWidth * 0.18;
    const y = height - padding - barHeight;
    const actualWidth = barWidth * 0.64;

    context.fillStyle = "#0f766e";
    context.fillRect(x, y, actualWidth, barHeight);
    context.fillStyle = "#1c1f24";
    context.font = "12px sans-serif";
    context.fillText(item.label, x, height - 12);
    context.fillText(`${item.value.toFixed(2)}${suffix}`, x, Math.max(14, y - 6));
  });
}

function updateCharts() {
  drawBarChart(
    werChartCanvas,
    werChartContext,
    averageByModel("wer_percent"),
    "Run tests to generate WER chart",
    "%",
  );
  drawBarChart(
    timeChartCanvas,
    timeChartContext,
    averageByModel("processing_time_seconds"),
    "Run tests to generate timing chart",
    "s",
  );
}

function drawWaveform() {
  if (!analyser) {
    return;
  }

  const width = waveformCanvas.clientWidth;
  const height = waveformCanvas.clientHeight;
  const samples = new Uint8Array(analyser.fftSize);

  analyser.getByteTimeDomainData(samples);
  waveformContext.clearRect(0, 0, width, height);
  waveformContext.lineWidth = 2;
  waveformContext.strokeStyle = "#0f766e";
  waveformContext.beginPath();

  for (let index = 0; index < samples.length; index += 1) {
    const x = (index / (samples.length - 1)) * width;
    const y = (samples[index] / 255) * height;

    if (index === 0) {
      waveformContext.moveTo(x, y);
    } else {
      waveformContext.lineTo(x, y);
    }
  }

  waveformContext.stroke();
  waveformFrame = requestAnimationFrame(drawWaveform);
}

function drawSpectrogram() {
  if (!analyser) {
    return;
  }

  const width = spectrogramCanvas.width;
  const height = spectrogramCanvas.height;
  const frequencyData = new Uint8Array(analyser.frequencyBinCount);

  analyser.getByteFrequencyData(frequencyData);
  if (width > 1) {
    const image = spectrogramContext.getImageData(1, 0, width - 1, height);
    spectrogramContext.putImageData(image, 0, 0);
  }

  for (let y = 0; y < height; y += 1) {
    const frequencyIndex = Math.floor((1 - y / height) * (frequencyData.length - 1));
    const value = frequencyData[frequencyIndex];
    spectrogramContext.fillStyle = getSpectrogramColor(value);
    spectrogramContext.fillRect(width - 1, y, 1, 1);
  }

  spectrogramFrame = requestAnimationFrame(drawSpectrogram);
}

function startWaveform(stream) {
  resizeWaveformCanvas();
  resizeSpectrogramCanvas();
  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.72;

  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  drawEmptySpectrogram();
  drawWaveform();
  drawSpectrogram();
}

function stopWaveform() {
  if (waveformFrame) {
    cancelAnimationFrame(waveformFrame);
    waveformFrame = undefined;
  }

  if (spectrogramFrame) {
    cancelAnimationFrame(spectrogramFrame);
    spectrogramFrame = undefined;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = undefined;
    analyser = undefined;
  }

  drawFlatWaveform();
}

async function transcribeAudio(
  audioBlob,
  fileName = "speech.webm",
  modelSize = modelSelect.value,
) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);
  formData.append("language", languageSelect.value);
  formData.append("model_size", modelSize);
  formData.append("noise_snr_db", getNoiseAugmentationValue());

  const response = await fetch("/transcribe", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Whisper transcription failed.");
  }

  return {
    engine: "whisper",
    ...payload,
    text: payload.text.trim(),
  };
}

async function transcribeVosk(audioBlob, fileName = "speech.webm") {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);
  formData.append("noise_snr_db", getNoiseAugmentationValue());

  const response = await fetch("/transcribe-vosk", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Vosk transcription failed.");
  }

  return {
    ...payload,
    text: payload.text.trim(),
  };
}

async function analyzeAudio(audioBlob, fileName = "speech.webm") {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);
  formData.append("noise_snr_db", getNoiseAugmentationValue());

  const response = await fetch("/analyze-audio", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Audio feature analysis failed.");
  }

  return payload;
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    supportNote.textContent =
      "This browser cannot record microphone audio. Try Google Chrome or Microsoft Edge.";
    return;
  }

  try {
    supportNote.textContent = "";
    interimTranscript.textContent = "Listening with Whisper...";
    syncTranscript();

    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    startWaveform(audioStream);

    const mimeType = getBestMimeType();
    mediaRecorder = new MediaRecorder(
      audioStream,
      mimeType ? { mimeType } : undefined,
    );

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      setStatus("Transcribing");
      interimTranscript.textContent = "Whisper is converting speech to text...";
      syncTranscript();

      const audioBlob = new Blob(audioChunks, {
        type: mediaRecorder.mimeType || "audio/webm",
      });

      audioStream.getTracks().forEach((track) => track.stop());
      stopWaveform();

      try {
        const result = await transcribeAudio(audioBlob);
        const transcript = result.text;
        finalText = `${finalText} ${transcript}`.replace(/\s+/g, " ").trim();
        interimTranscript.textContent = "";
        supportNote.textContent = transcript
          ? `Transcription completed using Whisper in ${result.processing_time_seconds}s.`
          : "Whisper did not detect clear speech in this recording.";

        if (referenceText.value.trim()) {
          renderWer(werResult, referenceText.value, finalText);
        }
      } catch (error) {
        supportNote.textContent = error.message;
      } finally {
        setRecordingState(false);
        setStatus("Idle");
        syncTranscript();
      }
    });

    mediaRecorder.start();
    setRecordingState(true);
    setStatus("Recording", true);
  } catch (error) {
    setRecordingState(false);
    setStatus("Idle");
    stopWaveform();
    supportNote.textContent =
      error.name === "NotAllowedError"
        ? "Microphone permission was blocked. Allow microphone access and try again."
        : error.message;
  }
}

function stopRecording() {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
}

async function runDatasetTest() {
  const files = [...datasetAudio.files];

  if (!files.length) {
    datasetResult.textContent = "Choose at least one audio file first.";
    return;
  }

  datasetButton.disabled = true;
  compareModelsButton.disabled = true;
  compareVoskButton.disabled = true;
  datasetTranscript.textContent = `Processing ${files.length} uploaded file(s) with Whisper...`;
  datasetResult.textContent = "";
  datasetTime.textContent = "";
  noiseResult.textContent = "";

  try {
    for (const [index, file] of files.entries()) {
      datasetTranscript.textContent = `Processing ${file.name} (${index + 1}/${files.length})...`;
      await processDatasetFile(file, modelSelect.value);
    }
  } catch (error) {
    datasetTranscript.textContent = "";
    datasetResult.textContent = error.message;
  } finally {
    datasetButton.disabled = false;
    compareModelsButton.disabled = false;
    compareVoskButton.disabled = false;
  }
}

async function runModelComparison() {
  const files = [...datasetAudio.files];

  if (!files.length) {
    datasetResult.textContent = "Choose at least one audio file first.";
    return;
  }

  const models = ["tiny", "base", "small"];
  datasetButton.disabled = true;
  compareModelsButton.disabled = true;
  compareVoskButton.disabled = true;
  datasetResult.textContent = "";
  datasetTime.textContent = "";
  noiseResult.textContent = "";

  try {
    for (const [fileIndex, file] of files.entries()) {
      for (const model of models) {
        datasetTranscript.textContent = `Comparing ${model} on ${file.name} (${fileIndex + 1}/${files.length})...`;
        await processDatasetFile(file, model);
      }
    }
  } catch (error) {
    datasetTranscript.textContent = "";
    datasetResult.textContent = error.message;
  } finally {
    datasetButton.disabled = false;
    compareModelsButton.disabled = false;
    compareVoskButton.disabled = false;
  }
}

async function runVoskComparison() {
  const files = [...datasetAudio.files];

  if (!files.length) {
    datasetResult.textContent = "Choose at least one audio file first.";
    return;
  }

  datasetButton.disabled = true;
  compareModelsButton.disabled = true;
  compareVoskButton.disabled = true;
  datasetResult.textContent = "";
  datasetTime.textContent = "";
  noiseResult.textContent = "";

  try {
    for (const [index, file] of files.entries()) {
      datasetTranscript.textContent = `Comparing Whisper and Vosk on ${file.name} (${index + 1}/${files.length})...`;
      await processDatasetFile(file, modelSelect.value, "whisper");
      await processDatasetFile(file, "vosk", "vosk");
    }
  } catch (error) {
    datasetTranscript.textContent = "";
    datasetResult.textContent = error.message;
  } finally {
    datasetButton.disabled = false;
    compareModelsButton.disabled = false;
    compareVoskButton.disabled = false;
  }
}

async function processDatasetFile(file, modelSize, engine = "whisper") {
  const transcriptionPromise =
    engine === "vosk"
      ? transcribeVosk(file, file.name)
      : transcribeAudio(file, file.name, modelSize);
  const [transcription, analysis] = await Promise.all([
    transcriptionPromise,
    analyzeAudio(file, file.name),
  ]);
  const transcript = transcription.text;
  const wer = renderWer(datasetResult, datasetReference.value, transcript);
  const errors = renderErrorAnalysis(datasetReference.value, transcript);

  datasetTranscript.textContent = transcript || "No clear speech detected.";
  datasetTime.textContent = `Processing time: ${transcription.processing_time_seconds}s using ${transcription.engine || "whisper"} ${transcription.model}`;
  noiseResult.textContent = `Noise test: ${environmentSelect.value}; augmentation ${getNoiseAugmentationLabel()}; estimated SNR ${analysis.estimated_snr_db} dB; RMS ${analysis.rms}; duration ${analysis.duration_seconds}s`;

  drawMatrix(
    uploadedSpectrogramCanvas,
    uploadedSpectrogramContext,
    analysis.spectrogram,
    "Upload audio to generate spectrogram",
  );
  drawMatrix(mfccCanvas, mfccContext, analysis.mfcc, "Upload audio to generate MFCC map");
  addResultRow(file, transcription, analysis, wer, errors);
}

function addResultRow(file, transcription, analysis, wer, errors) {
  if (resultsTableBody.querySelector("td[colspan]")) {
    resultsTableBody.innerHTML = "";
  }

  const row = document.createElement("tr");
  const werText = wer ? `${(wer.wer * 100).toFixed(2)}%` : "N/A";
  const resultRecord = {
    timestamp: new Date().toISOString(),
    file_name: file.name,
    speaker_id: speakerId.value.trim() || "unknown",
    environment: environmentSelect.value,
    microphone: microphoneType.value.trim() || "not specified",
    noise_type: noiseLabel.value.trim() || "not specified",
    language: transcription.language || languageSelect.value || "auto",
    model: transcription.model,
    engine: transcription.engine || "whisper",
    reference_text: datasetReference.value.trim(),
    recognized_text: transcription.text,
    wer_percent: wer ? (wer.wer * 100).toFixed(2) : "N/A",
    word_edits: wer ? wer.edits : "N/A",
    reference_words: wer ? wer.wordCount : "N/A",
    processing_time_seconds: transcription.processing_time_seconds,
    duration_seconds: analysis.duration_seconds,
    sample_rate: analysis.sample_rate,
    rms: analysis.rms,
    noise_floor: analysis.noise_floor,
    estimated_snr_db: analysis.estimated_snr_db,
    noise_augmentation_snr_db: transcription.noise_augmentation_snr_db ?? "none",
    substitutions: errors.substitutions.join(" | "),
    deletions: errors.deletions.join(" | "),
    insertions: errors.insertions.join(" | "),
  };
  const cells = [
    resultRecord.speaker_id,
    resultRecord.environment,
    resultRecord.noise_type,
    resultRecord.model,
    resultRecord.engine,
    werText,
    `${resultRecord.processing_time_seconds}s`,
    `${resultRecord.estimated_snr_db} dB`,
    resultRecord.file_name,
    resultRecord.noise_augmentation_snr_db,
  ];

  for (const cell of cells) {
    const element = document.createElement("td");
    element.textContent = cell;
    row.appendChild(element);
  }

  row.title = microphoneType.value.trim()
    ? `Microphone: ${microphoneType.value.trim()}`
    : "Microphone not specified";
  datasetResults.unshift(resultRecord);
  exportCsvButton.disabled = false;
  resultsTableBody.prepend(row);
  updateCharts();
}

startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

clearButton.addEventListener("click", () => {
  finalText = "";
  finalTranscript.textContent = "";
  interimTranscript.textContent = "";
  placeholder.hidden = false;
  supportNote.textContent = "";
  werResult.textContent = "";
});

copyButton.addEventListener("click", async () => {
  const textToCopy = `${finalText} ${interimTranscript.textContent}`.trim();

  if (!textToCopy) {
    supportNote.textContent = "There is no recognized text to copy yet.";
    return;
  }

  await navigator.clipboard.writeText(textToCopy);
  supportNote.textContent = "Recognized text copied to clipboard.";
});

evaluateButton.addEventListener("click", () => {
  renderWer(werResult, referenceText.value, finalText);
});

datasetButton.addEventListener("click", runDatasetTest);
compareModelsButton.addEventListener("click", runModelComparison);
compareVoskButton.addEventListener("click", runVoskComparison);
exportCsvButton.addEventListener("click", downloadCsv);
window.addEventListener("resize", drawFlatWaveform);
window.addEventListener("resize", drawEmptySpectrogram);
window.addEventListener("resize", () => {
  drawFeaturePlaceholder(
    uploadedSpectrogramCanvas,
    uploadedSpectrogramContext,
    "Upload audio to generate spectrogram",
  );
  drawFeaturePlaceholder(mfccCanvas, mfccContext, "Upload audio to generate MFCC map");
  updateCharts();
});

drawFlatWaveform();
drawEmptySpectrogram();
drawFeaturePlaceholder(
  uploadedSpectrogramCanvas,
  uploadedSpectrogramContext,
  "Upload audio to generate spectrogram",
);
drawFeaturePlaceholder(mfccCanvas, mfccContext, "Upload audio to generate MFCC map");
updateCharts();
