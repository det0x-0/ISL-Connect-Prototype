# 🤟 ISL-Connect — Indian Sign Language Interpreter

> **Hackathon Prototype** · Real-time ISL gesture recognition with voice output and a browser-based web interface.

---

## 📽️ Demo

Check out `prototype_video.mp4` in this repo (or the `demo/` folder) for a live demonstration of the desktop detection + voice output in action.

---

## 🚀 What This Project Does

ISL-Connect bridges the communication gap between sign-language users and non-signers by:

| Feature | Description |
|---|---|
| 🎥 **Real-time Sign Detection** | Uses your webcam + MediaPipe to track hand & body landmarks |
| 🧠 **Deep Learning (LSTM)** | A trained LSTM model predicts ISL words/letters from 30-frame sequences |
| 🔊 **Voice Output** | Detected words are spoken aloud using text-to-speech |
| 🌐 **Web Interface** | A Flask-powered browser app with skeleton visualisation and NLP sentence correction |
| 🔤 **Finger-spelling** | Supports individual letter detection (`A`, `S`, `H`, `I`, `Q`) that combine into words |

---

## 📂 Project Structure

```
prototype ISL-connect/
│
├── detectandvoice.py     # ★ Main desktop app — opens webcam, detects signs & speaks
├── app.py             # Flask backend for the web version (with NLP + SSE streaming)
├── index.html         # Frontend HTML for the web app (served by Flask)
├── train.py        # Model training script (LSTM on MediaPipe keypoint data)
│
├── requirements.txt       # Python dependencies
└── README.md              # You are here
```

> **Note:** The trained model file `asl_model_filteredaugment.h5` is **not included** in this repo due to file size. See [Model Weights](#-model-weights) below.

---

## ⚡ Quick Start

### 1 · Prerequisites

- Python 3.9 – 3.11
- A webcam

### 2 · Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 · Get the Model Weights

Download `asl_model_filteredaugment.h5` (or `asl_weights_filteredaugment.npy`) and place it in the **same folder** as the scripts.

> If you trained your own model using `train_claude.py`, rename the output to `asl_model_filteredaugment.h5`.

---

## 🖥️ Running the Desktop App (Camera + Voice)

```bash
python detectandvoice2.py
```

- A window will open showing your webcam feed with skeleton landmarks drawn on it.
- Detected words appear in the orange bar at the top and are **spoken aloud**.
- **Keyboard shortcuts:**
  - `Q` — quit
  - `C` — clear the sentence

**Supported signs:** `hello`, `my`, `name`, `thanks`, and finger-spelling letters `A`, `S`, `H`, `I`, `Q`

---

## 🌐 Running the Web App

```bash
python app2nlp.py
```

Then open your browser and go to: **http://localhost:5000**

### Web App Features

- **Sign → Text tab:** Live camera feed with skeleton overlay, confidence bars, top-3 predictions, and NLP-corrected sentences.
- **Text → Sign tab:** Type a word/sentence and it will play the corresponding ASL sign video.
- **Live threshold slider** — adjust detection sensitivity without restarting.
- **Export transcript** — download your session as a `.txt` file.
- **Translation** — translate the detected sentence to other languages (uses MyMemory API, no key needed).

> The web app requires the `templates/` folder to contain `index3nlp.html`. If you are running from the project root, Flask will find it automatically since it is configured to serve it directly.

---

## 🏋️ Training Your Own Model

1. Collect MediaPipe keypoint data and save as `.npy` files under `MP_Data/<action>/<sequence>/<frame>.npy`.
2. Run:

```bash
python train_claude.py
```

3. The script will:
   - Load and validate all sequences
   - Train a 3-layer LSTM model (64 → 128 → 64 units)
   - Evaluate per-class accuracy
   - Save `asl_model1.h5` + `asl_weights1.npy`
   - Plot training curves to `training_history.png`

**Model architecture:**
```
Input (30 frames × 258 keypoint features)
  → LSTM(64) → BatchNorm → Dropout(0.3)
  → LSTM(128) → BatchNorm → Dropout(0.3)
  → LSTM(64) → BatchNorm → Dropout(0.3)
  → Dense(64) → Dense(32) → Dense(9, softmax)
```

---

## 📦 Model Weights

The pre-trained weights file (`asl_model_filteredaugment.h5`) was trained on custom-collected ISL gesture data for 9 classes. It is **not committed** to this repository due to GitHub's 100 MB file size limit.

**Options to get the weights:**
- Train your own using `train_claude.py` with your own data.
- Contact the project author for access to the pre-trained file.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Gesture Tracking | [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Computer Vision | OpenCV |
| Text-to-Speech | pyttsx3 |
| Web Backend | Flask + Flask-CORS |
| NLP Correction | HuggingFace Transformers (`flan-t5-small`) — optional |
| Translation | MyMemory API (free, no key) |
| Frontend | Vanilla HTML / CSS / JS |

---

## 📋 Known Limitations (Prototype)

- Trained on a small custom dataset — recognition works best under good lighting with clear hand visibility.
- The NLP correction feature requires `transformers` and `sentencepiece` to be installed separately; it gracefully falls back if not available.
- The web app currently requires local camera access via the Flask backend (not browser MediaPipe).

---

## 👤 Author

Built for a hackathon as an MVP prototype for real-time ISL communication.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
