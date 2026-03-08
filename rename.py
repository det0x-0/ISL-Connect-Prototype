"""
ASLBridge — app.py  (Laptop Camera Edition)
Fixes: stability logic, cooldown spam prevention, live threshold.
"""

import os, cv2, json, time, threading, queue, urllib.request, urllib.parse
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, Response, render_template, jsonify, request, stream_with_context
from flask_cors import CORS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

app = Flask(__name__, template_folder=".")
app.secret_key = os.environ.get("SECRET_KEY", "aslbridge-2024")
CORS(app)

# ---------------------------------------------------------------------------
# Constants  — tuned for real use
# ---------------------------------------------------------------------------
actions          = np.array(['hello', 'my', 'name', 'thanks', 'A', 'S', 'H', 'I', 'Q'])
SPELLING_LETTERS = {'A', 'S', 'H', 'I', 'Q'}
SEQUENCE_LENGTH  = 30
STABILITY_FRAMES = 6       # ← was 10; 6 consecutive matching frames is enough
COOLDOWN_SECONDS = 2.5     # ← was 1.5; prevent same word repeating for 2.5s
CAMERA_INDEX     = 0
JPEG_QUALITY     = 70

# Live-adjustable threshold (POST /set_threshold to change without restart)
_threshold_lock = threading.Lock()
_threshold      = 0.70     # default — UI slider can change this

def get_threshold():
    with _threshold_lock: return _threshold

def set_threshold(v):
    global _threshold
    with _threshold_lock:
        _threshold = max(0.10, min(0.99, float(v)))
    print(f"[THRESH] Updated to {_threshold:.2f}")

# ---------------------------------------------------------------------------
# Optional NLP
# ---------------------------------------------------------------------------
corrector = None
try:
    from transformers import pipeline as hf_pipeline
    print("[NLP] Loading flan-t5-small …")
    corrector = hf_pipeline("text2text-generation", model="google/flan-t5-small")
    print("[NLP] ✅ Ready")
except Exception as e:
    print(f"[NLP] Skipped ({e}). pip install transformers sentencepiece")

def fix_sentence(raw_words):
    if not raw_words: return ""
    if not corrector:  return " ".join(raw_words)
    try:
        prompt = f"Fix this ASL gloss into natural English sentence: {' '.join(raw_words)}"
        return corrector(prompt, max_new_tokens=60)[0]["generated_text"].strip()
    except Exception:
        return " ".join(raw_words)

_nlp_queue = queue.Queue(maxsize=10)

def _nlp_worker():
    while True:
        try:
            words_snapshot, eng = _nlp_queue.get(timeout=60)
        except queue.Empty:
            continue
        corrected = fix_sentence(words_snapshot)
        payload = {"nlp_update": True, "corrected_sentence": corrected,
                   "raw": " ".join(words_snapshot)}
        eng._broadcast(payload)
        with eng._lock:
            eng.latest_pred["corrected_sentence"] = corrected
            if eng.history:
                eng.history[-1]["corrected"] = corrected

threading.Thread(target=_nlp_worker, daemon=True).start()

def trigger_nlp(words, engine):
    if not words: return
    try: _nlp_queue.put_nowait((list(words), engine))
    except queue.Full: pass

# ---------------------------------------------------------------------------
# Translation (MyMemory — no API key)
# ---------------------------------------------------------------------------
def translate_text(text, target_lang):
    if not text or target_lang == "en": return text
    try:
        params = urllib.parse.urlencode({"q": text, "langpair": f"en|{target_lang}"})
        req = urllib.request.Request(
            f"https://api.mymemory.translated.net/get?{params}",
            headers={"User-Agent": "ASLBridge/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())["responseData"]["translatedText"]
    except Exception as e:
        print(f"[TRANSLATE] {e}"); return text

# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 258)),
    BatchNormalization(), Dropout(0.3),
    LSTM(128, return_sequences=True, activation='tanh'),
    BatchNormalization(), Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])
try:
    model.load_weights('asl_model_filteredaugment.h5')
    print("✅ Weights loaded from .h5")
except Exception:
    try:
        model.set_weights(list(np.load('asl_weights_filteredaugment.npy', allow_pickle=True)))
        print("✅ Weights loaded from .npy")
    except Exception as e:
        print(f"❌ Could not load weights: {e}")

# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    pose = (np.array([[r.x,r.y,r.z,r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(132))
    lh   = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(63))
    rh   = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(63))
    return np.concatenate([pose, lh, rh])

def draw_skeleton_only(results, shape=(480, 640)):
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    cyan, dim, blue = (0,229,200), (0,130,110), (80,160,255)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(canvas, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=cyan, thickness=3, circle_radius=5),
            mp_drawing.DrawingSpec(color=dim,  thickness=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(canvas, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=cyan, thickness=3, circle_radius=5),
            mp_drawing.DrawingSpec(color=dim,  thickness=2))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(canvas, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=blue, thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(40,80,160), thickness=1))
    return canvas

# ---------------------------------------------------------------------------
# CameraEngine
# ---------------------------------------------------------------------------
class CameraEngine:
    def __init__(self):
        self.latest_video_jpeg    = None
        self.latest_skeleton_jpeg = None
        self.latest_pred  = {}
        self.running      = False
        self._lock        = threading.Lock()
        self._thread      = None

        self.sequence     = deque(maxlen=SEQUENCE_LENGTH)
        self.predictions  = deque(maxlen=STABILITY_FRAMES)
        self.last_word    = None
        self.last_word_t  = 0.0   # timestamp of last confirmed word
        self.current_word = ""
        self.sentence     = []
        self.history      = []

        self._sse_queues  = []
        self._sse_lock    = threading.Lock()

    def add_subscriber(self):
        q = queue.Queue(maxsize=60)
        with self._sse_lock: self._sse_queues.append(q)
        return q

    def remove_subscriber(self, q):
        with self._sse_lock:
            try: self._sse_queues.remove(q)
            except ValueError: pass

    def _broadcast(self, payload):
        msg = "data: " + json.dumps(payload) + "\n\n"
        with self._sse_lock:
            dead = []
            for q in self._sse_queues:
                try: q.put_nowait(msg)
                except queue.Full: dead.append(q)
            for q in dead: self._sse_queues.remove(q)

    def start(self):
        if self.running: return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self): self.running = False

    def reset_inference(self):
        self.sequence.clear(); self.predictions.clear()
        self.last_word = None; self.last_word_t = 0.0
        self.current_word = ""; self.sentence = []

    def flush_spelling(self):
        if self.current_word:
            word = self.current_word
            self.sentence.append(word)
            self.current_word = ""
            trigger_nlp(list(self.sentence), self)
            return word
        elif self.sentence:
            trigger_nlp(list(self.sentence), self)
        return None

    def inject_word(self, word):
        word = word.strip().lower()
        if not word: return
        self.sentence.append(word)
        entry = {"word": word, "time": time.strftime("%H:%M:%S"),
                 "raw": " ".join(self.sentence), "corrected": ""}
        self.history.append(entry)
        if len(self.history) > 50: self.history.pop(0)
        trigger_nlp(list(self.sentence), self)
        self._broadcast({
            "confirmed_word": word, "sentence": list(self.sentence),
            "current_word": self.current_word, "live_label": word,
            "confidence": 1.0, "top3": [], "history": self.history[-10:],
            "corrected_sentence": self.latest_pred.get("corrected_sentence",""),
            "threshold": get_threshold(),
        })

    def _loop(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"[CAM] ERROR: Cannot open camera {CAMERA_INDEX}")
            self.running = False; return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("[CAM] Capture loop running")

        while self.running:
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_draw_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())

            skeleton = draw_skeleton_only(results, (frame.shape[0], frame.shape[1]))

            kp = extract_keypoints(results)
            self.sequence.append(kp)

            confirmed_word = None
            live_label     = None
            confidence     = 0.0
            top3           = []
            thresh         = get_threshold()

            if len(self.sequence) == SEQUENCE_LENGTH:
                inp = np.expand_dims(np.array(self.sequence), axis=0)
                raw = model.predict(inp, verbose=0)[0]
                idx        = int(np.argmax(raw))
                confidence = float(raw[idx])

                # ── FIX 1: label exists if confidence >= threshold ─────────
                live_label = actions[idx] if confidence >= thresh else None

                top3_idx = np.argsort(raw)[-3:][::-1]
                top3 = [{"label": str(actions[i]), "conf": round(float(raw[i]),3)} for i in top3_idx]

                # ── FIX 2: stability buffer tracks the raw label (None counts) ─
                # We only append to predictions when we have a label above threshold
                if live_label is not None:
                    self.predictions.append(live_label)
                else:
                    # Below threshold — reset buffer so it must earn stability fresh
                    self.predictions.clear()

                # ── FIX 3: confirm only when buffer full AND all same ──────
                now = time.time()
                if (len(self.predictions) == STABILITY_FRAMES
                        and all(p == live_label for p in self.predictions)
                        and live_label is not None):

                    # ── FIX 4: cooldown is purely TIME-based, not word-based ─
                    # Same word: must wait full COOLDOWN_SECONDS before repeating
                    # Different word: can confirm immediately
                    time_since_last = now - self.last_word_t
                    same_word = (live_label == self.last_word)

                    if not same_word or time_since_last >= COOLDOWN_SECONDS:
                        self.last_word   = live_label
                        self.last_word_t = now
                        # Clear buffer so the same sign doesn't instantly re-trigger
                        self.predictions.clear()

                        if live_label in SPELLING_LETTERS:
                            self.current_word += live_label
                        else:
                            if self.current_word:
                                self.sentence.append(self.current_word)
                                self.current_word = ""
                            self.sentence.append(live_label)
                            confirmed_word = live_label

                        entry = {"word": live_label, "time": time.strftime("%H:%M:%S"),
                                 "raw": " ".join(self.sentence), "corrected": ""}
                        self.history.append(entry)
                        if len(self.history) > 50: self.history.pop(0)
                        trigger_nlp(list(self.sentence), self)
                        print(f"[PREDICT] ✅ {live_label} ({int(confidence*100)}%)")

            # Overlay
            label_txt = f"{live_label}  {int(confidence*100)}%" if live_label else "..."
            cv2.rectangle(frame, (0,0), (460,42), (0,0,0), -1)
            cv2.putText(frame, label_txt, (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,229,200), 2)
            sentence_txt = " ".join(self.sentence)
            if self.current_word: sentence_txt += f" [{self.current_word}...]"
            cv2.rectangle(frame, (0,frame.shape[0]-42), (frame.shape[1],frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, sentence_txt, (8,frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(skeleton, label_txt, (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,229,200), 2)

            ok,  buf  = cv2.imencode(".jpg", frame,   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ok2, buf2 = cv2.imencode(".jpg", skeleton, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with self._lock:
                if ok:  self.latest_video_jpeg    = buf.tobytes()
                if ok2: self.latest_skeleton_jpeg = buf2.tobytes()

            with self._lock:
                cached_corrected = self.latest_pred.get("corrected_sentence","")
            pred = {
                "live_label":         live_label,
                "confidence":         round(confidence,3),
                "confirmed_word":     confirmed_word,
                "current_word":       self.current_word,
                "sentence":           list(self.sentence),
                "corrected_sentence": cached_corrected,
                "top3":               top3,
                "history":            self.history[-10:],
                "threshold":          thresh,
            }
            with self._lock: self.latest_pred = pred
            self._broadcast(pred)

        holistic.close(); cap.release()
        print("[CAM] Capture loop stopped")


engine = CameraEngine()
engine.start()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index3nlp.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with engine._lock: jpeg = engine.latest_video_jpeg
            if jpeg: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.033)
    return Response(stream_with_context(gen()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/skeleton_feed")
def skeleton_feed():
    def gen():
        while True:
            with engine._lock: jpeg = engine.latest_skeleton_jpeg
            if jpeg: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.033)
    return Response(stream_with_context(gen()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/prediction_stream")
def prediction_stream():
    q = engine.add_subscriber()
    def gen():
        with engine._lock: init = dict(engine.latest_pred)
        yield "data: " + json.dumps(init) + "\n\n"
        try:
            while True:
                try: yield q.get(timeout=15)
                except queue.Empty: yield ": keepalive\n\n"
        finally: engine.remove_subscriber(q)
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no",
                             "Access-Control-Allow-Origin":"*"})

@app.route("/flush_spelling", methods=["POST"])
def flush_spelling():
    flushed = engine.flush_spelling()
    return jsonify({"flushed": flushed, "sentence": list(engine.sentence)})

@app.route("/reset", methods=["POST"])
def reset():
    engine.reset_inference()
    return jsonify({"status": "reset"})

@app.route("/history")
def get_history():
    return jsonify(engine.history)

@app.route("/voice_input", methods=["POST"])
def voice_input():
    data = request.get_json(force=True)
    text = data.get("text","").strip()
    if not text: return jsonify({"error":"empty"}), 400
    for word in text.lower().split(): engine.inject_word(word)
    return jsonify({"injected": text, "sentence": list(engine.sentence)})

@app.route("/set_threshold", methods=["POST"])
def set_thresh():
    data = request.get_json(force=True)
    set_threshold(data.get("value", 0.7))
    return jsonify({"threshold": get_threshold()})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(force=True)
    result = translate_text(data.get("text",""), data.get("lang","ta"))
    return jsonify({"translated": result, "lang": data.get("lang")})

@app.route("/export_txt")
def export_txt():
    lines = ["ASLBridge Session Transcript",
             f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}","="*40,""]
    for e in engine.history:
        lines.append(f"[{e['time']}] {e['word']}")
        if e.get("corrected"): lines.append(f"        → {e['corrected']}")
        elif e.get("raw"):     lines.append(f"        ~ {e['raw']}")
        lines.append("")
    return Response("\n".join(lines), mimetype="text/plain",
                    headers={"Content-Disposition":"attachment; filename=aslbridge_transcript.txt"})

@app.route("/camera_status")
def camera_status():
    return jsonify({"running": engine.running,
                    "has_frame": engine.latest_video_jpeg is not None})

@app.route("/videos")
def list_videos():
    video_dir = os.path.join(app.static_folder, "videos")
    try: files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    except FileNotFoundError: files = []
    return jsonify({os.path.splitext(f)[0].lower(): f for f in files})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)