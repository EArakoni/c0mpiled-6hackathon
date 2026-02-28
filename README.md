# ✍️ Sketched

**Turn your finger and hands into a keyboard — no touching required.**

Sketched is a real-time, camera-based text input tool that lets you write by drawing letters in the air with your finger *or* signing ASL finger spelling in front of your webcam. Everything you input builds into a live paragraph saved automatically to a text file.

---

## 🎯 What It Does

| Mode | How it works |
|------|-------------|
| ✏️ **Draw Mode** | Trace letters/words in the air with your index finger. Apple's on-device Vision framework reads your handwriting and appends it to your document. |
| 🤟 **ASL Mode** | Hold an ASL finger-spelling sign in front of the camera. A trained RandomForest classifier reads your hand landmarks in real time and adds the letter when you hold it steady. |

Both modes output to a plain `sketched_output.txt` file you can open in any text editor — it updates live as you write.

---

## 🚀 Quick Start

### 1. Requirements
- **macOS** (Apple Vision framework required for handwriting recognition)
- **Python 3.12** (MediaPipe is not compatible with 3.13+)
- A webcam

### 2. Setup

```bash
# Create a virtual environment with Python 3.12
python3.12 -m venv ~/sketched-env
source ~/sketched-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```bash
python sketched.py
```

---

## 🖐️ Controls

### Gestures
| Gesture | Action |
|---------|--------|
| ☝️ Index finger up | **Draw** a letter in the air |
| ✌️ Two fingers up | **Erase** your current stroke |
| ✊ Fist / lower hand | **Confirm** — sends stroke to Apple Vision for recognition |
| 🤟 Hold ASL sign | Letter recognized after ~1 second hold |

### Keyboard
| Key | Action |
|-----|--------|
| `S` | Switch between Draw and ASL mode |
| `SPACE` | Insert a space |
| `ENTER` | Start a new paragraph |
| `Z` | Undo last word/letter |
| `C` | Clear everything |
| `Q` | Quit |

---

## 🧠 How It Works

### Draw Mode — Apple Vision OCR
1. MediaPipe tracks your hand at 30fps via webcam
2. When your index finger is raised, your fingertip position is recorded as a stroke
3. When you lower your hand (fist gesture), the stroke is rendered as black ink on a white background
4. The image is passed to **Apple's `VNRecognizeTextRequest`** — the same on-device engine powering Live Text, Scribble, and the iOS Camera app
5. The recognized text is appended to your paragraph and saved

### ASL Mode — Landmark Classifier
1. MediaPipe extracts **21 3D hand landmarks** every frame
2. An **84-dimensional feature vector** is computed: normalized joint coordinates, finger curl angles, fingertip distances, inter-finger spread
3. A **RandomForest classifier** (trained on synthetic augmented ASL hand poses) predicts the letter
4. A rolling majority vote over 30 frames filters noise
5. When the same letter holds for ~1 second with ≥70% vote confidence, it's added to the text

---

## 📁 Project Structure

```
sketched/
├── sketched.py          # Main application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── sketched_output.txt  # Auto-generated output (created on first run)
```

---

## ✅ ASL Letters Supported

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```
> J and Z require motion gestures (not static poses) — coming in a future update.

---

## 💡 Tips for Best Results

**Draw Mode:**
- Write **large** — fill most of the camera view
- **Print clearly**, don't use cursive
- One letter or word per stroke works best
- Draw slowly and deliberately for better recognition

**ASL Mode:**
- Hold your hand **centered in the frame**
- Keep fingers **clearly separated** for letters like V, W, U
- Hold the sign **steady** — the progress ring fills as you hold
- Good lighting makes a big difference

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand tracking | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) |
| Handwriting OCR | [Apple Vision Framework](https://developer.apple.com/documentation/vision) (`VNRecognizeTextRequest`) |
| ASL classification | [scikit-learn](https://scikit-learn.org/) RandomForest |
| Camera & rendering | [OpenCV](https://opencv.org/) |
| Feature extraction | Custom 84-dim landmark geometry vectors |

---

## 🔮 Future Ideas

- [ ] Motion-based letters (J, Z) using stroke tracking
- [ ] Word-level ASL gesture recognition
- [ ] Export to .docx / .pdf
- [ ] Voice readback of recognized text
- [ ] Windows/Linux support via alternative OCR (Tesseract)
- [ ] Custom vocabulary training for better accuracy

---

## 👨‍💻 Built At

Built during a hackathon to explore accessible, touch-free text input using computer vision and on-device machine learning.

---

## 📄 License

MIT License — free to use, modify, and build on.
