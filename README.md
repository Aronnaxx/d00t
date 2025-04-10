
# 🦆 Duck VLA: Vision-Language-Action Control System

This project enables a duck droid to autonomously move, observe, and respond to people using a **Vision-Language-Action (VLA)** stack. It runs on a **Radxa Zero 3W** and integrates an Arducam, speaker, and microphone. The droid already supports walking and expressive emotes controlled by an Xbox joystick—this adds autonomous reasoning on top.

---

## 🎯 Goal

To build a modular system that allows the duck to:
- Understand voice commands (e.g., "Come here", "Look at the red ball")
- See its environment using an Arducam
- Reason about what it sees using a lightweight VLM (like Moondream)
- Move, turn, strafe, and emote autonomously using joystick-like APIs
- Respond with expressive beep sounds (no TTS required)

---

## 🧱 Project Structure

Note: we also have a submodule called OpenDuckPlayground that uses MuJoCo to simulate the droid, where this will be tested. This submodule has items such as joystick.py and others

```
duck_vla/
├── camera/
│   ├── arducam_capture.py         # Capture frames from Arducam
│   └── object_detector.py         # Object detection models (e.g. YOLOv8-tiny)
│
├── vision/
│   └── moondream_wrapper.py       # Captioning and visual Q&A using Moondream
│
├── language/
│   ├── stt.py                     # Speech-to-text (Vosk or Whisper.cpp)
│   ├── intent_parser.py           # Maps phrases to structured commands
│
├── action/
│   ├── joystick_interface.py      # Interface to existing joystick-style motor control
│   ├── motion_controller.py       # Maps intents to movement commands
│   ├── emotes.py                  # Triggers emote beeps and sound effects
│
├── brain/
│   └── decision_loop.py           # Main controller: vision + language → action
│
├── utils/
│   └── audio.py                   # Utility for recording and playing beeps
│
├── sounds/                        # Folder for WAV beep/emote sounds
│
├── models/                        # Store downloaded models (e.g. .onnx or .bin)
│
├── run_duck.py                    # Entrypoint script
└── requirements.txt               # Python dependencies
```

---

## 🛠️ Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download models**:
   - Moondream: from Hugging Face (`huggingface/Moondream1`)
   - Optional: YOLOv8-tiny or RT-DETR ONNX model
   - Vosk speech recognition model

3. **Test camera input**:
   ```bash
   python camera/arducam_capture.py
   ```

4. **Run the duck decision loop**:
   ```bash
   python run_duck.py
   ```

---

## 🧠 How It Works

- **Voice Input:** Duck listens using mic and transcribes speech with STT
- **Command Parsing:** Maps "Come here" → `follow_person` intent
- **Visual Input:** Captures frame and captions it using Moondream
- **Spatial Reasoning:** Identifies object location from bounding box or caption
- **Action Layer:** Converts intent + visual context into joystick-style movement
- **Response Output:** Plays matching emote sound effect from `sounds/`

---

## 🗣️ Example Behaviors

| Command                  | Behavior                                  |
|--------------------------|-------------------------------------------|
| “Duckie, come here”      | Detects person, walks to them, beeps happily |
| “What do you see?”       | Captions image and plays curious beep     |
| “Turn around”            | Executes a 180° turn with dramatic beep   |
| “Wave hello!”            | Plays a friendly wave beep emote          |
| “Look at the red ball”   | Turns head toward object, plays beep      |

---

## 📦 Example `requirements.txt`

```txt
torch
transformers
opencv-python
numpy
sounddevice
vosk
aiofiles
```

---

## ✅ Roadmap

- [ ] Integrate Moondream for on-device image captioning
- [ ] Use Vosk for offline STT
- [ ] Build rule-based intent parser for basic commands
- [ ] Wrap movement/emote functions from joystick control
- [ ] Assign beep files to different moods and intents
- [ ] Add fallback sound for unknown commands or states

---

## 🤖 Requirements

- Python 3.9+
- Radxa Zero 3W (8GB RAM recommended)
- Arducam camera
- Microphone + speaker
- Emote sound effects (WAV files)

---

## 🧩 Credits & Inspiration

Built for the Duck Droid project using:
- Hugging Face [Moondream](https://huggingface.co/spaces/huggingface/Moondream)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)

Let’s make Duckie smart, autonomous, and adorably beep-filled. 🦆✨
