# Phantom Touch: The Software-Defined Spatial Interface

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

**Phantom Touch** transitions Human-Computer Interaction (HCI) from rigid physical peripherals to a completely software-defined spatial interface using standard webcams and edge AI.

---

## 👥 Authors & Team
- **Shree Manvi** (25BCE5567)
- **Devayani Manoj** (25BCE5595)
- **Ananya Harithas** (25BCE5615)

---

## 🚀 Features & Architecture
- **Zero External Hardware:** Runs purely on built-in laptop / desktop webcams.
- **Edge Vision AI:** Uses Google MediaPipe to track 21 3D hand landmarks in real-time at 30 FPS.
- **Spatial Geometry & Layout:** Projects a full 5-row QWERTY keyboard (`1-0`, `Q-P`, `A-L`, `Z-M`, `CLR`, `SPACE`, `ENTER`) in mid-air and desk space.
- **Dynamic In-Air Tap Engine:** Kinematic trajectory inflection detection to distinguish between hovering and deliberate keystroke taps.
- **3D Mechanical Feedback:** Real-time visual key depression (6px key sink) with non-blocking audio clicks.
- **Zero-Install Cloud Independence:** Runs 100% locally on your machine with near-zero latency.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Phantom-Touch.git
cd Phantom-Touch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe Hand Landmarker Model
If not already downloaded, fetch the task model:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

### 4. Run the Application
```bash
python main.py
```
*(Press `q` on the OpenCV window to exit).*

---

## 📋 Project Roadmap & Milestones
- [x] **Phase 1 (20%):** Core Edge-Vision Pipeline — Camera capture & 21-point 3D hand landmark tracking.
- [x] **Phase 2 (30%):** Spatial Mapping & Tap Detection — Full 5-row virtual keyboard projection & in-air tap engine.
- [ ] **Phase 3 (50%):** Core Integration — OS-level keystroke injection using `pynput` / `pyautogui`.
- [ ] **Phase 4 (80%):** Dynamic Layout & Accessibility — Dynamic keyboard resizing for motor-impaired users and camera jitter suppression.
- [ ] **Phase 5 (100%):** Final Polish & Ambient Lighting Validation.
