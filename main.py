"""
Phantom Touch - Phase 2 (30% Milestone) [High-Precision Air-Tap Engine]

Why standard Z-threshold failed:
- Monocular RGB webcams do not have hardware LiDAR/ToF sensors, so raw Z is noisy.
- Pure screen Y/X motion caused accidental triggers.

The Proven Computer Vision Solution for In-Air Typing:
1. Fingertip Trajectory Peak Detection (In-Air Tap Pulse):
   - When you tap a key in mid-air, your index fingertip performs a micro-dip:
     it moves down toward the key, stops/hits the bottom, and rebounds upward.
   - We track the inflection point (downward velocity turning to upward rebound).
   - This inflection is mathematically impossible during normal hand hovering or waving.
2. Direct Visual Hover:
   - When hovering, the key turns Cyan.
   - The instant you complete a tap strike on that key, it snaps Green, plays a click, and types 1 letter.
   - It will NEVER auto-type just by resting or lingering on a key.
"""

import time
import math
import collections
import threading
import winsound
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------- Configuration & Layout -----------------
MODEL_PATH = 'hand_landmarker.task'

# Ergonomic QWERTY layout with generous hitboxes
KEYS = [
    # Row 1: Numbers 1-0 (Y: 0.38 - 0.48)
    {"label": "1", "rect": (0.04, 0.38, 0.12, 0.48)},
    {"label": "2", "rect": (0.13, 0.38, 0.21, 0.48)},
    {"label": "3", "rect": (0.22, 0.38, 0.30, 0.48)},
    {"label": "4", "rect": (0.31, 0.38, 0.39, 0.48)},
    {"label": "5", "rect": (0.40, 0.38, 0.48, 0.48)},
    {"label": "6", "rect": (0.49, 0.38, 0.57, 0.48)},
    {"label": "7", "rect": (0.58, 0.38, 0.66, 0.48)},
    {"label": "8", "rect": (0.67, 0.38, 0.75, 0.48)},
    {"label": "9", "rect": (0.76, 0.38, 0.84, 0.48)},
    {"label": "0", "rect": (0.85, 0.38, 0.93, 0.48)},

    # Row 2: Q - P (Y: 0.50 - 0.60)
    {"label": "Q", "rect": (0.04, 0.50, 0.12, 0.60)},
    {"label": "W", "rect": (0.13, 0.50, 0.21, 0.60)},
    {"label": "E", "rect": (0.22, 0.50, 0.30, 0.60)},
    {"label": "R", "rect": (0.31, 0.50, 0.39, 0.60)},
    {"label": "T", "rect": (0.40, 0.50, 0.48, 0.60)},
    {"label": "Y", "rect": (0.49, 0.50, 0.57, 0.60)},
    {"label": "U", "rect": (0.58, 0.50, 0.66, 0.60)},
    {"label": "I", "rect": (0.67, 0.50, 0.75, 0.60)},
    {"label": "O", "rect": (0.76, 0.50, 0.84, 0.60)},
    {"label": "P", "rect": (0.85, 0.50, 0.93, 0.60)},

    # Row 3: A - L (Y: 0.62 - 0.72)
    {"label": "A", "rect": (0.08, 0.62, 0.16, 0.72)},
    {"label": "S", "rect": (0.17, 0.62, 0.25, 0.72)},
    {"label": "D", "rect": (0.26, 0.62, 0.34, 0.72)},
    {"label": "F", "rect": (0.35, 0.62, 0.43, 0.72)},
    {"label": "G", "rect": (0.44, 0.62, 0.52, 0.72)},
    {"label": "H", "rect": (0.53, 0.62, 0.61, 0.72)},
    {"label": "J", "rect": (0.62, 0.62, 0.70, 0.72)},
    {"label": "K", "rect": (0.71, 0.62, 0.79, 0.72)},
    {"label": "L", "rect": (0.80, 0.62, 0.88, 0.72)},

    # Row 4: Z - M + DEL (Y: 0.74 - 0.84)
    {"label": "Z", "rect": (0.12, 0.74, 0.20, 0.84)},
    {"label": "X", "rect": (0.21, 0.74, 0.29, 0.84)},
    {"label": "C", "rect": (0.30, 0.74, 0.38, 0.84)},
    {"label": "V", "rect": (0.40, 0.74, 0.48, 0.84)},
    {"label": "B", "rect": (0.49, 0.74, 0.57, 0.84)},
    {"label": "N", "rect": (0.58, 0.74, 0.66, 0.84)},
    {"label": "M", "rect": (0.67, 0.74, 0.75, 0.84)},
    {"label": "DEL", "rect": (0.77, 0.74, 0.88, 0.84)},

    # Row 5: Actions & Space (Y: 0.86 - 0.95)
    {"label": "CLR", "rect": (0.10, 0.86, 0.22, 0.95)},
    {"label": "SPACE", "rect": (0.25, 0.86, 0.75, 0.95)},
    {"label": "ENTER", "rect": (0.78, 0.86, 0.90, 0.95)},
]

# MediaPipe Skeleton Connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]


def play_click_sound():
    """Non-blocking mechanical click audio."""
    def _beep():
        try:
            winsound.Beep(1200, 25)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


class InflectionTapEngine:
    """
    Detects true physical keystrokes through peak inflection:
    Fingertip goes DOWN (tap) -> reaches bottom -> moves UP (rebound).
    """
    def __init__(self):
        self.y_history = collections.deque(maxlen=6)
        self.t_history = collections.deque(maxlen=6)

        self.last_tap_time = 0
        self.cooldown = 0.22  # Minimum seconds between keystrokes

        self.active_key = None
        self.is_key_pressed = False
        self.visual_sink = 0
        self.buffer_text = ""

    def process_tap(self, tip, pip, mcp, hovered_key, now):
        """
        Tracks relative position of index tip (8) relative to index knuckle (6).
        """
        # Relative finger curl/dip: tip.y - pip.y
        rel_pos = tip.y - pip.y
        self.y_history.append(rel_pos)
        self.t_history.append(now)

        if len(self.y_history) < 4:
            return False

        # Calculate recent velocity
        dt1 = self.t_history[-1] - self.t_history[-2]
        dt2 = self.t_history[-2] - self.t_history[-4]

        if dt1 <= 0 or dt2 <= 0:
            return False

        v_curr = (self.y_history[-1] - self.y_history[-2]) / dt1  # Current velocity
        v_prev = (self.y_history[-2] - self.y_history[-4]) / dt2  # Velocity right before

        # ----------------- Inflection Peak Detection -----------------
        # A tap happens when:
        # 1. v_prev was downward (positive velocity > 0.12)
        # 2. v_curr is decelerating or rebounding upward (v_curr <= 0.02)
        # 3. Finger is hovered over a key
        # 4. Cooldown passed
        is_stroke = False
        time_since_last = now - self.last_tap_time

        if (not self.is_key_pressed) and (time_since_last > self.cooldown) and (hovered_key is not None):
            # Downward tap strike followed by bottoming out
            if (v_prev > 0.14) and (v_curr < 0.04):
                self.is_key_pressed = True
                self.active_key = hovered_key
                self.last_tap_time = now
                self.visual_sink = 6
                is_stroke = True
                play_click_sound()

        # Release detection: when finger has finished rebounding upward
        elif self.is_key_pressed:
            if (now - self.last_tap_time > 0.18) and (v_curr <= 0.05):
                self.is_key_pressed = False
                self.visual_sink = 0

        if not self.is_key_pressed and self.visual_sink > 0:
            self.visual_sink = max(0, self.visual_sink - 1)

        return is_stroke


def get_hovered_key(x, y):
    """Find virtual key under coordinates (x, y)."""
    for key in KEYS:
        x1, y1, x2, y2 = key["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None


def draw_keyboard(frame, hovered_key, active_key, is_pressed, sink_px):
    """Draws 3D keyboard with visual key depression."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Outer chassis
    cv2.rectangle(overlay, (int(0.02 * w), int(0.35 * h)), (int(0.98 * w), int(0.97 * h)), (20, 20, 20), -1)
    cv2.rectangle(frame, (int(0.02 * w), int(0.35 * h)), (int(0.98 * w), int(0.97 * h)), (0, 190, 255), 2)

    for key in KEYS:
        x1 = int(key["rect"][0] * w)
        y1 = int(key["rect"][1] * h)
        x2 = int(key["rect"][2] * w)
        y2 = int(key["rect"][3] * h)

        is_this_hovered = (hovered_key is not None and hovered_key["label"] == key["label"])
        is_this_pressed = (active_key is not None and active_key["label"] == key["label"] and is_pressed)

        sink = sink_px if is_this_pressed else 0
        ky1 = y1 + sink
        ky2 = y2 + sink

        if is_this_pressed:
            cap_color = (0, 255, 100)    # Tactile Green
            border_color = (255, 255, 255)
        elif is_this_hovered:
            cap_color = (240, 160, 20)   # Cyan / Blue Hover
            border_color = (0, 240, 255)
        else:
            cap_color = (48, 48, 48)     # Neutral Dark Gray
            border_color = (95, 95, 95)

        # 3D Drop Shadow
        if not is_this_pressed:
            cv2.rectangle(overlay, (x1 + 2, ky2), (x2, ky2 + 4), (25, 25, 25), -1)

        # Keycap Surface
        cv2.rectangle(overlay, (x1, ky1), (x2, ky2), cap_color, -1)
        cv2.rectangle(frame, (x1, ky1), (x2, ky2), border_color, 2 if (is_this_hovered or is_this_pressed) else 1)

        # Label Text
        lbl = key["label"]
        scale = 0.44 if len(lbl) > 2 else (0.52 if len(lbl) > 1 else 0.68)
        t_sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        tx = x1 + (x2 - x1 - t_sz[0]) // 2
        ty = ky1 + (ky2 - ky1 + t_sz[1]) // 2
        txt_col = (10, 10, 10) if (is_this_hovered or is_this_pressed) else (255, 255, 255)
        cv2.putText(frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, txt_col, 2)

    # Blend transparency
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)


def main():
    print("[Phantom Touch] Running High-Precision In-Air Tap Engine...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    engine = InflectionTapEngine()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = detector.detect(mp_image)
        now = time.time()

        hovered_key = None

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            # Hand Skeleton
            for start, end in CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], (0, 180, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 255, 128), -1)

            # Key Landmarks
            lm_mcp = hand_landmarks[5]
            lm_pip = hand_landmarks[6]
            lm_tip = hand_landmarks[8]

            # Direct physical fingertip position
            tip_x, tip_y = lm_tip.x, lm_tip.y

            # Identify hovered key
            hovered_key = get_hovered_key(tip_x, tip_y)

            # Evaluate Tap
            is_tap = engine.process_tap(lm_tip, lm_pip, lm_mcp, hovered_key, now)

            # Output Character
            if is_tap and hovered_key:
                lbl = hovered_key["label"]
                if lbl == "SPACE":
                    engine.buffer_text += " "
                elif lbl == "DEL":
                    engine.buffer_text = engine.buffer_text[:-1]
                elif lbl == "CLR":
                    engine.buffer_text = ""
                elif lbl == "ENTER":
                    engine.buffer_text += "\n"
                else:
                    engine.buffer_text += lbl

        # Draw Keyboard
        draw_keyboard(frame, hovered_key, engine.active_key, engine.is_key_pressed, engine.visual_sink)

        # ----------------- Top Header & Status -----------------
        cv2.rectangle(frame, (0, 0), (w, 34), (15, 15, 15), -1)
        cv2.putText(frame, "PHANTOM TOUCH | Phase 2 (30%) - Spatial Keyboard",
                    (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 230, 255), 2)

        # Status Box (Top-Right)
        box_x = w - 220
        cv2.rectangle(frame, (box_x, 40), (w - 10, 105), (25, 25, 25), -1)
        cv2.rectangle(frame, (box_x, 40), (w - 10, 105), (65, 65, 65), 1)
        cv2.putText(frame, f"Key   : {hovered_key['label'] if hovered_key else '--'}",
                    (box_x + 8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        switch_txt = "CLICKED!" if engine.is_key_pressed else "HOVERING"
        switch_col = (0, 255, 100) if engine.is_key_pressed else (0, 200, 255)
        cv2.putText(frame, f"State : {switch_txt}",
                    (box_x + 8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, switch_col, 1)
        cv2.putText(frame, "Tap finger downward in air",
                    (box_x + 8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Output Text Buffer (Top-Left)
        buf_w = box_x - 15
        cv2.rectangle(frame, (10, 40), (buf_w, 105), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 40), (buf_w, 105), (0, 200, 255), 2)
        cv2.putText(frame, "OUTPUT TEXT BUFFER:", (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
        display_line = engine.buffer_text.replace('\n', ' ')[-32:] if engine.buffer_text else "(tap finger down on key to type...)"
        text_col = (0, 255, 255) if engine.buffer_text else (120, 120, 120)
        cv2.putText(frame, display_line, (18, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, text_col, 2)

        cv2.imshow('Phantom-Touch: Phase 2 Prototype', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
