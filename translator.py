# translator.py — SMOOTH, STABLE, PROFESSIONAL (FINAL PATCH)
# Key improvements in this version:
# 1. Stability parameters maintained.
# 2. Camera Fix (Windows): Uses cv2.CAP_DSHOW.
# 3. Display Fix: Explicitly names the window.
# 4. **CRITICAL FIX:** Added a fallback FPS value to prevent ZeroDivisionError.

import cv2
import torch
import mediapipe as mp
import time
from torchvision import transforms, models
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model with error handling
try:
    model = models.resnet18(pretrained=False)
    # Assuming the original 29 output classes are correct (A-Z, del, nothing, space)
    model.fc = torch.nn.Linear(512, 29) 
    # NOTE: Ensure 'raw_model.pth' is in the same directory as this script.
    model.load_state_dict(torch.load("raw_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: raw_model.pth not found. Please ensure the model file is in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Letter indices (0-25 for A-Z)
LETTER_INDICES = list(range(26))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# --- CAMERA AND WINDOW FIXES ---
WINDOW_NAME = "ASL Translator V3 — PATCHED"

# Use cv2.CAP_DSHOW for better Windows compatibility. Try index 1 or 2 if 0 fails.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
if not cap.isOpened():
    print("Error: Could not open camera. Try changing the index (0) to 1 or 2.")
    exit(1)

# Explicitly create the window to ensure it initializes before the loop
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE) 

# --- STABILITY SETTINGS (PATCHED VALUES) ---
STABLE_FRAMES = 10       # Frames to stabilize on a prediction
CONF_THRESHOLD = 0.85    # Confidence threshold for a prediction
AUTO_SPACE_PAUSE = 2.0   # Time (s) to wait for auto-space
DEFAULT_FPS = 30.0       # Fallback FPS value for safe calculation
# ---------------------------------------------

current_prediction = "?"
stable_count = 0
prev_stable_count = 0   # Track to detect "just stabilized"
buffer = ""             # Current word being spelled
sentence = ""           # Full sentence
last_time = time.time() # For auto-space on pause (hand removal)

# --- ZERO DIVISION ERROR FIX ---
# Get camera FPS, or use a default if it's 0 or not available
camera_fps = cap.get(cv2.CAP_PROP_FPS)

# Use the default if the retrieved FPS is less than a small epsilon (e.g., 1.0)
if camera_fps < 1.0: 
    camera_fps = DEFAULT_FPS

# Calculate the required hold time safely
hold_time = STABLE_FRAMES / camera_fps

print("Starting ASL Translator V3 (FINAL FIX). Press 'q' to quit.")
# Use the calculated hold_time in the print statement
print(f"Tip: Hold each letter gesture for ~{hold_time:.2f}s. Lower hand for {AUTO_SPACE_PAUSE}s to space.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural mirror view
    frame = cv2.flip(frame, 1) 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pred = "?"
    confidence = 0.0
    hand_detected = results.multi_hand_landmarks is not None

    if hand_detected:
        # Reset pause timer when hand is in frame
        last_time = time.time()

        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand.landmark]
        y_coords = [lm.y * h for lm in hand.landmark]
        x1 = max(0, int(min(x_coords)) - 50)
        x2 = min(w, int(max(x_coords)) + 50)
        y1 = max(0, int(min(y_coords)) - 50)
        y2 = min(h, int(max(y_coords)) + 50)
        crop = rgb[y1:y2, x1:x2]
        
        # Draw bounding box for visual feedback (Optional but helpful)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if crop.size > 0:
            img = Image.fromarray(crop)
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                if conf > CONF_THRESHOLD:
                    pred = classes[idx.item()]
                    confidence = conf.item()

    # STABILITY LOGIC
    if pred == current_prediction:
        stable_count += 1
    else:
        stable_count = 1
        current_prediction = pred

    # Detect if we just reached stable threshold
    just_stabilized = (stable_count >= STABLE_FRAMES) and (prev_stable_count < STABLE_FRAMES)
    prev_stable_count = stable_count

    # Set display_letter: show pred if stable (even if > STABLE_FRAMES), else "?"
    display_letter = pred if (stable_count >= STABLE_FRAMES and pred not in ["?", "nothing"]) else "?"

    # AUTO-SPACE ON PAUSE (hand removed for the new, longer duration)
    if buffer and (time.time() - last_time > AUTO_SPACE_PAUSE):
        sentence += buffer + " "
        buffer = ""
        print(f"Auto-spaced: {buffer}")  # Debug print

    # TRIGGER ACTIONS ONLY ON JUST_STABILIZED
    if just_stabilized:
        if pred in classes and classes.index(pred) in LETTER_INDICES:  # Is a letter (A-Z)
            buffer += pred
            print(f"Added letter: {pred} (buffer: {buffer})")  # Debug print
        elif pred == "space" and buffer:
            sentence += buffer + " "
            buffer = ""
            print(f"Spaced word: {buffer}")  # Debug print
        elif pred == "del" and len(buffer) > 0:
            buffer = buffer[:-1]
            print(f"Deleted: buffer now {buffer}")  # Debug print
        # For "nothing" or "?": no action

    # DISPLAY (enhanced for debugging)
    cv2.putText(frame, f"Pred: {pred} (stable: {stable_count}/{STABLE_FRAMES})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Display: {display_letter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Word: {buffer}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Hand: {'Yes' if hand_detected else 'No'} (Pause: {time.time() - last_time:.1f}s)", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Final sentence:", sentence.strip())