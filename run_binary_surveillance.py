import cv2    
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

# --- Configuration ---
MODEL_PATH = "activity_binary_model.h5"
IMG_SIZE = (64, 64)
ROLLING_AVERAGE_LEN = 8 

# --- Load Model ---
print("⏳ Loading binary model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

predictions_queue = deque(maxlen=ROLLING_AVERAGE_LEN)

# --- Camera ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocess
    resized = cv2.resize(frame, IMG_SIZE)
    # Ensure this matches training (0-1 range)
    normalized = resized.astype("float32") / 255.0
    input_image = np.expand_dims(normalized, axis=0)

    # 2. Predict
    prediction = model.predict(input_image, verbose=0)[0][0]
    predictions_queue.append(prediction)
    
    # 3. Smooth results
    avg_score = np.mean(predictions_queue)

    # --- DEBUGGING: Print the score to the terminal ---
    # Watch this number! 
    # Close to 0.0 = Anomaly
    # Close to 1.0 = Normal
    print(f"Raw Score: {avg_score:.4f}") 

    # 4. Logic (Threshold 0.5)
    # If the score is high (closer to 1), it is Normal
    if avg_score > 0.5:
        label = "NORMAL"
        color = (0, 255, 0) # Green
    else:
        label = "ANOMALY DETECTED"
        color = (0, 0, 255) # Red

    # 5. Display (Text Only, No Percentage)
    # Alert bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), color, -1)
    
    # Centered text (approximate)
    cv2.putText(frame, label, (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.imshow("Binary Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()