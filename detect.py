import cv2
import numpy as np
from model import load_model

# Load AI model
model = load_model()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face, (128, 128)) / 255.0  # Resize & normalize
        face_input = np.expand_dims(face_resized, axis=0)  # Add batch dimension

        # Predict using AI model
        prediction = model.predict(face_input)[0][0]

        # Display result
        label = "Real Face" if prediction > 0.5 else "Fake Face"
        color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
