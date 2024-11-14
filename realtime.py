import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model("ISLR_model.h5")

# Manually define the class names as a list (order matters)
class_names = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J','H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z'] # Ensure the order matches your model's output

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle for hand gesture capture area
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cropped_frame = frame[100:300, 100:300]

    # Preprocess the cropped frame for prediction
    img = cv2.resize(cropped_frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = class_names[class_index]  # Use list indexing
    confidence = np.max(prediction)

    # Display the prediction on the frame
    cv2.putText(frame, f"{class_label} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('ISL Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
