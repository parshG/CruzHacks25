import cv2
import numpy as np
import time
import tensorflow.lite as tflite

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Start webcam
cap = cv2.VideoCapture(0)
print("🚀 Running model... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess
    img = cv2.resize(frame, (width, height))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get top prediction
    top_index = np.argmax(output_data)
    confidence = output_data[top_index]
    label = labels[top_index]

    # Draw label on frame
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Grocery Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
