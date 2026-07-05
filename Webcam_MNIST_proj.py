import tensorflow as tf
import numpy as np
import cv2 as cv
from collections import deque

model = tf.keras.models.load_model('mnist_model.keras')

cap = cv.VideoCapture(0)
prediction_buffer = deque(maxlen=15)  # increase buffer to 15

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    x1, y1 = w//2 - 120, h//2 - 120
    x2, y2 = w//2 + 120, h//2 + 120
    
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    resized = cv.resize(roi, (28, 28))
    _, thresholded = cv.threshold(resized, 128, 255, cv.THRESH_BINARY)
    inverted = cv.bitwise_not(thresholded)
    cv.imshow('Model View', cv.resize(inverted, (280, 280)))
    normalized = inverted / 255.0
    flattened = normalized.reshape(1, 784)

    logits = model(flattened)
    prob = tf.nn.softmax(logits)
    prediction = tf.argmax(prob, axis=1).numpy()[0]
    confidence = tf.reduce_max(prob).numpy() * 100

    prediction_buffer.append(prediction)
    stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

    cv.putText(frame, f"Digit: {stable_prediction} ({confidence:.1f}%)", (50, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv.imshow('MNIST Detector', frame)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
