

MNIST Handwritten Digit Recognition with Live Webcam Detection
A deep learning project that trains a Neural Network on the MNIST dataset and detects handwritten digits in real time using a webcam.
Demo
Hold a handwritten digit inside the green box → model predicts the digit in real time with confidence percentage.
Model Performance

Test Accuracy: 97.34%
Precision, Recall, F1 Score: 97% across all 10 digit classes
Architecture: Dense Neural Network with Dropout regularization

Model Architecture
Input (784) → Dense(60, ReLU) → Dropout(0.2) → Dense(40, ReLU) → Dropout(0.2) → Dense(10, Linear)
Project Structure
mnist-digit-recognition/
├── mnist_digit_recognition.ipynb  # training, evaluation, confusion matrix
├── webcam_mnist.py                # real time webcam detection
└── README.md
How to Run
Training (Google Colab):
Open mnist_digit_recognition.ipynb in Google Colab and run all cells. Model trains in ~2 minutes on Colab GPU.
Webcam detection (local):

Install dependencies:

pip install tensorflow opencv-python numpy

Save the trained model as mnist_model.keras in the same folder
Run:

python webcam_mnist.py

Hold a handwritten digit (thick black marker on white paper) inside the green box
Press q to quit

Key Technical Details

MNIST digits are white on black background — real world paper is inverted. Applied cv2.bitwise_not() to bridge this gap
Used prediction buffer (last 15 frames) for stable real time output
Applied thresholding to convert grayscale to pure black and white for cleaner model input
Tuned model using bias-variance framework — added dropout to reduce overfitting from 2.49% gap to 0.58%

Dependencies

TensorFlow 2.x
OpenCV
NumPy
Scikit-learn

Author
Swetank — B.Tech ECE, NIT Silchar
