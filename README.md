# MNIST Handwritten Digit Recognition with Live Webcam Detection

A deep learning project that trains a Neural Network on the MNIST dataset 
and detects handwritten digits in real time using a webcam.

---

## Demo
Hold a handwritten digit inside the green box → model predicts the digit 
in real time with confidence percentage.

> **Tip:** Use a thick black marker on white paper for best results.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 97.34% |
| Macro Precision | 97% |
| Macro Recall | 97% |
| Macro F1 Score | 97% |

---

## Model Architecture
Input (784)
→ Dense(60, ReLU)
→ Dropout(0.2)
→ Dense(40, ReLU)
→ Dropout(0.2)
→ Dense(10, Linear + Softmax)
---

## Project Structure
mnist-digit-recognition/
├── mnist_digit_recognition.ipynb   # training, evaluation, confusion matrix
├── webcam_mnist.py                 # real time webcam detection
└── README.md
---

## How to Run

### Training (Google Colab)
1. Open `mnist_digit_recognition.ipynb` in Google Colab
2. Run all cells
3. Model trains in ~2 minutes on Colab GPU
4. Download the saved `mnist_model.keras` file

### Webcam Detection (Local)
1. Install dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```
2. Place `mnist_model.keras` in the same folder as `webcam_mnist.py`
3. Run:
```bash
python webcam_mnist.py
```
4. Hold digit inside green box — press `q` to quit

---

## Key Technical Details

- **Distribution shift fix** — MNIST digits are white on black, real paper is inverted. Applied `cv2.bitwise_not()` to match training format
- **Prediction smoothing** — buffer of last 15 frames used to stabilize real time output
- **Thresholding** — converted grayscale to pure black/white for cleaner model input
- **Dropout tuning** — reduced overfitting gap from 2.49% to 0.58% using dropout regularization

---

## Dependencies

- TensorFlow 2.x
- OpenCV
- NumPy
- Scikit-learn

---

## Author

**Swetank** — B.Tech ECE, NIT Silchar
