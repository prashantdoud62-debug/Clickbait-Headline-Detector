# 🎯 Clickbait Headline Detector

A machine learning application and interactive Streamlit web dashboard to classify whether a news headline is **Clickbait** or **Standard News**.

## 🚀 Features
- **Machine Learning Pipeline**: TF-IDF n-gram vectorizer paired with a Logistic Regression classifier trained on headline datasets.
- **Interactive Web App**: Built with [Streamlit](https://streamlit.io/) for fast, intuitive real-time classification.
- **CLI Utility**: Fast terminal-based predictions via `predict.py`.
- **Model Evaluation**: Generates accuracy, precision, recall, F1 scores, and a confusion matrix visualization.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prashantdoud62-debug/Clickbait-Headline-Detector.git
   cd Clickbait-Headline-Detector
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test headlines with confidence scores.

### 2. Command-Line Predictions
```bash
python predict.py "21 Pictures That Will Restore Your Faith In Humanity"
```

### 3. Model Training & Evaluation
To retrain the model and regenerate evaluation metrics:
```bash
python train.py
```
This updates `model.joblib` and saves `confusion_matrix.png`.

---

## 📊 Evaluation
The model outputs key classification metrics (Accuracy, Precision, Recall, and F1 Score) along with a confusion matrix plot (`confusion_matrix.png`).
