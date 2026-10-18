import sys
import joblib


def predict_headline(headline: str):
    """Load model and predict whether headline is clickbait or news."""
    model = joblib.load("model.joblib")
    probs = model.predict_proba([headline])[0]
    pred = model.predict([headline])[0]

    label_str = "Clickbait" if pred == 1 else "News"
    confidence = probs[pred] * 100.0

    print(f"Headline  : {headline}")
    print(f"Prediction: {label_str}")
    print(f"Confidence: {confidence:.2f}%")


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "<headline>"')
        sys.exit(1)

    headline = " ".join(sys.argv[1:]).strip()
    if not headline:
        print("Error: Empty headline provided.")
        sys.exit(1)

    predict_headline(headline)


if __name__ == "__main__":
    main()
