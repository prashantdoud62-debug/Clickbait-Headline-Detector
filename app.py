import os
import streamlit as st
import joblib

st.set_page_config(page_title="Clickbait Headline Detector", page_icon="🎯", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained classification pipeline."""
    if not os.path.exists("model.joblib"):
        return None
    return joblib.load("model.joblib")


def main():
    st.title("🎯 Clickbait Headline Detector")
    st.write("Enter a headline below to check whether it is clickbait or standard news.")

    model = load_model()
    if model is None:
        st.error("Model file (`model.joblib`) not found. Please run `train.py` first.")
        return

    headline = st.text_input(
        "Headline:",
        placeholder="e.g. 21 Pictures That Will Restore Your Faith In Humanity",
    )

    if st.button("Predict"):
        if not headline.strip():
            st.warning("Please enter a headline first.")
        else:
            probs = model.predict_proba([headline])[0]
            pred = model.predict([headline])[0]
            label = "Clickbait" if pred == 1 else "News"
            confidence = probs[pred] * 100.0

            if pred == 1:
                st.error(f"**Prediction:** {label}")
            else:
                st.success(f"**Prediction:** {label}")

            st.metric(label="Confidence", value=f"{confidence:.2f}%")


if __name__ == "__main__":
    main()
