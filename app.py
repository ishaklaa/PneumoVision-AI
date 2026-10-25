
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------
# 1. Configure the Streamlit page
# ---------------------------------------
st.set_page_config(
    page_title="ChestScan AI",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 ChestScan AI")
st.subheader("Chest X-ray Classification")
st.write(
    "Upload a chest X-ray image to classify it as "
    "NORMAL or PNEUMONIA using MobileNetV2."
)


# ---------------------------------------
# 2. Define file paths
# ---------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "saved_models"

MODEL_PATH = MODEL_DIR / "ChestScan_MobileNetV2_Finetuned.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ---------------------------------------
# 3. Load the trained model
# ---------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------
# 4. Load metadata
# ---------------------------------------
@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not METADATA_PATH.exists():
        st.error(f"Metadata file not found: {METADATA_PATH}")
        st.stop()

    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


IMG_SIZE = tuple(metadata["image_size"])
THRESHOLD = float(metadata.get("threshold", 0.5))


# ---------------------------------------
# 5. Upload an X-ray image
# ---------------------------------------
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


# ---------------------------------------
# 6. Process and predict
# ---------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded chest X-ray",
        width="stretch"
    )

    if st.button("Analyze X-ray", type="primary"):

        with st.spinner("Analyzing image..."):

            # Resize image to match the model input
            resized_image = image.resize(IMG_SIZE)

            # Convert image to NumPy array
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Apply MobileNetV2 preprocessing: [0, 255] -> [-1, 1]
            image_array = preprocess_input(image_array)

            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            image_array = np.expand_dims(image_array, axis=0)

            # Predict pneumonia probability
            prediction = model.predict(
                image_array,
                verbose=0
            )

            pneumonia_score = float(np.asarray(prediction).ravel()[0])

            # Classify using the configured threshold
            if pneumonia_score >= THRESHOLD:
                predicted_class = "PNEUMONIA"
            else:
                predicted_class = "NORMAL"

        # ---------------------------------------
        # 7. Display results
        # ---------------------------------------
        st.divider()
        st.subheader("Analysis Result")

        if predicted_class == "PNEUMONIA":
            st.error(f"Predicted class: {predicted_class}")
        else:
            st.success(f"Predicted class: {predicted_class}")

        st.metric(
            "Model's predicted pneumonia score",
            f"{pneumonia_score:.2%}"
        )

        st.progress(
            min(max(pneumonia_score, 0.0), 1.0)
        )

        st.caption(
            f"Classification threshold: {THRESHOLD:.0%}"
        )

        st.warning(
            "This is an experimental AI prediction, not a medical "
            "diagnosis. It must not replace evaluation by a qualified "
            "healthcare professional."
        )

else:
    st.info("Upload an X-ray image to get started.")
