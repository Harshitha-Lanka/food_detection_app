import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Food Detection App",
    page_icon="🍎",
    layout="wide"
)

st.title("🍽️ Food Detection App")
st.write("Upload a food image and detect items using your trained YOLO model.")

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# -------------------------
# Prediction
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    try:
        results = model.predict(
            source=temp_path,
            conf=confidence,
            save=False
        )

        result = results[0]

        # Plot predicted image
        plotted_image = result.plot()
        
        with col2:
            st.subheader("Detected Output")
            st.image(plotted_image[:, :, ::-1], use_container_width=True)

        # Extract detections
        st.subheader("Detection Results")

        names = model.names
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for cls_id, conf_score in zip(cls_ids, confs):
                detections.append({
                    "Food Item": names[int(cls_id)],
                    "Confidence": round(float(conf_score), 4)
                })

            df = pd.DataFrame(detections)
            st.dataframe(df, use_container_width=True)

            # Summary counts
            st.subheader("Detection Summary")
            count_df = df["Food Item"].value_counts().reset_index()
            count_df.columns = ["Food Item", "Count"]
            st.dataframe(count_df, use_container_width=True)

        else:
            st.warning("No food items detected in this image.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("Please upload an image to begin.")