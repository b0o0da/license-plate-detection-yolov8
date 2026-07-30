"""
Streamlit app for License Plate Detection using a trained YOLOv8s model.

Run with:
    streamlit run app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = r"C:\Users\NAIRA\Desktop\New Github\license-plate-detection-yolov8\runs\detect\train\weights\best.pt"

st.set_page_config(page_title="License Plate Detection - YOLOv8s", page_icon="🚘", layout="centered")

# ------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load model from:\n`{MODEL_PATH}`\n\nError: {e}")
    st.stop()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🚘 License Plate Detection")
st.caption("Detect license plates in images using a custom-trained YOLOv8s model.")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    st.markdown("---")
    st.caption(f"Model: `best.pt`")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, width="stretch")

    # Convert PIL (RGB) to OpenCV/YOLO expected order (BGR)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection..."):
        results = model.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

    result = results[0]
    annotated = result.plot()  # returns BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Detected")
        st.image(annotated_rgb, width="stretch")

    # ------------------------------------------------------------------
    # Detection details
    # ------------------------------------------------------------------
    num_boxes = len(result.boxes)
    st.markdown("---")
    st.subheader(f"📋 Results ({num_boxes} plate{'s' if num_boxes != 1 else ''} detected)")

    if num_boxes == 0:
        st.info("No license plates detected. Try lowering the confidence threshold.")
    else:
        for i, box in enumerate(result.boxes, start=1):
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            st.write(
                f"**Plate {i}** — confidence: `{confidence:.2f}` — "
                f"bbox: `[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]`"
            )

    # ------------------------------------------------------------------
    # Download annotated image
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    st.download_button(
        "⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name="detected_plate.png",
        mime="image/png",
    )

else:
    st.info("👆 Upload an image to start detecting license plates.")