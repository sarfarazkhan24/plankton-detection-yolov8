import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Plankton AI Detector",
    page_icon="🔬",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main-title {
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    font-size:18px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown('<p class="main-title">🔬 Plankton Detection AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect plankton in microscope images using YOLOv8</p>', unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙️ Controls")

conf = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

st.sidebar.markdown("---")
st.sidebar.info("Model: YOLOv8 trained for plankton detection")

# ---------------- LOAD MODEL ---------------- #

model = YOLO("best.pt")

# ---------------- IMAGE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload Microscope Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- INFERENCE ---------------- #

if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    with st.spinner("Running plankton detection..."):
        results = model(image_np, conf=conf)

    annotated = results[0].plot()

    # ---------------- DISPLAY ---------------- #

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(annotated, use_column_width=True)

else:
    st.info("Upload a microscope image to run plankton detection.")