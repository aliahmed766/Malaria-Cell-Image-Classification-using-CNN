import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Malaria Cell Detection | CNN",
    page_icon="🦠",
    layout="centered"
)

# ----------------------------
# CUSTOM BACKGROUND + STYLING
# ----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(135deg, #042A2B, #5EB1BF);
    background-attachment: fixed;
    color: white;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: #FDFEFE;
    text-align: center;
    text-shadow: 1px 1px 2px #000;
}

.stButton>button {
    background-color: #20B2AA;
    color: white;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    padding: 10px 24px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #1E90FF;
    transform: scale(1.05);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL (CACHED)
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_corvit.h5")
    return model

model = load_model()

# ----------------------------
# TITLE SECTION
# ----------------------------
st.markdown("<h1>🧫 Malaria Cell Image Classification</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; font-size:18px;'>
        Upload a microscope cell image to detect whether it is:
        <br>🦠 <b>Parasitized (Malaria Detected)</b> or 
        🧫 <b>Uninfected (Healthy Cell)</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
uploaded_file = st.file_uploader("📸 Upload Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🧬 Uploaded Image", use_container_width=True)

    # Process button
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image... Please wait ⏳"):
            # Preprocess image
            img = np.array(image)
            img = cv2.resize(img, (128, 128)) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = model.predict(img)
            result = "🦠 Parasitized (Malaria Detected)" if prediction[0][0] > 0.5 else "🧫 Uninfected (Healthy Cell)"
            confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (100 - prediction[0][0] * 100)

        # Show results in styled card
        st.markdown(
            f"""
            <div style="
                background-color: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                color: #00FA9A;">
                ✅ Prediction: {result}<br>
                🎯 Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("📥 Please upload a cell image to begin the analysis.")

# Footer
st.markdown(
    """
    ---
    <div style='text-align:center; font-size:14px; color:#ccc;'>
        Made with ❤️ using TensorFlow & Streamlit | Malaria Detection Project
    </div>
    """,
    unsafe_allow_html=True,
)
