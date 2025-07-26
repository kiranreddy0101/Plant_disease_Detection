
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# ---------- Set Background Image ----------
def set_bg_from_local(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        html, body {{
            background-image: url("data:image/png;base64,{img_bytes}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔁 Replace with your actual image file
set_bg_from_local("background.jpg")

# ---------- Styling ----------
st.markdown("""
    <style>
    .hero-box {
        background: rgba(0, 50, 0, 0.3);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 50px;
        width: 100%;
        max-width: 800px;
        margin: 100px auto;
        text-align: center;
        box-shadow: 0 0 25px #00ff88;
        backdrop-filter: blur(12px);
    }
    .hero-box h1 {
        font-size: 60px;
        color: #00ff88;
        margin-bottom: 10px;
        font-weight: 800;
    }
    .hero-box h3 {
        font-weight: 400;
        color: #ffffffee;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .hero-box p {
        color: #ffffffcc;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .hero-button {
        background-color: #00ff88;
        color: #000000;
        font-size: 16px;
        padding: 12px 26px;
        border-radius: 30px;
        text-decoration: none;
        font-weight: bold;
        transition: background 0.3s;
        border: none;
    }
    .hero-button:hover {
        background-color: #00cc66;
        cursor: pointer;
    }
    footer, header, .stDeployButton { display: none; }
    </style>
""", unsafe_allow_html=True)

# ---------- UI Banner ----------
st.markdown("""
    <div class="hero-box">
        <h3>🌿 AI POWERED</h3>
        <h1>Plant Disease Detection</h1>
        <p>Detect plant diseases using deep learning and get fertilizer advice.</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final.h5")

model = load_trained_model()

# ---------- Class Names and Fertilizer Tips ----------
class_names = [
    'Apple_leaf', 'Apple_rust_leaf', 'Apple_Scab_Leaf', 'Bell_pepper_leaf', 'Bell_pepper_leaf__spot',
    'Blueberry_leaf', 'Cherry_leaf', 'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'grape_leaf_black_rot',
    'Peach_leaf', 'Potato_leaf_early_blight', 'Potato_leaf_late_blight', 'Raspberry_leaf', 'Soyabean_leaf',
    'Squash_Powdery_mildew_leaf', 'Strawberry_leaf', 'Tomato_Early_blight_leaf', 'Tomato_leaf',
    'Tomato_leaf_late_blight', 'Tomato_leaf_bacterial_spot', 'Tomato_leaf_mosaic_virus',
    'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_Septoria_leaf_spot',
    'Tomato_two_spotted_spider_mites_leaf'
]

fertilizer_map = {
    'Apple_rust_leaf': 'Apply sulfur-based fungicides. Prune affected areas.',
    'Apple_Scab_Leaf': 'Use mancozeb or captan fungicides. Remove infected leaves.',
    'Bell_pepper_leaf__spot': 'Use copper-based bactericide. Avoid overhead irrigation.',
    'Corn_Gray_leaf_spot': 'Apply fungicide like azoxystrobin. Use resistant hybrids.',
    'Corn_leaf_blight': 'Spray fungicide early. Use crop rotation techniques.',
    'grape_leaf_black_rot': 'Use myclobutanil or mancozeb. Prune infected vines.',
    'Potato_leaf_early_blight': 'Apply chlorothalonil or copper-based fungicides.',
    'Potato_leaf_late_blight': 'Use phosphorus acid or cymoxanil sprays.',
    'Squash_Powdery_mildew_leaf': 'Apply neem oil or sulfur fungicide.',
    'Tomato_Early_blight_leaf': 'Use mancozeb fungicide. Remove lower infected leaves.',
    'Tomato_leaf_late_blight': 'Spray with copper or potassium bicarbonate fungicide.',
    'Tomato_leaf_bacterial_spot': 'Use copper fungicides. Avoid wetting foliage.',
    'Tomato_leaf_mosaic_virus': 'Remove infected plants. Use certified virus-free seeds.',
    'Tomato_leaf_yellow_virus': 'Control whiteflies. Remove infected plants.',
    'Tomato_mold_leaf': 'Use sulfur or potassium bicarbonate sprays. Improve ventilation.',
    'Tomato_Septoria_leaf_spot': 'Apply chlorothalonil. Remove lower leaves to reduce splash.',
    'Tomato_two_spotted_spider_mites_leaf': 'Use miticides or insecticidal soap. Increase humidity.'
}

# ---------- Upload and Predict ----------
st.subheader("📷 Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🩺 **Prediction:** {predicted_class}")
    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

    if predicted_class in fertilizer_map:
        st.warning(f"💡 **Fertilizer Suggestion:** {fertilizer_map[predicted_class]}")
    else:
        st.success("✅ This plant appears healthy. No treatment needed!")
