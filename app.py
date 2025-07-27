import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

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
            overflow: auto;
        }}
        .main-container {{
            background-color: rgba(0, 0, 0, 0.65); 
            padding: 2.5rem;
            border-radius: 16px;
            max-width: 900px;
            margin: 2rem auto;
        }}
        h1, h3, p {{
            color: #ffffff;
            text-align: center;
        }}
        footer, header, .stDeployButton {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_bg_from_local("download.jpg")

# ---------- Dark style for messages ----------
st.markdown("""
    <style>
    .stAlert {
        background-color: rgba(20, 20, 20, 0.85) !important;
        border-left: 0.4rem solid #00ff88 !important;
        color: white !important;
    }
    .stAlert p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🌱 Plant Guardian")
    st.markdown("This app detects plant leaf diseases using deep learning and suggests fertilizers.")
    st.info("📸 Upload a leaf image on the Detection tab.")

# ---------- Load Model ----------
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final.h5")

model = load_trained_model()

# ---------- Class Names ----------
class_names = [ ... ]  # Keep your existing class_names list

# ---------- Fertilizer Suggestions ----------
fertilizer_map = { ... }  # Keep your existing fertilizer_map

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🌿 Detection", "📘 About"])

with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("## 🌿 Plant Disease Detection")
    st.markdown(
        "<p style='color:white; font-size:18px; text-align:center;'>Upload a leaf image to identify the disease and get fertilizer suggestions.</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📘 About")
    st.markdown("""
        **Plant Guardian** uses a Convolutional Neural Network trained on the PlantVillage dataset to detect plant leaf diseases.
        
        It supports over 35 plant classes including:
        - 🍅 Tomato
        - 🥔 Potato
        - 🍇 Grape
        - 🌽 Corn
        - 🥭 Peach
        - 🍓 Strawberry

        🚀 **Features**:
        - Fast disease prediction
        - Fertilizer recommendation
        - Works entirely offline after deployment

        👉 Built using TensorFlow, Keras, and Streamlit.
    """)

