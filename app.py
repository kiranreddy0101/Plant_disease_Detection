import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---------- Dark Mode Styling with Google Fonts ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #121212;
        color: #000000;
    }
    .main-container {
        background-color: rgba(30, 30, 30, 0.95);
        padding: 2rem;
        border-radius: 16px;
        max-width: 900px;
        margin: 2rem auto;
        box-shadow: 0 0 20px rgba(0, 255, 150, 0.3);
    }
    h1, h3, p {
        color: #00ffcc;
        text-align: center;
    }
    .stButton > button {
        background-color: #00cc88;
        color: black;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #009966;
        color: white;
    }
    .css-1aumxhk, .css-ffhzg2, .stMarkdown {
        text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final.h5")

model = load_trained_model()

# ---------- Class Names and Fertilizer Tips ----------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

fertilizer_map = {
    'Apple___Apple_scab': 'Use copper-based fungicides',
    'Apple___Black_rot': 'Apply sulfur sprays or captan',
    'Apple___Cedar_apple_rust': 'Use myclobutanil or mancozeb',
    'Apple___healthy': 'No fertilizer needed',
    'Background_without_leaves': 'N/A',
    'Blueberry___healthy': 'Use ammonium sulfate',
    'Cherry___Powdery_mildew': 'Spray with sulfur or neem oil',
    'Cherry___healthy': 'Fertilize with potassium-rich mix',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Use nitrogen-balanced fertilizers',
    'Corn___Common_rust': 'Apply fungicides like propiconazole',
    'Corn___Northern_Leaf_Blight': 'Use mancozeb or chlorothalonil',
    'Corn___healthy': 'Apply NPK (20-20-20) for growth',
    'Grape___Black_rot': 'Spray with captan or mancozeb',
    'Grape___Esca_(Black_Measles)': 'Avoid excess nitrogen; apply phosphorus',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use Bordeaux mixture',
    'Grape___healthy': 'Balanced NPK mix',
    'Orange___Haunglongbing_(Citrus_greening)': 'Apply zinc & manganese-rich foliar sprays',
    'Peach___Bacterial_spot': 'Use oxytetracycline sprays',
    'Peach___healthy': 'Use low-nitrogen fertilizer',
    'Pepper_bell___Bacterial_spot': 'Spray copper-based bactericides',
    'Pepper_bell___healthy': 'Use 5-10-10 fertilizer',
    'Potato___Early_blight': 'Use azoxystrobin and increase potassium',
    'Potato___Late_blight': 'Apply metalaxyl-M fungicide',
    'Potato___healthy': 'Use nitrogen-rich compost',
    'Raspberry___healthy': 'Use 10-10-10 fertilizer',
    'Soybean___healthy': 'Use rhizobium inoculants + phosphorus',
    'Squash___Powdery_mildew': 'Use neem oil or sulfur-based spray',
    'Strawberry___Leaf_scorch': 'Apply copper-based fungicide',
    'Strawberry___healthy': 'Use phosphorus-heavy fertilizer',
    'Tomato___Bacterial_spot': 'Use copper sprays, avoid overhead watering',
    'Tomato___Early_blight': 'Apply chlorothalonil',
    'Tomato___Late_blight': 'Spray with mancozeb or copper-based fungicide',
    'Tomato___Leaf_Mold': 'Increase airflow and use fungicides',
    'Tomato___Septoria_leaf_spot': 'Apply fungicide with chlorothalonil',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil',
    'Tomato___Target_Spot': 'Apply fungicides like pyraclostrobin',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant varieties; spray imidacloprid',
    'Tomato___Tomato_mosaic_virus': 'Use resistant cultivars and disinfect tools',
    'Tomato___healthy': 'Use balanced NPK fertilizer (10-10-10)',
}

# Sidebar
st.sidebar.title("🌿 Plant Guardian")
st.sidebar.markdown("Upload a leaf image on the Detection tab to identify diseases and get fertilizer advice.")

# Tabs
tab1, tab2 = st.tabs(["🌱 Detection", "📘 Info"])

with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("## 🌿 Plant Disease Detection")
    st.markdown("<p style='font-size:16px;'>Upload a leaf image to identify the disease and get fertilizer suggestions.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.markdown(f"<div style='background-color:#1e1e1e;padding:20px;border-radius:12px; color:white;'>🩺 <strong>Prediction:</strong> {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#333333;padding:15px;border-radius:12px; color:white;'>🎯 <strong>Confidence:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)

        if predicted_class in fertilizer_map:
            st.markdown(f"<div style='background-color:#262626;padding:15px;border-radius:12px; color:white;'>💡 <strong>Fertilizer Tip:</strong> {fertilizer_map[predicted_class]}</div>", unsafe_allow_html=True)

        else:
            st.success("✅ This plant appears healthy. No treatment needed!")

with tab2:
  st.markdown("## 📘 About This App")
  st.markdown("""
   This application helps farmers and gardeners detect plant diseases from leaf images
   and recommends suitable fertilizers or treatments.

   **Features:**
   - Deep learning–based leaf disease classification
   - Custom fertilizer recommendations
   - Mobile-friendly responsive layout
   - Dark mode UI
   """)

