import streamlit as st
from streamlit_toggle import st_toggle_switch
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Theme initialization
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Light/Dark mode toggle switch
mode = st_toggle_switch(
    label="Toggle Theme",
    key="theme_switch",
    default_value=(st.session_state["theme"] == "dark"),
    label_after=True,
    inactive_color="#DDD",
    active_color="#11567f",
    track_color="#29B5E8"
)

st.session_state["theme"] = "dark" if mode else "light"

# Set color variables
bg_color = '#121212' if st.session_state["theme"] == 'dark' else '#ffffff'
text_color = '#ffffff' if st.session_state["theme"] == 'dark' else '#000000'
card_bg = '#1e1e1e' if st.session_state["theme"] == 'dark' else '#f5f5f5'

# Dynamic Theme CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}

    h1, h3, p {{
        text-align: center;
        color: {text_color};
    }}

    .prediction-card {{
        margin: 1rem auto;
        padding: 1rem 2rem;
        border-radius: 16px;
        background-color: {card_bg};
        color: {text_color};
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        width: 90%;
        max-width: 600px;
        font-size: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final.h5")

model = load_trained_model()

# Class Labels and Fertilizer Tips
class_names = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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

fertilizer_map = { 'Apple___Apple_scab': 'Use copper-based fungicides',
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
    'Tomato___healthy': 'Use balanced NPK fertilizer (10-10-10)',}  

# Sidebar
st.sidebar.title("🌿 Plant Guardian")
st.sidebar.markdown(f"<p style='color:{text_color}; font-size: 16px;'>Upload a leaf image on the Detection tab to identify diseases and get fertilizer advice.</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🌱 Detection", "📘 Info"])

with tab1:
    st.markdown("## 🌿 Plant Disease Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_data}" alt="Uploaded Image" width="300"/>
                <p style="color: {text_color}; font-size: 14px;">Uploaded Image</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(f"<div class='prediction-card'>🔎 <strong>Prediction:</strong> {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-card'>🎯 <strong>Confidence:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)

        if predicted_class in fertilizer_map:
            st.markdown(f"<div class='prediction-card'>💡 <strong>Fertilizer Tip:</strong> {fertilizer_map[predicted_class]}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ This plant appears healthy. No treatment needed!")

with tab2:
    st.markdown("## 📘 About This App")
    st.markdown("""
    This AI-powered application helps farmers and gardeners detect plant diseases from leaf images 
    and recommends suitable fertilizers or treatments.

    **Features:**
    - Deep learning–based leaf disease classification  
    - Custom fertilizer recommendations  
    - Mobile-friendly responsive layout  
    - Dark mode UI with instant toggle
    """)
