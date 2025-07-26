

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

# 🔁 Replace with your actual image file path
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
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

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

