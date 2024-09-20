import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import base64  # Import for encoding image to base64

# Load  model 
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model = tf.keras.models.load_model("D:/FinalProjectML/Brain Cancer project/brain_cancer.keras")  
    return model

model = load_model()

def set_background(image_file):
  
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('D:/FinalProjectML/Brain Cancer project/medical4.jpg')

# Function to preprocess and predict
def predict_image(image, model):
    image = image.resize((80, 80))  # Ensure the input size matches your model's input
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image 

    # Predict using the loaded model
    predictions = model.predict(img_array)
    
    class_names = ['healthy', 'glioma', 'meningioma', 'pituitary']
    predicted_class = class_names[np.argmax(predictions)]  # Get the highest probability class
    return predicted_class

# Streamlit app
st.title("Brain Cancer Prediction System")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a button for classification
    if st.button("Classify Image"):
        st.write("Classifying...")
        prediction = predict_image(image, model)
        st.write(f"The image is a **{prediction}**.")








