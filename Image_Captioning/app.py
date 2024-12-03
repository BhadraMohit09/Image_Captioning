import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import numpy as np

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("captioning_model.h5")
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

# Load pre-trained feature extractor (ResNet50)
@st.cache_resource
def load_feature_extractor():
    from tensorflow.keras.applications import ResNet50
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model, tokenizer = load_model_and_tokenizer()
feature_extractor = load_feature_extractor()

# Maximum length of captions
MAX_LENGTH = 20

# Function to extract image features
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(np.expand_dims(image, axis=0))
    features = feature_extractor.predict(image)
    return features.flatten()

# Function to generate captions
def generate_caption(image_path, model, tokenizer, max_length):
    features = extract_features(image_path)
    input_text = "<start>"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")

        # Predict the next word
        predicted_id = model.predict([np.expand_dims(features, axis=0), sequence]).argmax()
        predicted_word = tokenizer.index_word.get(predicted_id, "")

        if predicted_word == "<end>" or not predicted_word:
            break

        input_text += " " + predicted_word

    return input_text.replace("<start>", "").strip()

# Streamlit UI
st.title("Image Captioning AI")
st.write("Upload an image to generate a descriptive caption!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Save the uploaded image to a temporary file
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(uploaded_image.read())

    # Display the uploaded image
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    # Generate and display the caption
    with st.spinner("Generating caption..."):
        caption = generate_caption(temp_path, model, tokenizer, MAX_LENGTH)
    st.success("Caption Generated!")
    st.write(f"**Caption:** {caption}")
