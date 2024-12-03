from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load ResNet50
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Extract features
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(np.expand_dims(image, axis=0))
    features = model.predict(image)
    return features.flatten()

# Extract features for all images
features_dict = {path: extract_features(path) for path in image_paths}
