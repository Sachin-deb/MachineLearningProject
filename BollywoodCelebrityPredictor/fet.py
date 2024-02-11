import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

# Load filenames from pickle file
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    features = model.predict(preprocessed_img)
    return features.flatten()

# Extract features for all images
features = []
for file in tqdm(filenames):
    features.append(extract_features(file, model))

# Save features to pickle file
pickle.dump(features, open('embedding.pkl', 'wb'))
