import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import joblib

# Load the pre-trained ResNet-50 model
# Look more into chagning this : 
# https://pytorch.org/vision/0.8/models.html
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()

# Preprocessing function (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_image(image_path: str):

    # Feature extraction function
    image = Image.open(image_path).convert("RGB")

    # batch dimension
    image = transform(image).unsqueeze(0) 

    # No need to calculate gradients
    with torch.no_grad(): 
        features = model(image)

    return features.squeeze().numpy().flatten()

def load_dataframe(df, image_folder: str, 
                   save=True,
                   feature_path="cache/image_features.npy",
                   label_path="cache/image_labels.npy"):
    """
    Args:
        df: DataFrame containing image labels
        image_folder: Folder containing images
        save: Whether to save the extracted features and labels
        feature_path: Path to save the extracted features
        label_path: Path to save the extracted labels
    Return: 
        X: Extracted features
        y: Extracted labels
    """
    # Process all images
    features = []
    labels = []

    if os.path.exists(feature_path) and os.path.exists(label_path):
        return np.load(feature_path), np.load(label_path)

    # go through each image and extract features
    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{idx}.png")
        if os.path.exists(image_path):
            feature_vector = extract_features_from_image(image_path)
            features.append(feature_vector)
            labels.append(row['region_label'])

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    if save:
        # Save for training
        np.save(feature_path, X)
        np.save(label_path, y)

    return X, y
    

def test(rel_image_path: str):
    image_features = extract_features_from_image(rel_image_path).reshape(1, -1)

    model_path = "cache/logistic_regression_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    classifier = joblib.load(model_path)

    predicted_cluster = classifier.predict(image_features)[0]

    return predicted_cluster

# For testing specific images
# image_path = "test7.png"
# predicted_cluster = test(image_path)
# print(f"Predicted cluster: {predicted_cluster}")