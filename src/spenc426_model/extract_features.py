import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import data_prep


def extract_features(image_folder="../../database/dataset", df=None):

    # Load ResNet
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Feature extraction function
    def extract_features(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)
        print(features.shape)
        return features.squeeze().numpy().flatten()  # Flatten output

    # Process all images
    features = []
    labels = []

    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{idx}.png")  
        if os.path.exists(image_path):
            feature_vector = extract_features(image_path)
            features.append(feature_vector)
            labels.append(row['region_label'])

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Save for training
    np.save("cache/image_features.npy", X)
    np.save("cache/image_labels.npy", y)
