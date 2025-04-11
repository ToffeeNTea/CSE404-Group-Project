# import torch
#
#
# device = torch.device('cuda')
# print("GPU is available")
# print('Device name:', torch.cuda.get_device_name(0))
#
#
# # Example of moving a tensor to the GPU
# tensor = torch.randn(3, 3)
# tensor_gpu = tensor.to(device)
#
# # Example of moving a model to the GPU
# model = torch.nn.Linear(10, 2)
# model.to(device)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "sorted_by_state/sorted_by_state"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_sbs_weights_2.pth"))
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

# print("Training model...")
#
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
#
#     for inputs, labels in progress_bar:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())
#
#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
#
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())
#
#     acc = accuracy_score(all_labels, all_preds)
#     print(f"Validation Accuracy: {acc:.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Final Validation Accuracy: {acc:.4f}")

# Save model weights
# torch.save(model.state_dict(), "resnet18_sbs_weights_2.pth")
# print("Model saved to resnet18_sbs_weights_2.pth")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# ... same setup as before ...
fig, ax = plt.subplots(figsize=(10, 10))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
# disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()  # helps with label cut-off
plt.savefig("confusion_matrix_2.png")  # 🔽 Save it here!
plt.show()

# model.load_state_dict(torch.load("alexnet_sbs_weights.pth"))
# model.to(device)
# model.eval()