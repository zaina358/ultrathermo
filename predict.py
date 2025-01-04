from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms
from PIL import Image

# Load the model architecture
model = EfficientNet.from_pretrained('efficientnet-b0')

# Modify the final layer for binary classification
model._fc = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    # Single output for binary classification
    torch.nn.Linear(model._fc.in_features, 1),
    torch.nn.Sigmoid()  # Sigmoid activation for binary classification
)

# Load the trained weights
model.load_state_dict(torch.load('fire_detection_model.pth'))
model.eval()  # Set the model to evaluation mode

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image transform (same as used during training)
transform = transforms.Compose([
    # Resize to 224x224 as expected by EfficientNet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load an image to classify
img_path = 'image2.png'  # Replace with the path to your image
image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB

# Apply the transformations to the image
input_tensor = transform(image)

# Add batch dimension (PyTorch expects a batch of images)
input_tensor = input_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# Perform inference (forward pass)
with torch.no_grad():  # No need to compute gradients during inference
    output = model(input_tensor)

# The output is a probability (since we used Sigmoid in the final layer)
probability = output.item()  # Get the probability as a scalar

# Set threshold to classify
threshold = 0.5
prediction = 'no fire' if probability > threshold else 'fire'

# Print the result
print(f'Prediction: {prediction}, Probability: {probability:.4f}')
