import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
# from sklearn.metrics import accuracy_score

dataset_dir = 'fire_dataset'
# Define transformations
transform = transforms.Compose([
    # Resize to 224x224 as expected by EfficientNet
    transforms.Resize((224, 224)),
    # Apply random horizontal flip for augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # Apply random rotation for augmentation
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Calculate the number of validation samples (20% of total dataset)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Load pre-trained EfficientNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')

# Modify the final fully connected layer to output a single value (fire or no fire)
model._fc = nn.Sequential(
    nn.Dropout(0.2),
    # Single output for binary classification
    nn.Linear(model._fc.in_features, 1),
    nn.Sigmoid()  # Sigmoid to get values between 0 and 1 (probability of fire)
)


criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
# Adam optimizer with a small learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 10  # Set the number of epochs for training
print("Starting Training...")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs.squeeze(), labels.float()
                         )  # Squeeze to match dimensions
        running_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        # Convert to binary labels
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100

    print(f'Epoch [{
          epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


# Evaluation on validation set

model.eval()  # Set the model to evaluation mode
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation during evaluation
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Squeeze the output to remove unnecessary dimensions
        outputs = outputs.squeeze()  # Remove extra dimensions

        # Ensure labels are in float32
        labels = labels.float()

        # Compute loss
        loss = criterion(outputs, labels)  # Binary Cross-Entropy loss
        val_loss += loss.item()

        # Calculate accuracy
        # Apply threshold for binary classification (0 or 1)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = val_loss / len(val_loader)
val_accuracy = correct / total * 100

print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

torch.save(model.state_dict(), 'fire_detection_model.pth')
