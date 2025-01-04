import numpy as np
import pygame
import av
from pygame.locals import *

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
model.load_state_dict(torch.load('fire_detection_model_zaina.pth'))
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


def is_fire(image):

    # # Load an image to classify
    # img_path = 'image7.png'  # Replace with the path to your image
    # image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB

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
    threshold = 0.95
    return False if probability > 1- threshold else True, probability
    # prediction = 'no fire' if probability > threshold else 'fire'

def main():
    pygame.init()

    # Set up the display
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Camera Recolor")

    # Initialize the camera using PyAV
    input_container = av.open("/dev/video0", format="v4l2")  # Change to 0 or other index on Windows

    try:
        for frame in input_container.decode(video=0):
            # Convert the frame to a numpy array
            frame_array = np.array(frame.to_image())
            print(is_fire(frame.to_image()))
            # frame_array = np.flip(frame_array, axis=0)  # Flip vertically if needed

            # Recolor the frame
            # recolored_frame = recolor_frame(frame_array)

            recolored_frame = frame_array

            # Display the recolored frame
            surface = pygame.surfarray.make_surface(np.transpose(recolored_frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Check for quit events
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                    return

    finally:
        # Release resources and quit pygame
        input_container.close()
        pygame.quit()

if __name__ == "__main__":
    main()
