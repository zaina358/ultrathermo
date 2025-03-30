import numpy as np
import pygame
import cv2  # OpenCV for camera input
from pygame.locals import *
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model._fc.in_features, 1),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load('fire_detection_model_zaina.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_fire(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    probability = output.item()
    threshold = 0.95
    return probability > threshold, probability

def main():
    pygame.init()
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Fire Detection with Raspberry Pi Camera")

    # Initialize Raspberry Pi Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Works for most RPi cameras

    if not cap.isOpened():
        print("Error: Could not access Raspberry Pi Camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame from OpenCV (BGR) to PIL (RGB)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            fire_detected, prob = is_fire(image)
            print(f"Fire Detection: {fire_detected}, Probability: {prob:.4f}")

            # Convert and display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Quit event
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                    return

    finally:
        cap.release()
        pygame.quit()

if __name__ == "__main__":
    main()
