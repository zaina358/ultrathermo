import numpy as np
import pygame
from pygame.locals import *

from picamera2 import Picamera2
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms
from PIL import Image
import time

# Pin configuration (Uncomment if using GPIO)
# import RPi.GPIO as GPIO
# OUTPUT_PIN = 17
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(OUTPUT_PIN, GPIO.OUT)

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
    pygame.display.set_caption("Fire Detection with Pi Camera")

    # Initialize Pi Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            frame_array = picam2.capture_array()
            image = Image.fromarray(frame_array)

            fire_detected, prob = is_fire(image)
            print(f"Fire Detection: {fire_detected}, Probability: {prob:.4f}")

            # Uncomment to control GPIO
            # GPIO.output(OUTPUT_PIN, GPIO.HIGH if fire_detected else GPIO.LOW)

            # Convert and display the frame
            surface = pygame.surfarray.make_surface(np.transpose(frame_array, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Quit event
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                    return

    finally:
        picam2.close()
        pygame.quit()
        # GPIO.cleanup()

if __name__ == "__main__":
    main()
