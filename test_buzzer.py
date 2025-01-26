import RPi.GPIO as GPIO
import time

# Pin configuration
OUTPUT_PIN = 17

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
GPIO.setup(OUTPUT_PIN, GPIO.OUT)  # Set pin as output

try:
    # Provide voltage to pin 17
    print(f"Setting GPIO pin {OUTPUT_PIN} to HIGH.")
    GPIO.output(OUTPUT_PIN, GPIO.HIGH)
    
    # Keep the pin HIGH for 10 seconds (for demonstration)
    time.sleep(10)
    
    # Optional: Turn the pin LOW
    print(f"Setting GPIO pin {OUTPUT_PIN} to LOW.")
    GPIO.output(OUTPUT_PIN, GPIO.LOW)

except KeyboardInterrupt:
    print("Script interrupted by user.")

finally:
    # Cleanup GPIO settings
    GPIO.cleanup()
    print("GPIO cleanup completed.")
