import cv2 
import pyautogui
import numpy as np
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Get the screen resolution of the first monitor
screen_width, screen_height = pyautogui.size()

# Create a window to display the captured screen
cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
logging.info(f"The screen resolution is {screen_width},{screen_height}")
while True:
    # Capture the screen of the first monitor
    screenshot = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))

    # Convert the screenshot to a format that OpenCV can handle
    screenshot_cv2 = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Display the captured screen in the OpenCV window
    cv2.imshow("Screen Capture", screenshot_cv2)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release OpenCV window and close it
cv2.destroyAllWindows()