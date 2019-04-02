import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def detect_blue(frame, background):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create HSV color mask and segment the image based on green color
    sensitivity = 20
    H_Value = 20
    light_blue = np.array([H_Value - sensitivity, 60, 60])
    dark_blue = np.array([H_Value + sensitivity, 255, 255])
    mask = cv2.inRange(hsv_image, light_blue, dark_blue)

    # Apply closing operation to fill out the unwanted gaps in the image. Bigger the kernel size, lesser the gaps
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the contour coordinates of the biggest area and return the x, y, w, h
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contour_mask = cv2.fillPoly(np.zeros((500, 500, 3), dtype=np.uint8), pts =[cont_sorted[0]], color=(255,255,255))

    object_mask = cv2.fillPoly(frame, pts =[cont_sorted[0]], color=(0,0,0))
    background_mask = np.bitwise_and(contour_mask, background)

    final_img = cv2.bitwise_or(object_mask, background_mask)
    
    return final_img



cap = cv2.VideoCapture(0)

ret, background = cap.read()
background = cv2.resize(background, (500, 500))
cv2.imshow('Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (500,500))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))
    image = detect_blue(frame, background)

    out.write(image)

    # Display the resulting frame
    cv2.imshow('Image',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
