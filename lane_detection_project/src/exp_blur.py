import os
import cv2
import numpy as np
from utils import region_of_interest, draw_lines


output_dir = "../outputs/exp1_blur"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

cap = cv2.VideoCapture('../data/raw_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (640, 480))

    # Select only a few frames for experiment
    if frame_count % 300 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Different blur sizes
    blur_kernels = [(3,3), (5,5), (9,9)]

    for k in blur_kernels:
        blur = cv2.GaussianBlur(equalized, k, 0)

        edges = cv2.Canny(blur, 50, 150)

        # Save output
        filename = f"{output_dir}/frame{frame_count}_blur{k[0]}_edges.jpg"
        cv2.imwrite(filename, edges)

        # Optional: save blur image also
        blur_name = f"{output_dir}/frame{frame_count}_blur{k[0]}.jpg"
        cv2.imwrite(blur_name, blur)

cap.release()