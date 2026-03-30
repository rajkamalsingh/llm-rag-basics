import os
import cv2


output_dir = "../outputs/exp2_canny"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

cap = cv2.VideoCapture('../data/raw_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (640, 480))

    # Select frames
    if frame_count % 300 != 0:
        continue

    # Preprocessing (fixed)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Different Canny thresholds
    thresholds = [(50,150), (100,200), (150,300)]

    for (low, high) in thresholds:
        edges = cv2.Canny(blur, low, high)

        filename = f"{output_dir}/frame{frame_count}_canny_{low}_{high}.jpg"
        cv2.imwrite(filename, edges)

cap.release()