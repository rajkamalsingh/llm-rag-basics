import cv2
import numpy as np
import os

#from lane_detection_project.src.mains import frame_count

output_dir = "../outputs/exp4_comparison"
os.makedirs(output_dir, exist_ok=True)

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height),
         (img.shape[1], height),
         (img.shape[1]//2, int(height*0.6))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines):
    line_img = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.5:
                continue

            cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 3)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)


def baseline_pipeline(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
    return draw_lines(frame, lines)


def improved_pipeline(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (5,5), 0)

    v = np.median(blur)
    lower = int(max(0, 0.66*v))
    upper = int(min(255, 1.33*v))

    edges = cv2.Canny(blur, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, 1)
    edges = cv2.erode(edges, kernel, 1)

    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)

    return draw_lines(frame, lines)


# Process your selected frames
frames = ["clear.jpg", "shadow.jpg", "bright.jpg", "blur.jpg"]
frame_count=0
cap = cv2.VideoCapture('../data/raw_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count+=1
    if frame_count % 300 == 0:
        base = baseline_pipeline(frame.copy())
        improved = improved_pipeline(frame.copy())

        cv2.imwrite(f"{output_dir}/{frame_count}_baseline.jpg", base)
        cv2.imwrite(f"{output_dir}/{frame_count}_improved.jpg", improved)