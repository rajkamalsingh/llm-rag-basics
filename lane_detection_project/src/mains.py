import cv2
import numpy as np
from utils import region_of_interest, draw_lines

cap = cv2.VideoCapture('../data/raw_video.mp4')
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional)
    frame = cv2.resize(frame, (640, 480))
    frame_count+=1
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # NEW: Improve contrast
    equalized = cv2.equalizeHist(gray)

    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    v = np.median(blur)

    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    #edges = cv2.Canny(blur, lower, upper)
    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)


    # 4. Region of Interest
    roi = region_of_interest(edges)

    # 5. Hough Line Transform
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 6. Draw lines
    line_img = draw_lines(frame, lines)

    cv2.imshow("Lane Detection", line_img)

    #cv2.imwrite("../outputs/before.jpg", frame)
    #cv2.imwrite("../outputs/after_equalization.jpg", line_img)
    # SAVE INTERMEDIATE RESULTS
    if frame_count % 300 == 0:
        cv2.imwrite(f"../outputs/exp3/rain/frame_{frame_count}_original.jpg", frame)
        #cv2.imwrite(f"../outputs/frame_{frame_count}_gray.jpg", gray)
        #cv2.imwrite(f"../outputs/frame_{frame_count}_equalized.jpg", equalized)
        #cv2.imwrite(f"../outputs/frame_{frame_count}_blur.jpg", blur)
        cv2.imwrite(f"../outputs/exp3/rain/frame_{frame_count}_edges.jpg", edges)
        cv2.imwrite(f"../outputs/exp3/rain/frame_{frame_count}_roi.jpg", roi)
        cv2.imwrite(f"../outputs/exp3/rain/frame_{frame_count}_final.jpg", line_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()