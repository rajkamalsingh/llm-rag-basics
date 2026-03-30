import numpy as np
import cv2


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height),
         (img.shape[1], height),
         (img.shape[1] // 2, int(height * 0.6))]
    ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_lines(img, lines):
    line_img = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid division by zero
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter near-horizontal lines
            if abs(slope) < 0.5:
                continue

            cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0), 3)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)