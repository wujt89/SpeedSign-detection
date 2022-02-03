import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas

boundaries = [([17, 15, 80], [100, 100, 200])]
def loadAndCirclePhoto(path):
    img = cv2.imread(path, 0)
    actual_img = cv2.imread(path)
    img = cv2.medianBlur(img, 5)
    circles = [0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 20,
                               param1=300, param2=0.85, minRadius=10, maxRadius=200)
    print(circles)
    is_empty = False
    if circles is None:
        is_empty = True
    return circles, actual_img, is_empty


def checkAndDrawRedCircles(circles, actual_img, is_empty):
    if not is_empty:
        circles = np.uint16(np.around(circles))
        height, width = actual_img.shape[:2]
        blank_image = np.zeros((height, width, 3), dtype="uint8")
        for i in circles[0, :]:
            x = i[0] - i[2] - 5
            y = i[1] - i[2] - 5
            x2 = i[0] + i[2] + 5
            y2 = i[1] + i[2] + 5
            cv2.rectangle(actual_img, (x, y), (x2, y2), (0, 255, 0), 1)
            roi = actual_img[y:y2, x:x2]
            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(roi, lower, upper)
                output = cv2.bitwise_and(roi, roi, mask=mask)
                blank_image[y:y2, x:x2] = output
        cv2.imshow("images", np.hstack([actual_img, blank_image]))
        cv2.waitKey(0)


def main():
    circlesFound, img, is_empty = loadAndCirclePhoto('images/road426.png')
    checkAndDrawRedCircles(circlesFound, img, is_empty)


if __name__ == '__main__':
    main()
