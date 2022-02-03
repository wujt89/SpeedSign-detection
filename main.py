import os
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import glob
import shutil
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas



def loadAndCirclePhoto(path):
    img = cv2.imread(path, 0)
    actual_img = cv2.imread(path)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 20,
                               param1=300, param2=0.85, minRadius=20, maxRadius=200)
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
            boundaries = [([17, 15, 40], [100, 100, 200])]
            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(roi, lower, upper)
                output = cv2.bitwise_and(roi, roi, mask=mask)
                blank_image[y:y2, x:x2] = output
        cv2.imshow("images", np.hstack([actual_img, blank_image]))
        cv2.waitKey(0)


def main():
    circlesFound, img, is_empty = loadAndCirclePhoto('images/road51.png')
    checkAndDrawRedCircles(circlesFound, img, is_empty)


if __name__ == '__main__':
    # Grouping pictures and xml files
    # for i in range(876):
    #     file = ET.parse(f"annotations/road{i}.xml")
    #     root = file.getroot()
    #     elemList = []
    #     found = 0
    #     for elem in root.iter("name"):
    #         elemList.append(elem.text)
    #     for el in elemList:
    #         if el == "speedlimit" :
    #             found = 1
    #             shutil.copy2(f'annotations/road{i}.xml', 'isSpeedAnnotations')
    #             shutil.copy2(f'annotations/road{i}.png', 'isSpeedLimit')
    #     if found == 0:
    #         shutil.copy2(f'annotations/road{i}.xml', 'notSpeedAnnotations')
    #         shutil.copy2(f'annotations/road{i}.png', 'notSpeedLimit')
    # Creating folders for training and testing
    # val = 0
    # for filename in glob.glob('notSpeedLimit/*.png'):
    #     val = val+1
    #     if val<148:
    #         shutil.copy2(filename, 'train')
    #     else:
    #         shutil.copy2(filename, 'test')
    # for filename in glob.glob('notSpeedAnnotations/*.xml'):
    #     val = val+1
    #     if val<148:
    #         shutil.copy2(filename, 'trainAnnotations')
    #     else:
    #         shutil.copy2(filename, 'testAnnotations')

    main()
