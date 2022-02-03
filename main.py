import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas



# load image
# 0 means reading photo in grayscale
boundaries = [
    ([17, 15, 100], [120, 90, 200])
]
img = cv2.imread('images/road426.png',0)
actual_img = cv2.imread('images/road426.png')
img = cv2.medianBlur(img,5)

# Finds circles in a grayscale image using the Hough transform.
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=70,param2=100,minRadius=17,maxRadius=200)
# Evenly round to the given number of decimals. default 0
circles = np.uint16(np.around(circles))

roi=[]
x=0
y=0
x2=0
y2=0
height, width = actual_img.shape[:2]
blank_image = np.zeros((height, width, 3), dtype="uint8")
for i in circles[0,:]:
    x = i[0]-i[2]-5
    y = i[1]-i[2]-5
    x2 = i[0]+i[2]+5
    y2 = i[1] + i[2] + 5
    cv2.rectangle(actual_img, (x,y), (x2,y2), (0,255,0), 1 )
    roi = actual_img[y:y2, x:x2]
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(roi, lower, upper)
        output = cv2.bitwise_and(roi, roi, mask=mask)
        blank_image[y:y2, x:x2] = output
        # show the images
cv2.imshow("images", np.hstack([actual_img, blank_image]))
cv2.waitKey(0)
