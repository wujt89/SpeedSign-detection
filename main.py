import os
import random
import numpy as np
import cv2
import glob
from xml.dom import minidom
import shutil
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas


def loadAndCirclePhoto(path):
    img = cv2.imread(path, 0)
    actual_img = cv2.imread(path)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 20,
                               param1=300, param2=0.97, minRadius=10, maxRadius=200)
    is_empty = False
    if circles is None:
        is_empty = True
    return checkAndDrawRedCircles(circles, actual_img, is_empty)


def checkAndDrawRedCircles(circles, actual_img, is_empty):
    if not is_empty:
        circles = np.uint16(np.around(circles))
        height, width = actual_img.shape[:2]
        # blank_image = np.zeros((height, width, 3), dtype="uint8")
        rois = []
        for i in circles[0, :]:
            x = 0 if ((i[0] - i[2] - 5 < 0) or (i[0] - i[2] - 5 > width)) else i[0] - i[2] - 5
            y = 0 if ((i[1] - i[2] - 5 < 0) or (i[1] - i[2] - 5 > height)) else i[1] - i[2] - 5
            x2 = width if ((i[0] + i[2] + 5 < 0) or (i[0] + i[2] + 5 > width)) else i[0] + i[2] + 5
            y2 = height if ((i[1] + i[2] + 5 < 0) or (i[1] + i[2] + 5 > height)) else i[1] + i[2] + 5
            cv2.rectangle(actual_img, (x, y), (x2, y2), (0, 255, 0), 1)
            roi = actual_img[y:y2, x:x2]
            rois.append(roi)
        #     boundaries = [([30, 30, 50], [150, 150, 255])]
        #     for (lower, upper) in boundaries:
        #         lower = np.array(lower, dtype="uint8")
        #         upper = np.array(upper, dtype="uint8")
        #         mask = cv2.inRange(roi, lower, upper)
        #         output = cv2.bitwise_and(roi, roi, mask=mask)
        #         blank_image[y:y2, x:x2] = output
        # cv2.imshow("images", np.hstack([actual_img, blank_image]))
        # cv2.waitKey(0)
        return rois


def load(xmlFolder):
    data = []
    for filename in glob.glob(f'{xmlFolder}/*.xml'):
        doc = minidom.parse(filename)
        imgName = doc.getElementsByTagName("filename")[0].firstChild.data
        objects = doc.getElementsByTagName("object")
        height = doc.getElementsByTagName("height")[0].firstChild.data
        width = doc.getElementsByTagName("width")[0].firstChild.data
        boxes = []
        for obj in objects:
            label = '0'
            typeOfSign = obj.getElementsByTagName("name")[0].firstChild.data
            xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
            ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
            xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
            ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
            if typeOfSign == "speedlimit" and ((int(xmax) - int(xmin)) > (int(height) / 10)):
                label = '1'
            boxes.append({"typeOfSign": typeOfSign, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
        data.append({'imageName': imgName, 'height': height, 'width': width, "label": label, "boxes": boxes})
    return data


def learn(data):
    for element in data:
        bow = cv2.BOWKMeansTrainer(37)
        sift = cv2.SIFT_create()
        img = cv2.imread(f'trainImages/{element["imageName"]}')
        boxes = element['boxes']
        for box in boxes:
            roi = img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            kp = sift.detect(gray, None)
            kp, desc = sift.compute(gray, kp)
            # roi = cv2.drawKeypoints(gray, kp, roi)
            # img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])] = roi
            if desc is not None:
                bow.add(desc)
        # cv2.imshow("roi", img)
        # cv2.waitKey(0)
    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)

    #     values = []
    #     objects = doc.getElementsByTagName("object")
    #     for obj in objects:
    #         name = obj.getElementsByTagName("name")[0]
    #         if name.firstChild.data == 'speedlimit':
    #             n += 1
    #             xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
    #             ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
    #             xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
    #             ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
    #             val = [xmin, xmax, ymin, ymax]
    #             values.append(val)
    #     if n>0:
    #         number+=1
    #         print(f'{filename[16:len(filename)-4]}.png')
    #         print(n)
    #         for v in values:
    #             print(v[0], v[1], v[2], v[3])
    # print(number)


def extract(data, path):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for element in data:
        rois = loadAndCirclePhoto(f'{path}{element["imageName"]}')
        if rois is not None:
            for roi in rois:
                kp = sift.detect(roi, None)
                desc = bow.compute(roi, kp)
                if desc is not None:
                    element.update({'desc': desc})
                else:
                    element.update({'desc': np.zeros((1, 37))})
        else:
            element.update({'desc': np.zeros((1, 37))})

        # for box in element["boxes"]:
        #     img = cv2.imread(f'{path}{element["imageName"]}')
        #     roi = img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])]
        #     kp = sift.detect(roi, None)
        #     desc = bow.compute(roi, kp)
        #     if desc is not None:
        #         element.update({'desc': desc})
        #     else:
        #         element.update({'desc': np.zeros((1, 37))})
        #     if box["typeOfSign"] == 'speedlight':
        #         break
    return data


def train(data):
    clf = RandomForestClassifier(100)
    x_matrix = np.empty((1, 37))
    y_vector = []
    for element in data:
        y_vector.append(element['label'])
        x_matrix = np.vstack((x_matrix, element['desc']))
    clf.fit(x_matrix[1:], y_vector)
    return clf


def predict(rf, data):
    for element in data:
        element.update({'label_pred': rf.predict(element['desc'])[0]})
        print(element["imageName"])
        print(element['label_pred'])
        img = cv2.imread(f"testImages/{element['imageName']}")
        cv2.imshow("roi", img)
        cv2.waitKey(0)
    return data


def evaluate(data):
    y_pred = []
    y_real = []
    for element in data:
        y_pred.append(element['label_pred'])
        y_real.append(element['label'])

    print("Accuracy:", metrics.accuracy_score(y_real, y_pred))
    return


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

    train_data = load("trainAnnotations")
    test_data = load("testAnnotations")
    print("learn")
    learn(train_data)
    print("extracting train")
    train_data = extract(train_data, "trainImages/")
    print("train")
    rf = train(train_data)
    print("extracting test")
    test_data = extract(test_data, "testImages/")
    print("testing")
    predict(rf, test_data)
    evaluate(test_data)
    print("done")
