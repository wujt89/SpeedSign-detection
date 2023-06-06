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
from getpass import getpass


def checkCircle(x1, y1, x2, y2, r1, r2):
    distSq = (((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2))) ** (.5)
    if distSq + r2 <= r1:
        return True
    else:
        return False


def loadAndCirclePhoto(path):
    actual_img = cv2.imread(path)
    img = cv2.imread(path)
    # boundaries = [([20, 20, 0], [150, 150, 255])]
    # for (lower, upper) in boundaries:
    #     lower = np.array(lower, dtype="uint8")
    #     upper = np.array(upper, dtype="uint8")
    #     mask = cv2.inRange(img, lower, upper)
    #     img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    valueMin = width if width < height else height
    valueMax = width if width > height else height
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 30,
                               param1=300, param2=0.85, minRadius=int(valueMin / 20), maxRadius=int(valueMax / 2)+50)
    is_empty = False
    if circles is None:
        is_empty = True
    return checkAndDrawRedCircles(circles, actual_img, is_empty)


def checkAndDrawRedCircles(circles, actual_img, is_empty):
    if not is_empty:
        boxes = []
        circles = np.around(circles)
        newCircles = []
        for circle in circles:
            for circ in circle:
                n = 0
                for c in circle:
                    if circ[2] != c[2]:
                        if checkCircle(c[0], c[1], circ[0], circ[1], c[2], circ[2]):
                            n += 1
                            continue
                if n == 0:
                    newCircles.append(circ)
        height, width = actual_img.shape[:2]
        for i in newCircles:
            x = 0 if ((i[0] - i[2] < 0) or (i[0] - i[2] > width)) else i[0] - i[2]
            y = 0 if ((i[1] - i[2] < 0) or (i[1] - i[2] > height)) else i[1] - i[2]
            x2 = width if ((i[0] + i[2] < 0) or (i[0] + i[2] > width)) else i[0] + i[2]
            y2 = height if ((i[1] + i[2] < 0) or (i[1] + i[2] > height)) else i[1] + i[2]
            boxes.append({"xmin": str(int(float(x))), "xmax": str(int(float(x2))), "ymin": str(int(float(y))),
                          "ymax": str(int(float(y2)))})
        return boxes


def load(trainOrTest, xmlFolder):
    data = []
    for filename in glob.glob(
            f'{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/{trainOrTest}/{xmlFolder}/*.xml'):
        doc = minidom.parse(filename)
        imgName = doc.getElementsByTagName("filename")[0].firstChild.data
        objects = doc.getElementsByTagName("object")
        height = doc.getElementsByTagName("height")[0].firstChild.data
        width = doc.getElementsByTagName("width")[0].firstChild.data
        valueMin = width if width < height else height
        boxes = []
        for obj in objects:
            label = '0'
            typeOfSign = obj.getElementsByTagName("name")[0].firstChild.data
            xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
            ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
            xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
            ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
            if typeOfSign == "speedlimit" and ((int(xmax) - int(xmin)) > (int(valueMin) / 10)):
                label = '1'
            elif typeOfSign == "speedlimit" and ((int(xmax) - int(xmin)) < (int(valueMin) / 10)):
                continue
            else:
                typeOfSign = 'others'
            boxes.append({"typeOfSign": typeOfSign, "label": label, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
        # TODO Zmienna 'label' moze byc nie zainicjowana.
        # TODO Dobrze by bylo gdyby liczba przykladow negatywnych i pozytywnych byla podobna.
        if label == '0':
            xmin = random.randrange(0, int(width) - 150)
            ymin = random.randrange(0, int(height) - 150)
            xmax = random.randrange(xmin + 20, xmin + 120)
            ymax = random.randrange(ymin + 20, ymin + 120)
            boxes.append({"typeOfSign": 'others', "label": label, "xmin": str(xmin), "xmax": str(xmax), "ymin": str(ymin),
                 "ymax": str(ymax)})
        data.append({'imageName': imgName, 'height': height, 'width': width, "boxes": boxes})
    return data


def learn(data):
    bow = cv2.BOWKMeansTrainer(128)
    sift = cv2.SIFT_create()
    for element in data:
        img = cv2.imread(f'{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/train/images/{element["imageName"]}')
        boxes = element['boxes']
        for box in boxes:
            if box['typeOfSign'] == 'speedlimit':
                roi = img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])]
                kp = sift.detect(roi, None)
                kp, desc = sift.compute(roi, kp)
                if desc is not None:
                    bow.add(desc)
                w = random.randrange(1, 2)
                height, width = roi.shape[:2]
                # TODO Dodawanie takich specyficznych przypadkow moze wcale nie pomoc uczeniu.
                croppedw = random.randrange(int(width / 2) + 1, width - 1)
                croppedh = random.randrange(int(height / 2) + 1, height - 1)
                if w == 1:
                    roir = roi[0:croppedh, 0:width]
                if w == 2:
                    roir = roi[0:height, 0:croppedw]
                # TODO Z tego mozna by zrobic funkcje.
                kp = sift.detect(roir, None)
                kp, desc = sift.compute(roir, kp)
                if desc is not None:
                    bow.add(desc)
                if w == 1:
                    roil = roi[height-croppedh:height, 0:width]
                if w == 2:
                    roil = roi[0:height, width-croppedh:width]
                kp = sift.detect(roil, None)
                kp, desc = sift.compute(roil, kp)
                if desc is not None:
                    bow.add(desc)
                # TODO Przydalby sie komentarz.
                expandedRoi = img[(int(box["ymin"])-10):(int(box["ymax"])+10), (int(box["xmin"])-10):(int(box["xmax"])+10)]
                try:
                    kp = sift.detect(expandedRoi, None)
                    kp, desc = sift.compute(expandedRoi, kp)
                    if desc is not None:
                        bow.add(desc)
                except:
                    continue

    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)


def extractSinglePhotoClassify(name, coordinates):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{name}")
    roi = img[coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]]
    kp = sift.detect(roi, None)
    desc = bow.compute(roi, kp)
    if desc is None:
        desc = np.zeros((1, 128))
    return desc


def extractDetect(data, path):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for element in data:
        boxes = loadAndCirclePhoto(f'{path}{element["imageName"]}')
        element.update({'boxesFound': boxes})
        if boxes is not None:
            for box in element['boxesFound']:
                img = cv2.imread(f'{path}{element["imageName"]}')
                roi = img[int(float(box["ymin"])):int(float(box["ymax"])),
                      int(float(box["xmin"])):int(float(box["xmax"]))]
                kp = sift.detect(roi, None)
                desc = bow.compute(roi, kp)
                if desc is not None:
                    box.update({'desc': desc})
                else:
                    box.update({'desc': np.zeros((1, 128))})
    return data


def extract(data, path):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for element in data:
        for box in element["boxes"]:
            img = cv2.imread(f'{path}{element["imageName"]}')
            roi = img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])]
            kp = sift.detect(roi, None)
            desc = bow.compute(roi, kp)
            if desc is not None:
                box.update({'desc': desc})
            else:
                box.update({'desc': np.zeros((1, 128))})
    return data


def train(data):
    clf = RandomForestClassifier(128)
    x_matrix = np.empty((1, 128))
    y_vector = []
    for element in data:
        for box in element['boxes']:
            y_vector.append(box['label'])
            x_matrix = np.vstack((x_matrix, box['desc']))
    clf.fit(x_matrix[1:], y_vector)
    return clf


def predictSinglePhotoClassify(rf, desc):
    label = '0'
    if desc is not None:
        label = rf.predict(desc)
    if int(label) > 0:
        print("speedlimit")
    else:
        print("other")


def predictDetect(rf, data):
    for element in data:
        boxy = []
        n = 0
        if element["boxesFound"] is not None:
            for box in element["boxesFound"]:
                if box is not None:
                    box.update({'labelPred': rf.predict(box['desc'])[0]})
                    if int(rf.predict(box['desc'])[0]) > 0:
                        n += 1
                        boxy.append([box['xmin'], box['xmax'], box['ymin'], box['ymax']])

        print(element["imageName"])
        print(n)
        for i in range(n):
            print(f'{boxy[i][0]} {boxy[i][1]} {boxy[i][2]} {boxy[i][3]}')
    return data


def predictUniversal(rf, data):
    for element in data:
        for box in element["boxes"]:
            box.update({'labelPred': rf.predict(box['desc'])[0]})
            # print(box["labelPred"])
            # fileName = element['imageName']
            # img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{fileName}")
            # img = cv2.rectangle(img, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])),
            #                     (0, 255, 0), 1)
            # cv2.imshow("images", img)
            # cv2.waitKey(0)

    return data


def evaluate(data):
    yPred = []
    yReal = []
    for element in data:
        for box in element['boxes']:
            yPred.append(box['labelPred'])
            yReal.append(box['label'])
    print("Accuracy:", metrics.accuracy_score(yReal, yPred))
    return


def evaluateDetect(data):
    counter = 0
    for element in data:
        img = cv2.imread(f'{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{element["imageName"]}')
        if element["boxesFound"] is not None:
            for box in element['boxesFound']:
                img = cv2.rectangle(img, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])),
                                    (0, 255, 0), 1)
                if box['labelPred'] == '1':
                    counter = counter + 1
    counter = 0
    for element in data:
        for box in element["boxes"]:
            if box['label'] == '1':
                counter = counter + 1
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

    trainData = load("train", "annotations")
    testData = load("test", "annotations")
    learn(trainData)
    trainData = extract(trainData, f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/train/images/")
    rf = train(trainData)
    # print("extracting test")
    # test_data = extract(test_data, f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/")
    # print("testing")
    # predict(rf, test_data)
    # evaluate(test_data)
    # print("done")
    command = input()  # "classify or detect: "
    if command.lower() == 'classify':
        numberOfFiles = input()  # "number of files: "
        n = 0
        while n < int(numberOfFiles):
            fileName = input()  # "file name: "
            numberOfSigns = input()  # "number of signs: "
            for i in range(int(numberOfSigns)):
                coordinatesArray = []
                coordinates = input()  # "insert coordinates: "
                coordinateString = ''
                for letter in coordinates:
                    if letter == ' ':
                        coordinatesArray.append(int(coordinateString))
                        coordinateString = ''
                    coordinateString = coordinateString + letter
                coordinatesArray.append(int(coordinateString))
                img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{fileName}")
                cv2.rectangle(img, (coordinatesArray[0], coordinatesArray[2]),
                              (coordinatesArray[1], coordinatesArray[3]), (0, 255, 0), 1)
                desc = extractSinglePhotoClassify(fileName, coordinatesArray)
                predictSinglePhotoClassify(rf, desc)
    elif command.lower() == 'detect':
        testData = extractDetect(testData, f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/")
        predictDetect(rf, testData)
        # evaluateDetect(testData)
