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

    actual_img = cv2.imread(path)
    img = cv2.imread(path)
    # boundaries = [([20, 20, 0], [150, 150, 255])]
    # for (lower, upper) in boundaries:
    #     lower = np.array(lower, dtype="uint8")
    #     upper = np.array(upper, dtype="uint8")
    #     mask = cv2.inRange(img, lower, upper)
    #     img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("images", np.hstack([actual_img, img]))
    # cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 50,
                               param1=300, param2=0.95, minRadius=15, maxRadius=200)
    print(circles)
    is_empty = False
    if circles is None:
        is_empty = True
    return checkAndDrawRedCircles(circles, actual_img, is_empty)


def checkAndDrawRedCircles(circles, actual_img, is_empty):
    if not is_empty:
        circles = np.uint16(np.around(circles))
        height, width = actual_img.shape[:2]
        rois = []
        for i in circles[0, :]:
            x = 0 if ((i[0] - i[2] < 0) or (i[0] - i[2] > width)) else i[0] - i[2]
            y = 0 if ((i[1] - i[2] < 0) or (i[1] - i[2] > height)) else i[1] - i[2]
            x2 = width if ((i[0] + i[2] < 0) or (i[0] + i[2] > width)) else i[0] + i[2]
            y2 = height if ((i[1] + i[2] < 0) or (i[1] + i[2] > height)) else i[1] + i[2]
            cv2.rectangle(actual_img, (x, y), (x2, y2), (0, 255, 0), 1)
            # cv2.imshow("images", actual_img)
            # cv2.waitKey(0)
            roi = actual_img[y:y2, x:x2]
            rois.append(roi)
        return rois


def load(trainOrTest, xmlFolder):
    data = []
    for filename in glob.glob(f'{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/{trainOrTest}/{xmlFolder}/*.xml'):
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
            else:
                typeOfSign = 'others'
            boxes.append({"typeOfSign": typeOfSign, "label": label, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
        if label == '0':
            xmin = random.randrange(int(width)-100)
            xmax = xmin + random.randrange(20, 50)
            ymin = random.randrange(int(height)-100)
            ymax = ymin + random.randrange(20, 50)
            boxes.append({"typeOfSign": 'others', "label": label, "xmin": str(xmin), "xmax": str(xmax), "ymin": str(ymin), "ymax": str(ymax)})
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
                # roi = cv2.drawKeypoints(gray, kp, roi)
                # img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])] = roi
                if desc is not None:
                    bow.add(desc)
                w = random.randrange(1, 4)
                height, width = roi.shape[:2]
                croppedw = random.randrange(int(width / 2) + 5, width - 10)
                croppedh = random.randrange(int(height / 2) + 5, height - 10)
                if w == 1:
                    roi = roi[0:croppedh, 0:width]
                if w == 2:
                    roi = roi[0:height, 0:croppedw]
                kp = sift.detect(roi, None)
                kp, desc = sift.compute(roi, kp)
                if desc is not None:
                    bow.add(desc)
                # cv2.imshow("roi", roi)
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


def extractSinglePhotoClassify(name, coordinates):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{name}")
    roi = img[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
    kp = sift.detect(roi, None)
    desc = bow.compute(roi, kp)
    return desc


def extractSinglePhotoDetect(name, coordinates):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{name}")
    roi = img[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
    kp = sift.detect(roi, None)
    desc = bow.compute(roi, kp)
    return desc

def extract(data, path):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for element in data:
        # 10/10 - 85%
        rois = loadAndCirclePhoto(f'{path}{element["imageName"]}')
        desces = []
        if rois is not None:
            for roi in rois:
                kp = sift.detect(roi, None)
                desc = bow.compute(roi, kp)
                if desc is not None:
                    element.update({'desc': desc})
                else:
                    element.update({'desc': np.zeros((1, 128))})
        else:
            element.update({'desc': np.zeros((1, 128))})

        # classify - 100%
        # for box in element["boxes"]:
        #     img = cv2.imread(f'{path}{element["imageName"]}')
        #     roi = img[int(box["ymin"]):int(box["ymax"]), int(box["xmin"]):int(box["xmax"])]
        #     kp = sift.detect(roi, None)
        #     desc = bow.compute(roi, kp)
        #     if desc is not None:
        #         box.update({'desc': desc})
        #     else:
        #         box.update({'desc': np.zeros((1, 128))})
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

def predictSinglePhoto(rf, desc):
    label = rf.predict(desc)
    if int(label) > 0:
        print("speedlimit")
    else:
        print("others")


def predict(rf, data):
    for element in data:
        for box in element["boxes"]:
            box.update({'label_pred': rf.predict(box['desc'])[0]})
        # print(element["imageName"])
        # print(element['label_pred'])
        # img = cv2.imread(f"testImages/{element['imageName']}")
        # cv2.imshow("roi", img)
        # cv2.waitKey(0)
    return data


def evaluate(data):
    y_pred = []
    y_real = []
    for element in data:
        for box in element['boxes']:
            y_pred.append(box['label_pred'])
            y_real.append(box['label'])

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

    print("loading")
    train_data = load("train", "annotations")
    test_data = load("test", "annotations")
    print("learning")
    learn(train_data)
    print("extracting train")
    train_data = extract(train_data, f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/train/images/")
    print("training")
    rf = train(train_data)
    print("ready for action")
    command = input("classify or detect: ")
    if command.lower() == 'classify':
        numberOfFiles = input("number of files: ")
        n = 0
        while n < int(numberOfFiles):
            fileName = input("file name: ")
            numberOfSigns = input("number of signs: ")
            for i in range(int(numberOfSigns)):
                coordinatesArray = []
                coordinates = input("insert coordinates: ")
                coordinateString = ''
                for letter in coordinates:
                    if letter == ' ':
                        coordinatesArray.append(int(coordinateString))
                        coordinateString = ''
                    coordinateString = coordinateString + letter
                coordinatesArray.append(int(coordinateString))
                print(coordinatesArray)
                img = cv2.imread(f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/{fileName}")
                cv2.rectangle(img, (coordinatesArray[0], coordinatesArray[2]), (coordinatesArray[1], coordinatesArray[3]), (0, 255, 0), 1)
                cv2.imshow("images", img)
                cv2.waitKey(0)
                desc = extractSinglePhotoClassify(fileName, coordinatesArray)
                predictSinglePhoto(rf, desc)
    elif command.lower() == 'detect':
        photoName = input("photo name: ")

    # print("extracting test")
    # test_data = extract(test_data, f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/test/images/")
    # print("testing")
    # predict(rf, test_data)
    # evaluate(test_data)
    # print("done")
