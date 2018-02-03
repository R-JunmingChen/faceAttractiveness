# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import itertools
import dlib
import pickle
from sklearn import decomposition
from sklearn import linear_model
import glob
import os

prepath="./model/"
#dlib
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH =prepath+ "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#feature pointindices
indicesfilepath=prepath+"indices.pkl"
indicesfile = open(indicesfilepath, 'rb')
pointIndices = pickle.load(indicesfile)



def __facialRatio(points):
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x3 = points[4]
    y3 = points[5]
    x4 = points[6]
    y4 = points[7]

    dist1 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    dist2 = math.sqrt((x3-x4)**2 + (y3-y4)**2)

    ratio = dist1/dist2

    return ratio


def __generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, landmarkCoordinates):
    ratios = []
    for i in range(0, len(pointIndices1)):
        x1 = landmarkCoordinates[2 * (pointIndices1[i] - 1)]
        y1 = landmarkCoordinates[2 * pointIndices1[i] - 1]
        x2 = landmarkCoordinates[2 * (pointIndices2[i] - 1)]
        y2 = landmarkCoordinates[2 * pointIndices2[i] - 1]

        x3 = landmarkCoordinates[2 * (pointIndices3[i] - 1)]
        y3 = landmarkCoordinates[2 * pointIndices3[i] - 1]
        x4 = landmarkCoordinates[2 * (pointIndices4[i] - 1)]
        y4 = landmarkCoordinates[2 * pointIndices4[i] - 1]

        points = [x1, y1, x2, y2, x3, y3, x4, y4]
        ratios.append(__facialRatio(points))
    allFeatures = np.asarray(ratios)

    return allFeatures


def __feature_from_lankmark(landmarks):
    return __generateFeatures(pointIndices[0], pointIndices[1], pointIndices[2], pointIndices[3],
                              landmarks)


def __getlandmark_fromimg(img):
    rects = detector(img, 1)

    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        landmarks_faltten_list=[]
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            landmarks_faltten_list.append(point[0, 0])
            landmarks_faltten_list.append(point[0, 1])
        landmarks=np.array(landmarks_faltten_list)
        return landmarks
    return None

"""
predict face ratio from a img path
"""
def getfaceratio_fromimg(imgpath):
    #get features from pic
    image = cv2.imread(imgpath)
    landmarks = __getlandmark_fromimg(image)
    features = __feature_from_lankmark(landmarks)
    return features


def getfaceratios_fromDir(dirpath,imgratings):
    all_features=[]
    all_ratings=[]
    for filepath in glob.glob(os.path.join(dirpath,"*.jpg")):
        image = cv2.imread(filepath)
        landmarks = __getlandmark_fromimg(image)
        features = __feature_from_lankmark(landmarks)

        filename=filepath.split("/")[-1].split(".")[0]
        if "\\" in filename:
            filename=filename.split("\\")[-1]
        all_ratings.append(imgratings[filename])
        all_features.append(features)

    return all_features, all_ratings


def loadmodel():
    filePCA = open(prepath+"./PCAModel.pkl", 'rb')
    pca=pickle.load( filePCA)
    fileRegr = open(prepath+"./regrModel.pkl", 'rb')
    regr=pickle.load(fileRegr)

    return pca,regr



def predict(imgpath):
    """

    :param imgpath: 需要被预测颜值的图片路径
    :return:预测的颜值
    """
    #get the features
    image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
    landmarks = __getlandmark_fromimg(image)
    features = __feature_from_lankmark(landmarks)

    # predict facerank
    pca,regr=loadmodel()
    features_test = features.reshape([1, 11628])
    features_test = pca.transform(features_test)
    rank = regr.predict(features_test)

    # data structure
    rank = rank.tolist()[0]
    return rank

def train(imgdirpath,imgratings):
    """
    :param imgdirpath:训练图片所在目录的路径，批量读取目录下所有文件, 是一个字符串
    :param imgratings:训练图片对应的颜值的分数，用于做回归分析， 是一个字典，其中有每一个图片的名称，以及对应的颜值
    :return: 返回一个pca model用于对图片特征进行降维，一个线性回归模型，用于通过图片特征预测颜值
    """
    #get the features from imgs which is associated withe face attractiveness rank
    all_features,all_raingts=getfaceratios_fromDir(imgdirpath,imgratings)
    #PCA lower the demension of features
    pca = decomposition.PCA(n_components=20)
    pca.fit(all_features)
    features_train = pca.transform(all_features)


    regr = linear_model.LinearRegression()
    regr = regr.fit(features_train,all_raingts)

    #save model
    filePCA = open(prepath+"./PCAModel.pkl", 'wb')
    pickle.dump(pca, filePCA)
    fileRegr = open(prepath+"./regrModel.pkl", 'wb')
    pickle.dump(regr, fileRegr)

    print("train model done")



"""
generate  new point indices and save as a file
"""
def saveFeaturePointIndices(filepath):
    a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
    combinations = itertools.combinations(a, 4)
    i = 0
    pointIndices1 = []
    pointIndices2 = []
    pointIndices3 = []
    pointIndices4 = []

    for combination in combinations:
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[1])
        pointIndices3.append(combination[2])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[2])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[3])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[2])
        i = i + 1

    pointIndices=[pointIndices1,pointIndices2,pointIndices3,pointIndices4]
    file=open(filepath,'wb')
    pickle.dump(pointIndices,file)


if __name__ =="__main__":
    #这是一个训练的实例，输入图片的目录，以及每个图片对应的颜值就可以得到模型
    imgratings={"1":2.1,
                "2": 2.2,
                "3": 3.2,
                "4": 4.2,}
    train("./data",imgratings)

	#这是一个预测的实例，输入图片路径，可以预测得到颜值
    rank=predict("./data/4.jpg")
    print(rank)







