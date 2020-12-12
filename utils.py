import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import matplotlib.image as pltimg
from imgaug import augmenters as imgaugt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Dense,Flatten
from tensorflow.keras.optimizers import Adam

import os

def getName(filepath):
    return filepath.split("\\")[-1]

def importData(path):
    columns=["Center","Left","Right","Steering","Throttle","Break","Speed"]
    data = pd.read_csv(os.path.join(path,"driving_log.csv"),names = columns)
    # print(data.head())
    # print(data["Center"][0])
    # print(getName(data["Center"][0]))
    data["Center"]=data["Center"].apply(getName)
    # print(data["Center"][1])
    # print(len(data))
    return data


def balanceData(data):
    nbins=31
    samplepb=1000
    hist_val,bins=np.histogram(data["Steering"],nbins)
    center=(bins[:-1]+bins[1:])*0.5
    # plt.bar(center,hist_val,width = 0.06)
    # plt.plot((-1,1),(samplepb,samplepb))
    # plt.show()

    removelistdata=[]
    for i in range(nbins):
        binsdata =[]
        for j in range(len(data["Steering"])):
            if data["Steering"][j]>= bins[i] and data["Steering"][j]<= bins[i+1]:
                binsdata.append(j)
        binsdata=shuffle(binsdata)
        binsdata=binsdata[samplepb:]
        removelistdata.extend(binsdata)
    # print(len(removelistdata))

    data.drop(data.index[removelistdata], inplace=True)
    # print(len(data))
    # hist_val, _ = np.histogram(data["Steering"], nbins)
    # plt.bar(center,hist_val,width = 0.06)
    # plt.plot((-1,1),(samplepb,samplepb))
    # plt.show()
    return data


def splitData(path,data):
    imageDataList=[]
    steeringDataList=[]

    for i in range(len(data)):
        imageDataList.append(os.path.join(path,"IMG",data.iloc[i][0]))
        steeringDataList.append(float(data.iloc[i][3]))
    imageDataList=np.asarray(imageDataList)
    steeringDataList=np.asarray(steeringDataList)

    return imageDataList,steeringDataList


def ImageAug(imgpath,steeringAng):
    img=pltimg.imread(imgpath)

    if np.random.rand() <0.5:
        pan=imgaugt.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)

    if np.random.rand() < 0.5:
        zoom=imgaugt.Affine(scale=(1,2))
        img=zoom.augment_image(img)

    if np.random.rand() < 0.5:
        brightness=imgaugt.Multiply((0.5,1))
        img=brightness.augment_image(img)

    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steeringAng=-steeringAng

    return img ,steeringAng




def preprocessing(imgpath):
    img=imgpath[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img

# img=preprocessing(pltimg.imread('test.jpg'))
# # plt.imshow(img)
# # plt.show()


def batchGenerator(imgpathlist,steeringlist,batchsize,flag):
    while True:
        imgbatch=[]
        steeringbatch=[]
        for i in range(batchsize):
            index=random.randint(0,len(imgpathlist)-1)

            if flag:
                img,steering=ImageAug(imgpathlist[index],steeringlist[index])
            else:
                img=pltimg.imread(imgpathlist[index])
                steering=steeringlist[index]
            img=preprocessing(img)
            imgbatch.append(img)
            steeringbatch.append(steering)
            yield(np.asarray(imgbatch),np.asarray(steeringbatch))


def mymodel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')

    return model

