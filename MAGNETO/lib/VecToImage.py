import csv
import json
import pickle
import timeit

import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

from MAGNETO.lib.Cart2Pixel import Cart2Pixel
from MAGNETO.lib.ConvPixel import ConvPixel
from MAGNETO.lib.deep import CNN_Nature, CNN2
import matplotlib.pyplot as plt

import time



def toImage(param, dataset, norm):
    np.random.seed(param["seed"])
    print("modelling dataset")
    global YGlobal
    YGlobal = to_categorical(dataset["Classification"])
    del dataset["Classification"]
    global YTestGlobal
    YTestGlobal = to_categorical(dataset["Ytest"])
    del dataset["Ytest"]

    global XGlobal
    global XTestGlobal

    # norm
    Out = {}
    if norm:
        print('NORM Min-Max')
        Out["Max"] = float(dataset["Xtrain"].max().max())
        Out["Min"] = float(dataset["Xtrain"].min().min())
        # NORM
        dataset["Xtrain"] = (dataset["Xtrain"] - Out["Min"]) / (Out["Max"] - Out["Min"])
        dataset["Xtrain"] = dataset["Xtrain"].fillna(0)

    # TODO implement norm 2
    print("trasposing")

    q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
         "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": np.argmax(YGlobal, axis=1)}
    print(q["method"])
    print(q["max_A_size"])
    print(q["max_B_size"])

    # generate images
    XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                mutual_info=param["mutual_info"], params=param, only_model=False)



    del q["data"]
    print("Train Images done!")
    # generate testingset image
    print(dataset["Xtest"].columns)

    if param["mutual_info"]:
        dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)



    #da qui
    x = image_model["xp"]
    y = image_model["yp"]
    col=dataset["Xtest"].columns
    #col =col.delete(0)
    print(col)
    coor_model = {"coord":  ["xp: " + str(i) + "," "yp :" + str(z) + ":" + col for i, z, col in zip(x, y, col)]}
    j = json.dumps(coor_model)
    f = open(param["dir"] + "MI_model.json", "w")
    f.write(j)
    f.close()

    dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()
    print("generating Test Images")
    print(dataset["Xtest"].shape)

    # if image_model["custom_cut"] is None:
    #     XTestGlobal = list([np.ones([int(image_model["A"]), int(image_model["B"])])] * dataset["Xtest"].shape[1])
    # else:
    #     XTestGlobal = list([np.ones([int(image_model["A"] - image_model["custom_cut"]), int(image_model["B"])])] * \
    #                   dataset["Xtest"].shape[1])
    # for i in range(0, 100):  # dataset["Xtest"].shape[1]):
    #     print(str(i) + " of " + str(dataset["Xtest"].shape[1]))
    #     XTestGlobal[i] = ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
    #                                image_model["A"], image_model["B"],
    #                                custom_cut=range(0, image_model["custom_cut"]))
    if image_model["custom_cut"]:
        XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                 image_model["A"], image_model["B"], custom_cut=range(0, image_model["custom_cut"]))
                    for i in range(0, dataset["Xtest"].shape[1])]  # dataset["Xtest"].shape[1])]
    else:
        XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                 image_model["A"], image_model["B"])
                       for i in range(0, dataset["Xtest"].shape[1])]  # dataset["Xtest"].shape[1])]

    print("Test Images done!")

    # saving testingset
    name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))
    if param["No_0_MI"]:
        name = name + "_No_0_MI"
    if param["mutual_info"]:
        name = name + "_MI"
    else:
        name = name + "_Mean"
    if image_model["custom_cut"] is not None:
        name = name + "_Cut" + str(image_model["custom_cut"])
    filename = param["dir"] + "test" + name + ".pickle"
    f_myfile = open(filename, 'wb')
    pickle.dump(XTestGlobal, f_myfile)
    f_myfile.close()

    # GAN
    del dataset["Xtrain"]
    del dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size1, image_size2 = XTestGlobal[0].shape
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size1, image_size2, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    return XGlobal, YGlobal, XTestGlobal, YTestGlobal


