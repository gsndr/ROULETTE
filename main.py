import json
import os

import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle


from tensorflow.keras.models import load_model


import pandas as pd
import csv
from MAGNETO.lib import VecToImage
import numpy as np
from XAI import TrainModel
from XAI import ExplainationMap

import gc
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
from keras import Model
from XAI import TrainSingleOutput

# Parameters ../DS/UNSW-NB15/Training175341/Numeric/ classes:10 size 15 unsw
#12x12 ../DS/NSL-KDD/Multiclass/ classes 5  NSL-KDD
param = {"Max_A_Size": 12, "Max_B_Size": 12, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180,
         "dir": "../../DS/NSL-KDD/Multiclass/", "Mode": "MultiBAttention",  # Mode : CNN_Nature, CNN2, MultiAttention, MultiBAttention
         "LoadFromJson": True, "mutual_info": True, # Mean or MI
         "hyper_opt_evals": 20, "epoch": 150, "No_0_MI": False,  # True -> Removing 0 MI Features,
         "autoencoder": False, "cut": None, "dataset": 'NSL-KDD', "trainModel": False, "attentionLayer": True,
         "classes":5, "classification": "classification", "classificationBinary": "classificationBinary", "createImage": 1,#Create Attention heatmap
         }

# TODO delete
# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]

if not param["LoadFromJson"]: #TODO add multiclass
    print("Create images")
    data = {}
    with open(param["dir"] + 'Train.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": param["classes"]}
        df=data["Xtrain"]
        print(len(df.columns))
        #exit()
        data["Classification"] = data["Xtrain"][param["classification"]]
        data["Classification_b"] = data["Xtrain"][param["classificationBinary"]]
        del data["Xtrain"][param["classification"]]
        del data["Xtrain"][param["classificationBinary"]]
    with open(param["dir"] + 'Test.csv', 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest.astype(float)
        data["Ytest"] = data["Xtest"][param["classification"]]
        data["b_Ytest"] = data["Xtest"][param["classificationBinary"]]
        del data["Xtest"][param["classification"]]
        del data["Xtest"][param["classificationBinary"]]


    if param["No_0_MI"]:
        with open(param["dir"] + '0_MI.json') as json_file:
            j = json.load(json_file)
        data["Xtrain"] = data["Xtrain"].drop(columns=j)
        data["Xtest"] = data["Xtest"].drop(columns=j)
        print("0 MI features dropped!")

     # AUTOENCODER
    if param["autoencoder"]:
        autoencoder = load_model(param["dir"] + 'Autoencoder.h5')
        autoencoder.summary()
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encod2').output)
        encoder.summary()
        # usa l'encoder con predict sul train_X e poi su test_X. Io qui ho creato anche il dataframe per salvarlo poi come csv
        encoded_train = pd.DataFrame(encoder.predict(data["Xtrain"]))
        data["Xtrain"] = encoded_train.add_prefix('feature_')
        encoded_test = pd.DataFrame(encoder.predict(data["Xtest"]))
        data["Xtest"] = encoded_test.add_prefix('feature_')

        f_myfile = open(param["dir"] + 'Xtrain_auto.pickle', 'wb')
        pickle.dump(data["Xtrain"], f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'Xtest_auto.pickle', 'wb')
        pickle.dump(data["Xtest"], f_myfile)
        f_myfile.close()


    f_myfile = open(param["dir"] + 'Ytrain.pickle', 'wb')
    pickle.dump(data["Classification"], f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'b_Ytrain.pickle', 'wb')
    pickle.dump(data["Classification_b"], f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'Ytest.pickle', 'wb')
    pickle.dump(data["Ytest"], f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'b_Ytest.pickle', 'wb')
    pickle.dump(data["b_Ytest"], f_myfile)
    f_myfile.close()

    X_train, Y_train, X_test, Y_test = VecToImage.toImage(param, data, norm=False)

else:
    images = {}
    if param['mutual_info']:
        method = 'MI'
    else:
        method = 'Mean'

    f_myfile = open(param["dir"] + 'train_'+str(param['Max_A_Size'])+'x'+str(param['Max_B_Size'])+'_'+method+'.pickle', 'rb')
    images["Xtrain"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'Ytrain.pickle', 'rb')
    images["Classification"] = pickle.load(f_myfile)
    f_myfile.close()
    df=images["Classification"]
    print(df.nunique())
    

    f_myfile = open(param["dir"] + 'b_Ytrain.pickle', 'rb')
    images["Classification_b"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'test_'+str(param['Max_A_Size'])+'x'+str(param['Max_B_Size'])+'_'+method+'.pickle', 'rb')
    images["Xtest"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'Ytest.pickle', 'rb')
    images["Ytest"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'b_Ytest.pickle', 'rb')
    images["b_Ytest"] = pickle.load(f_myfile)
    f_myfile.close()




if param["trainModel"]:
    if param["Mode"]=="MultiBAttention":
        TrainModel.trainNN(param, images)
    elif param["Mode"]=="MultiAttention":
        TrainSingleOutput.trainNN(param, images)
        
path = 'models/'+ param["dataset"] + "res_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))
if param["No_0_MI"]:
    path = path + "_No_0_MI"
if param["mutual_info"]:
    path= path + "_MI"
else:
    path = path + "_Mean"
if param["attentionLayer"]:
    path = path + "_AL"
path=path+"_"+param["Mode"]
pathModel = path + "_model.tf"
model = load_model(pathModel)
model.summary()


print("Start prediction")
Utils.printPrediction(model, images, path, param['Mode'])

if param["createImage"] ==1:
    print("Create Attention Map dataset")
    ExplainationMap.createMapAttention(model, images,param ,method)





