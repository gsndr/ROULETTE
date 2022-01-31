import numpy as np
from keras import Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D,  Dense, Flatten, Lambda, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
import csv
my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)
from visual_attention import PixelAttention2D

import tensorflow
tensorflow.random.set_seed(12)

import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split

import time
from tensorflow import reduce_mean
from tensorflow import expand_dims
import Utils
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score, confusion_matrix


XGlobal = []
YGlobal = []
YGlobal_b = []

XTestGlobal = []
YTestGlobal = []
YTestGlobal_b = []

SavedParameters = []
Mode = ""
Name = ""
best_val_acc = 0
best_val_loss = np.inf
paramsGlobal=0

def trainNN(param,dataset):
    print("modelling dataset")
    global paramsGlobal
    paramsGlobal=param

    global YGlobal
    YGlobal = to_categorical(dataset["Classification"])
    #del dataset["Classification"]

    global YTestGlobal
    YTestGlobal = to_categorical(dataset["Ytest"])
    #del dataset["Ytest"]

    global YGlobal_b
    # YGlobal_b = to_categorical(dataset["Classification_b"])
    YGlobal_b = dataset["Classification_b"]
    #del dataset["Classification_b"]

    global YTestGlobal_b
    # YTestGlobal_b = to_categorical(dataset["b_Ytest"])
    YTestGlobal_b = dataset["b_Ytest"]
    #del dataset["b_Ytest"]

    global XGlobal
    global XTestGlobal

    XGlobal = dataset["Xtrain"]
    XTestGlobal = dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size1, image_size2 = XTestGlobal[0].shape
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size1, image_size2, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    global Mode
    Mode = param["Mode"]

    global Name
    Name = 'models/'+param["dataset"] + "res_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))
    if param["No_0_MI"]:
        Name = Name + "_No_0_MI"
    if param["mutual_info"]:
        Name = Name + "_MI"
    else:
        Name = Name + "_Mean"
    if param["attentionLayer"]:
        Name = Name + "_AL"
    Name = Name + "_" + Mode + ".csv"
    trials = Trials()
    if Mode == "MultiBAttention":
        hyperparams = {"kernel": hp.choice("kernel", np.arange(2, 4 + 1)),  # prima:2, 3
                           "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                           'dropout1': hp.uniform("dropout1", 0, 0.5),
                           #'dropout2': hp.uniform("dropout2", 0, 0.5),
                           "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                           "epoch": param["epoch"],
                           "w": hp.uniform("w", 0.5, 0.9),
                           "filter1": hp.choice("filter1", [16, 32, 64, 128,256, 512]),
                           "unit1": hp.choice("unit1", [16, 32, 64, 128, 256, 512]),
                           "unit2": hp.choice("unit2", [16, 32, 64, 128, 256, 512])
                           }
    fmin(hyperopt_fcn, hyperparams, trials=trials, algo=tpe.suggest, max_evals=param["hyper_opt_evals"])



    print("done")



def hyperopt_fcn(params):
    global SavedParameters

    start_time = time.time()

    print("start train")


    if Mode == "MultiBAttention":
        model, val = MultiBAttention(XGlobal, YGlobal, YGlobal_b, params)

    time_training = time.time() - start_time

    print("start predict")

    start_time = time.time()
    print(XTestGlobal.shape)
    (Y_predicted, Y_b_predicted) = model.predict(XTestGlobal, verbose=0, use_multiprocessing=True, workers=12)
    print(Y_predicted.shape)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    print(Y_predicted.shape)
    print(Y_b_predicted.shape)
    YTestGlobal_bb = ((Y_b_predicted > 0.5) + 0).ravel()

    time_predict = time.time() - start_time

    cf = confusion_matrix(YTestGlobal_b, YTestGlobal_bb)
    cm_test = [[cf[0][0], cf[0][1]], [cf[1][0], cf[1][1]]]
    r = Utils.res(cm_test)

    precision_macro_t, recall_macro_t, fscore_macro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='macro')
    precision_micro_t, recall_micro_t, fscore_micro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='micro')
    precision_weighted_t, recall_weighted_t, fscore_weighted_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                          Y_predicted,
                                                                                                          average='weighted')

    accuracy_t = accuracy_score(YTestGlobal, Y_predicted)

    K.clear_session()

    SavedParameters.append(val)

    global best_val_acc
    global best_test_acc
    global best_val_loss

    if Mode == "MultiBAttention":
        SavedParameters[-1].update(
            {"precision_macro_t": precision_macro_t, "recall_macro_t": recall_macro_t, "fscore_macro_t": fscore_macro_t,
             "precision_micro_t": precision_micro_t, "recall_micro_t": recall_micro_t, "fscore_micro_t": fscore_micro_t,
             "precision_weighted_t": precision_weighted_t, "recall_weighted_t": recall_weighted_t,
             "fscore_weighted_t": fscore_weighted_t,
             "accuracy_t": accuracy_t,
             "balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, Y_predicted) * 100, "TP_test": cf[0][0],
             "FN_test": cf[0][1], "FP_test": cf[1][0], "TN_test": cf[1][1],
             "time_training": time_training, "time_predict": time_predict, "kernel": params["kernel"],
             "learning_rate": params["learning_rate"], "batch": params["batch"], "dropout1": params["dropout1"],
              "filter1": params["filter1"], "weight_categ": params["w"], "unit1": params["unit1"],"unit2": params["unit2"]})

        SavedParameters[-1].update({
            "OA_test": r[0],
            "P_test": r[2],
            "R_test": r[3],
            "F1_test": r[4],
            "FAR_test": r[5],
            "TPR_test": r[6]})

    if SavedParameters[-1]["val_loss"] < best_val_loss:
        print("new saved model:" + str(SavedParameters[-1]))
        model.save(Name.replace(".csv", "_model.h5"))
        model.save(Name.replace(".csv", "_model.tf"))
        model.save_weights(Name.replace(".csv", "_weights.h5"))
        best_val_loss = SavedParameters[-1]["val_loss"]


    SavedParameters = sorted(SavedParameters, key=lambda i: i['val_loss'], reverse=False)


    try:
        with open(Name+'Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")


    # return {'loss': -val["fscore_weighted_val"], 'status': STATUS_OK}
    return {'loss': val["val_loss"], 'status': STATUS_OK}


def MultiBAttention(images, y, y_b, params):
    print(params)

    x_train, x_val, y_train, y_val, y_b_train, y_b_val  = train_test_split(images,
                                                                              y, y_b,
                                                                              test_size=0.2,
                                                                              stratify=y,
                                                                              random_state=100)
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    #print(x_train.shape)
    print(x_val.shape)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_val = np.reshape(x_val, [-1, image_size, image_size, 1])


    kernel = params["kernel"]
    k2 = kernel


    inputs = Input(shape=(image_size, x_train.shape[2], 1))

    X = Conv2D(params['filter1'], (k2, kernel), activation='relu', name='conv0', kernel_initializer='glorot_uniform')(inputs)
    X = Dropout(rate=params['dropout1'])(X)
    if paramsGlobal['attentionLayer']:
        X = PixelAttention2D(X.shape[-1])(X)
        X = Lambda(lambda x: expand_dims(reduce_mean(x, axis=-1), -1))(X)

    Z = Flatten()(X)
    Z = Dense(params['unit1'], activation='relu', kernel_initializer='glorot_uniform')(Z)
    mc_output = Dense(paramsGlobal["classes"], activation='softmax', kernel_initializer='glorot_uniform', name='mc_output')(Z)

    # b prediction
    Y = Flatten()(X)
    Y = Dense(params['unit2'], activation='relu', kernel_initializer='glorot_uniform')(Y)
    b_output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='b_output')(Y)

    model = Model(inputs=inputs, outputs=[mc_output, b_output])

    adam = Adam(params["learning_rate"])

    model.compile(optimizer=adam,
                  loss={'mc_output': 'categorical_crossentropy', 'b_output': 'binary_crossentropy'},
                  # metrics={'mc_output': 'acc', 'b_output': 'acc'}
                  loss_weights=[params['w'], 1-params['w']])

    model.summary()

    
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




    # Train the model.

    hist = model.fit(
        x_train,
        y={"mc_output": y_train, "b_output": y_b_train},
        epochs=params["epoch"],
        verbose=2,
        validation_data=(x_val, {"mc_output": y_val, "b_output": y_b_val}),
        batch_size=params["batch"],
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=1),
                   ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    )
    model.load_weights('best_model.h5')

    y_test = np.argmax(y_val, axis=1)


    (Y_predicted, Y_b_predicted) = model.predict(x_val, verbose=0, use_multiprocessing=True, workers=12)

    Y_predicted = np.argmax(Y_predicted, axis=1)

    Y_b_predicted = ((Y_b_predicted > 0.5) + 0).ravel()

    cf = confusion_matrix(y_b_val, Y_b_predicted)

    cm_val = [[cf[0][0], cf[0][1]], [cf[1][0], cf[1][1]]]
    r = Utils.res(cm_val)

    precision_macro_val, recall_macro_val, fscore_macro_val, support = precision_recall_fscore_support(y_test, Y_predicted, average='macro')
    precision_micro_val, recall_micro_val, fscore_micro_val, support = precision_recall_fscore_support(y_test, Y_predicted, average='micro')
    precision_weighted_val, recall_weighted_val, fscore_weighted_val, support = precision_recall_fscore_support(y_test, Y_predicted, average='weighted')
    accuracy_val = accuracy_score(y_test, Y_predicted)

    del support
    epoches=len(hist.history['val_loss'])
    min_val_loss = np.amin(hist.history['val_loss'])



    return model, {"val_loss": min_val_loss, "precision_macro_val": precision_macro_val, "recall_macro_val": recall_macro_val, "fscore_macro_val": fscore_macro_val,
                   "precision_micro_val": precision_micro_val, "recall_micro_val": recall_micro_val, "fscore_micro_val": fscore_micro_val,
                   "precision_weighted_val": precision_weighted_val, "recall_weighted_val": recall_weighted_val, "fscore_weighted_val": fscore_weighted_val,
                   "accuracy_val": accuracy_val, "balanced_accuracy_val": balanced_accuracy_score(y_test, Y_predicted) * 100, "TP_val": cf[0][0],
                   "FN_val": cf[0][1], "FP_val": cf[1][0], "TN_val": cf[1][1], "OA_val": r[0], "P_val": r[2], "R_val": r[3], "F1_val": r[4], "FAR_val": r[5],"TPR_val": r[6], "epochs":epoches}
