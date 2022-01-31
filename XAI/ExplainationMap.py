from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import cv2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from PIL import Image, ImageOps


def postprocess_activations(output, size1, size2):
    # using the approach in https://arxiv.org/abs/1612.03928
    # output = np.abs(activations)
    # output = np.sum(output, axis = -1).squeeze()

    # resize and convert to image
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    output = cv2.resize(output, (size1, size2))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    scaler = MinMaxScaler()
    output = scaler.fit_transform(output)
    output *= 255
    return output.astype('uint8')
    # return output


def create_heatmap(x, dir, model):
    predict_map = model.predict(x)
    print(predict_map.shape)
    f_myfile = open(dir, 'wb')
    s1, s2, s3, s4 = x.shape
    heatmap = np.empty((s1, s2, s3, 3))
    for (i, image) in enumerate(predict_map):
        img = postprocess_activations(image, s2, s3)
        heatmap[i] = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    print(heatmap.shape)
    pickle.dump(heatmap, f_myfile)
    print(predict_map.shape)
    return heatmap


def createMapAttention(model, images, param, method):
    """
        Function saving an visualizing the heatmpa images of training and testing sets.
        :param model: the learned model with Attention layer
        :param images: dictionary containing training and testing set
        :param param: dictionary containing the parameters of the execution
        :param method: MI o Mean
        :return: a 3-dim numpy array containing the image
        """
    feature_map_model = Model(inputs=model.input, outputs=model.get_layer("lambda").output)
    feature_map_model.summary()
    train = images["Xtrain"]
    test = images["Xtest"]
    test = np.array(test)
    train = np.array(train)

    print(train.shape)


    print(test.shape)
    image_size1, image_size2 = test[0].shape
    train = np.reshape(train, [-1, image_size1, image_size2, 1])
    test = np.reshape(test, [-1, image_size1, image_size2, 1])
    print(test.shape)
    heatmapTest = create_heatmap(test, param["dir"] + str(param['Max_A_Size']) + 'x' + str(
        param['Max_B_Size']) + '_'+ method+ 'Xtest_AttentionMap.pickle', feature_map_model)
    heatmapTrain = create_heatmap(train, param["dir"] + str(param['Max_A_Size']) + 'x' + str(
        param['Max_B_Size']) + '_' + method+ '_Xtrain_AttentionMap.pickle', feature_map_model)

    #example of visualization

    t = heatmapTrain[0, :, :, :]

    plt.imshow(t.astype('uint8'))
    plt.show()
    plt.imshow(heatmapTest[0, :, :, :].astype('uint8'))
    plt.show()





