from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

def res(cm):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return r

def prediction_binary(y_test, y_pred):
    cf = confusion_matrix(y_test, y_pred)
    return res(cf)


def printPrediction(model, images, path, type):
    """
        Function calculating the results
        :param model: learned model
        :param images: dictionary containing the training and testing sets
        :param path: path to save the results
        :param type: multi-output or singel output
        """
    x_test=images["Xtest"]
    y_test_mc = images["Ytest"]
    y_test_b=images["b_Ytest"]
    x_train = np.array(x_test)

    image_size = x_train.shape[1]
    x_test = np.reshape(x_train, [-1, image_size, image_size, 1])
    print(x_test.shape)

    if type=="MultiBAttention":
        (y_predicted, y_b_predicted) = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)
        y_predicted = np.argmax(y_predicted, axis=1)
        y_b_predicted = ((y_b_predicted > 0.5) + 0).ravel()
        print_results(type, path,y_test_mc, y_predicted, y_test_b, y_b_predicted, model)
    elif type=="MultiAttention":
        y_predicted = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)
        y_predicted = np.argmax(y_predicted, axis=1)
        print_Singleresults(type, path, y_test_mc, y_predicted, model)



def print_results(type, path, y_true, y_pred, y_test_b, y_pred_b, model, write=True):
    """
    FUnction saving the classification report in  .txt format
    :param path: path to save the report
    :param y_true: true y
    :param y_pred: vpredicted y
    :param write: booleano, if true write the file, otherwise return only the classification report
    :return: classification report if write equal to false otherwise void
    """
    from sklearn.metrics import confusion_matrix, f1_score, classification_report
    cm = confusion_matrix(y_true, y_pred)
    cf_b= confusion_matrix(y_test_b, y_pred_b)
    res_b=res(cf_b)

    val = ''
    val = val + ('\n****** ' + type + ' ******\n\n')
    val = val + (classification_report(y_true, y_pred))
    val = val + '\n\n----------- f1 macro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='macro'))
    val = val + '\n\n----------- f1 micro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='micro'))
    val = val + '\n\n----------- f1 weighted ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='weighted'))
    val = val + '\n\n----------- Confusion matrix ---------------\n'
    val = val + str(cm)
    val = val + '\n\n----------- Binary prediction ---------------\n'
    val = val + str(cf_b)
    val = val + '\n\n----------- f1  binary---------------\n'
    val = val+ str(res_b[4])
    val = val + '\n\n----------- oa  binary---------------\n'
    val = val+ str(res_b[0])


    if write:
        with open(path + '_results.txt', 'w', encoding='utf-16') as file:
            file.write(val)
            file.write('\n\n----------- Summary ---------------\n')
            model.summary(print_fn=lambda x: file.write(x + '\n'))


    else:
        return val

def print_Singleresults(type, path, y_true, y_pred, model, write=True):
    """
      FUnction saving the classification report of the single outputs classifiers in  .txt format
      :param path: path to save the report
      :param y_true: true y
      :param y_pred: vpredicted y
      :param write: booleano, if true write the file, otherwise return only the classification report
      :return: classification report if write equal to false otherwise void
      """
    from sklearn.metrics import confusion_matrix, f1_score, classification_report
    cm = confusion_matrix(y_true, y_pred)


    val = ''
    val = val + ('\n****** ' + type + ' ******\n\n')
    val = val + (classification_report(y_true, y_pred))
    val = val + '\n\n----------- f1 macro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='macro'))
    val = val + '\n\n----------- f1 micro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='micro'))
    val = val + '\n\n----------- f1 weighted ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='weighted'))
    val = val + '\n\n----------- Confusion matrix ---------------\n'
    val = val + str(cm)



    if write:
        with open(path + '_results.txt', 'w', encoding='utf-16') as file:
            file.write(val)
            file.write('\n\n----------- Summary ---------------\n')
            model.summary(print_fn=lambda x: file.write(x + '\n'))


    else:
        return val
