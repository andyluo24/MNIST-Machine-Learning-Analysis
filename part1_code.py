import numpy as np
import csv
from matplotlib import pyplot as plt
import math
import statistics as st
from sklearn.ensemble import RandomForestclassifier
from mnist.loader import MNIST

def bb(input_image, width=28, height=28):
    input_image = np.reshape(input_image, (height, width))
    nonzero = np.where(input_image != 0)
    w = np.min(nonzero[0])
    e = np.max(nonzero[0])
    u = np.min(nonzero[1])
    d = np.max(nonzero[1])
    return input_image[w:e+1, u:d+1]

def resize_bb(input_image, width=28, height=28):
    oriheight, oriwidth = input_image.shape
    new = np.zeros((height, width))
    w_ratio = float(oriwidth) / width
    h_ratio = float(oriheight) / height
    for i in range(0, height):
      for j in range(0, width):
        new[i, j] = input_image[int(math.floor(i * h_ratio)), int(math.floor(j * w_ratio))]
    return new.astype(int).reshape(width*height)

def randomforestcreator_raw(feature, labels, trees_num, tree_depth):
    rf = RandomForestClassifier(n_estimators=trees_num, max_depth=tree_depth)
    return rf.fit(feature, labels)

def randomforestcreator_stretched(feature, labels, trees_num, tree_depth):
    rf = RandomForestClassifier(n_estimators=trees_num, max_depth=tree_depth)
    feature = sbb(feature)
    return rf.fit(feature, labels)

def sbb(images_pool):
    return np.array([resize_bb(bb(img)) for img in images_pool])

def predict_raw(train, test):
    return train.predict(test)

def predict_stretched(train, test):
    test = sbb(test)
    return train.predict(test)

def evaluate(prediction, label):
    outcome = prediction == label
    print(outcome)
    return sum(outcome) / float(len(prediction))

mndata = MNIST('/content/drive/My Drive/cs498aml/mnist')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()
model_raw = randomforestcreator_raw(images_train, labels_train,30,16)
model_stretched = randomforestcreator_stretched(images_train, labels_train,30,16)
pred_raw = predict_raw(model_raw, images_test)
pred_stretched = predict_stretched(model_stretched, images_test)
print(evaluate(pred_raw, labels_test))
print(evaluate(pred_stretched, labels_test))
