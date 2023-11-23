import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning/Exercise 5/cifar-10-batches-py/test_batch')
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

print(X.shape)

# labeldict = unpickle('/home/kamarain/Data/cifar-10-batches-py/batches.meta')
labeldict = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning/Exercise 5/cifar-10-batches-py/batches.meta')

# change from label_names to batch_label
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1)
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)
