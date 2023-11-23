import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras
import pickle
import matplotlib.pyplot as plt
import numpy as np

"""
I conducted experiments to fine-tune the neural network's performance. 
Firstly, I expanded the network architecture by introducing additional layers, 
beginning with 1024 neurons and progressively reducing the number of neurons 
while using the 'relu' activation function. This helped in creating a more complex model.

Furthermore, I adjusted the learning rate during training. 
Initially, I started with a high learning rate of 0.5, following a suggestion from the TA. 
However, I noticed that this learning rate led to overshooting, causing a deterioration in accuracy
after a few epochs. To mitigate this issue, I gradually decreased the learning rate 
until it reached 0.001, which resulted in achieving the desired training outcome
"""


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    features = dict["data"]
    labels = dict["labels"]
    return features, labels


num_classes = 10
batch_1, labels_1 = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                      'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning'
                      '/Exercise 5/cifar-10-batches-py/data_batch_1')
batch_2, labels_2 = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                      'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning'
                      '/Exercise 5/cifar-10-batches-py/data_batch_2')
batch_3, labels_3 = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                      'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning'
                      '/Exercise 5/cifar-10-batches-py/data_batch_3')
batch_4, labels_4 = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                      'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning'
                      '/Exercise 5/cifar-10-batches-py/data_batch_4')
batch_5, labels_5 = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                      'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning'
                      '/Exercise 5/cifar-10-batches-py/data_batch_5')

# Merge files
x_train = np.concatenate([batch_1, batch_2, batch_3, batch_4, batch_5], 0)
y_train = np.concatenate([labels_1, labels_2, labels_3, labels_4, labels_5], 0)

x_test, y_test = unpickle('C:/Users/Phan Phuong Uyen/OneDrive/Documents/University_TUNI/'
                     'Academic year 2023-2024/DATA.ML.100-2023-2024 Introduction to Pattern Recognition and Machine Learning/'
                     'Exercise 5/cifar-10-batches-py/test_batch')


# Normalize the data to get value range from 0 to 1
x_train = x_train/255
x_test = x_test/255

# Convert class labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)


# Create a simple feedforward neural network
model = Sequential()

# Additional hidden layers
model.add(Dense(1024, input_dim=3072, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Final layer
model.add(Dense(num_classes, activation='softmax'))

# Add learning rate
custom_learning_rate = 0.001
custom_optimizer = Adam(learning_rate=custom_learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=custom_optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=6, batch_size=64)

# Plot the training loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

