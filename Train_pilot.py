# Convolutional Neural Network model for self-driving car training using Keras.
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
import cv2
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def keras_model(image_x, image_y):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(image_x, image_y, 1)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss="mse")
    filepath = r"D:\ML_Projects\Self_Driving_Car_Village_Roads\models\Autopilot.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    return model, callbacks_list

def load_data(in_format='img'):
    if in_format == 'img':
        with open(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\features', 'rb') as f:
            features = np.array(pickle.load(f))
        with open(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\labels', 'rb') as f:
            labels = np.array(pickle.load(f))
    else:
        feature = []
        labels = []
        cap = cv2.VideoCapture(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\train\train_i10.mp4')
        ret, frame = cap.read()
        while True:
            img = plt.imread(frame)
            feature.append(cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100)))
            ret, frame = cap.read()
            if not ret:
                break
        features = np.array(feature).astype('float32')
        with open(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\out_labels.txt', 'rb') as f:
            for line in f:
                labels.append(float(line) * scipy.pi / 180)
        labels = np.array(labels).astype('float32')
    return features, labels

def main():
    features, labels = load_data('img') #vid
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3)
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    model, callbacks_list = keras_model(100, 100)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=128, batch_size=32, callbacks=callbacks_list)
    print(model)
    model.save(r"D:\ML_Projects\Self_Driving_Car_Village_Roads\models\Autopilot.h5")
    K.clear_session()

main()
