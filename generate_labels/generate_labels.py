# This python file is to train steering wheel video in a neural network and generate labels for custom data
import cv2
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.models import Sequential
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
    return model

DATA_FOLDER = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'steering_wheel.mp4')
LABEL_FILE = os.path.join(DATA_FOLDER, 'train_labels.txt')
TEST_FILE = os.path.join(DATA_FOLDER, 'steering_test_resized.mp4')
OUTPUT_FILE = os.path.join(DATA_FOLDER, 'out_labels.txt')
LABEL_MODEL = os.path.join(DATA_FOLDER, 'label_model')

def return_data():
    y = []
    features = []
    cap = cv2.VideoCapture(TRAIN_FILE)
    ret, frame = cap.read()
    with open(LABEL_FILE) as fp:
        for line in fp:
            path, angle = line.strip().split()
            y.append(float(angle) * scipy.pi / 180)
            img = plt.imread(frame)
            features.append(cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100)))
            ret, frame = cap.read()
    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')
    cap.release()
    return features, labels

def main():
    if not os.path.exists(LABEL_MODEL):    
        features, labels = return_data()
        features, labels = shuffle(features, labels)
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3)
        train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
        test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
        model = keras_model(100, 100)
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=128, batch_size=32)
        model.save(LABEL_MODEL)
    else:
        model = keras.models.load_model(LABEL_MODEL)

    label_pred = []
    cap = cv2.VideoCapture(TEST_FILE)
    ret, frame = cap.read()
    while True:
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
        img = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[:, :, 1], (100, 100))
        img = img.reshape(1, 100, 100, 1)
        x = model.predict(img).item() * (180 / scipy.pi)
        label_pred.append(x)
        #cv2.imshow(str(x), frame)
        #cv2.waitKey(1000)
        
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    cv2.destroyAllWindows()
    with open(OUTPUT_FILE, 'a') as file:
        for item in label_pred:
            file.write(str(item) + '\n')
        file.flush()
main()
