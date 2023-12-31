"""
Load and preprocess training data for the Self-Driving Car model from images and corresponding steering angles.
"""
import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

DATA_FOLDER = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\train'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

def return_data():
    X = []
    y = []
    features = []
    with open(TRAIN_FILE) as fp:
        for line in fp:
            path, angle = line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            # convert angle from degree to radian
            y.append(float(angle) * scipy.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1], (100, 100)))

    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)

return_data()
