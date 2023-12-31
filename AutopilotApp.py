#This section of code is to show a demo of the self-driving car steering wheel simulation
import numpy as np
import cv2
from keras.models import load_model

model = load_model(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\models\Autopilot.h5')

def keras_predict(model, image):
    image = np.array(image, dtype=np.float32)
    image = np.reshape(image, (-1, 100, 100, 1))
    steering_angle = float(model.predict(image, batch_size=1))
    steering_angle = steering_angle * 60
    return steering_angle

steer = cv2.imread(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0
#This is a sample video to test the prediction of driving angles
cap = cv2.VideoCapture(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\test\forest.mp4')  #CarTop, forest, HighSpeed, my_forest

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 1], (100, 100))
    steering_angle = keras_predict(model, gray)

    frame_display = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA)
    smoothed_angle += 0.2 * pow(abs(steering_angle - smoothed_angle), 2.0 / 3.0) * (steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    rotated_steering_wheel = cv2.warpAffine(steer, M, (cols, rows))
    rotated_steering_wheel = cv2.cvtColor(rotated_steering_wheel, cv2.COLOR_GRAY2BGR)
    
    combined_image = np.zeros((frame_display.shape[0] + rows, frame_display.shape[1], 3), dtype=np.uint8)
    combined_image[:frame_display.shape[0], :] = frame_display
    combined_image[frame_display.shape[0]:, (combined_image.shape[1] - cols) // 2:(combined_image.shape[1] + cols) // 2] = rotated_steering_wheel

    cv2.imshow("Self Driving", combined_image)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
