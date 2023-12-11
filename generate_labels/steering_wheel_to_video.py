import cv2
import numpy as np
from itertools import islice

steer = cv2.imread(r'D:\ML_Projects\Self_Driving_Car_Village_Roads\steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0
output_video_path = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\steering_wheel.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (cols, rows))
label_file_path = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\train_labels.txt'

with open(label_file_path) as label_file:
    steering_angles = [float(line.strip().split()[1]) for line in label_file]

for steering_angle in steering_angles:
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (steering_angle - smoothed_angle) / (abs(steering_angle - smoothed_angle) + 1e-6)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    output_frame = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    output_video.write(output_frame)

output_video.release()
