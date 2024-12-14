# Self-Driving Car on Village Roads

## Description
This project demonstrates a self-driving car's ability to predict steering angles based on road images using a convolutional neural network (CNN) and focuses on navigating village roads to create a robust autopilot system. It includes data loading scripts, a CNN model for training, and a test script for real-time self-driving simulation, where the trained model predicts steering angles from input images, and the simulation visually represents the predicted steering angle on a virtual steering wheel.

## Features
- **Data Preprocessing:** Extracts and resizes features from road images for training.
- **Convolutional Neural Network (CNN):** Predicts steering angles using preprocessed images.
- **Real-Time Steering Simulation:** Simulates steering wheel behavior with video playback.
- **Video Cropping Utility:** Extracts relevant sections of videos for further processing.

## Utility Functions
- **crop_steering_from_video.py:** Crops a specified region from a video and saves the cropped version as a new video file.
- **generate_labels.py:** Trains a neural network to predict steering angles from video frames, generates labels for test videos, and saves them to a file.
- **manual_labelling.py:** Overlays a semi-transparent protractor image on video frames to assist in manually capturing steering angles.
- **steering_wheel_to_video.py:** Creates a video of a rotating steering wheel based on smoothed steering angle data from a label file.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** 
  - OpenCV
  - NumPy
  - Matplotlib
  - Keras
  - SciPy
  - scikit-learn
- **Tools:** 
  - Model checkpointing
  - Pickle for data serialization

## Data
The dataset consists of:
- **Images:** Frames extracted from video footage of village roads.
- **Labels:** Steering angles corresponding to the road images (converted from degrees to radians).
- **Image Features:** Preprocessed road images resized to 100x100 and converted to HSV format.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnoopCA/Self_Driving_Car_Village_Roads.git
   cd self-driving-car-village-roads
   ```

## Inspiration
This project draws inspiration from Sully Chen's work in the field of self-driving cars.

### Credits
- Sully Chen: [https://github.com/SullyChen/driving-datasets](#)

