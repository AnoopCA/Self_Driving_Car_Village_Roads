import cv2
import numpy as np

def overlay_image(background, overlay, alpha, position):
    y, x = position
    h, w = overlay.shape[:2]
    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0

    if background.shape[2] == 3:
        background = np.concatenate([background, np.ones((background.shape[0], background.shape[1], 1), dtype=np.uint8) * 255], axis=2)
    # Blend the overlay and background
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay_rgb[:, :, c]
    # Blend the alpha channel
    background[y:y+h, x:x+w, 3] = (1 - alpha) * background[y:y+h, x:x+w, 3] + alpha * overlay_alpha
    return background

video_path = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\steering_test_resized.mp4'
cap = cv2.VideoCapture(video_path)
protractor_path = r'D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\Protractor_360.png'
overlay = cv2.imread(protractor_path, cv2.IMREAD_UNCHANGED)
overlay_height, overlay_width = overlay.shape[:2]
scale_factor = 0.75
transparency = 0.25
new_overlay_height = int(overlay_height * scale_factor)
new_overlay_width = int(overlay_width * scale_factor)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the overlay image
    overlay_resized = cv2.resize(overlay, (new_overlay_width, new_overlay_height))
    # Calculate the position to center the overlay on the frame
    y_offset = (frame.shape[0] - new_overlay_height) // 2
    x_offset = (frame.shape[1] - new_overlay_width) // 2
    # Overlay the frame with the resized overlay
    result_frame = overlay_image(frame, overlay_resized, transparency, (y_offset, x_offset))
    cv2.imshow('Capture labels', result_frame)
    cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
