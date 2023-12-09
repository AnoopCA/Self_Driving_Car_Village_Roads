import cv2

def crop_video(input_path, output_path, crop_box):
    video_capture = cv2.VideoCapture(input_path)
    original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    x1, y1, x2, y2 = crop_box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(original_width, x2)
    y2 = min(original_height, y2)
    cropped_width = x2 - x1
    cropped_height = y2 - y1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (cropped_width, cropped_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        cropped_frame = frame[y1:y2, x1:x2]
        out.write(cropped_frame)
    video_capture.release()
    out.release()

input_path = r"D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\steering_test_raw.mp4"
output_path = r"D:\ML_Projects\Self_Driving_Car_Village_Roads\generate_labels\steering_test_resized.mp4"
crop_box = (20, 275, 700, 950)

crop_video(input_path, output_path, crop_box)
