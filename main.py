import cv2
import torch
from time import time


# Class to take a webcam input and produce an output marked up by objects detected by the yolov5 model
class RealTimeCameraObjectDetection:

    # Initialization method
    def __init__(self):
        # Get video stream from camera 0 (webcam)
        self.stream = cv2.VideoCapture(0)
        # Load yolov5 model from pytorch cache
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        # If GPU is available use it's 'cuda' cores. If GPU is not available instead use CPU for decreased performance.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Set minimum certainty that an object is identified
        self.min_weight = 0.2
        # Set color of detection features in (R, G, B) format
        self.detection_color = (0, 255, 0)
        # Set thickness of detection lines
        self.detection_line_thickness = 2
        # Set scaling factor of font for objects detected text
        self.font_scale = 0.9

    # Method that takes a single frame, runs it through the model and returns the labels and coordinates of the
    # objects detected.
    def score_frame(self, frame):
        # "Move" model to chosen device (cuda or cpu)
        self.model.to(self.device)
        # Puts np array into python list (I do not like this syntax because it is impossible to google)
        frame = [frame]
        # Results are received from frame run through model
        results = self.model(frame)
        # Separate labels and coords from the rest of results
        labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, coords

    # Method to draw the boxes and text which describe detected objects onto the frames which will be
    # outputted as video.
    def plot_boxes(self, results, frame):
        # From results get:
        # Labels = array of integers representing different object labels in image
        # Ex: [ apple, banana, orange]
        # coords = coordinates between 0 and 1 that can be multiplied by the frame's size to get object position
        # Ex: [bounding rect left, bounding rect top, bounding rect right, bounding rect bottom, detection certainty]
        labels, coords = results
        # Get dimensions of frame in stream
        stream_x_dimension, stream_y_dimension = frame.shape[1], frame.shape[0]
        # For all objects detected in frame
        for i in range(len(labels)):
            # Separate one object's coords and detection certainty
            row = coords[i]
            # If the model is more than self.min_weight sure about an object detected
            if row[4] >= self.min_weight:
                # Define top left and bottom right of bounding rectangle for drawing using points from coord
                rectangle_top_left = (int(row[0] * stream_x_dimension), int(row[1] * stream_y_dimension))
                rectangle_bottom_right = (int(row[2] * stream_x_dimension), int(row[3] * stream_y_dimension))
                # Draw rectangle using output image, top left and bottom right bounds, color, and line thickness
                cv2.rectangle(
                    frame, rectangle_top_left, rectangle_bottom_right, self.detection_color,
                    self.detection_line_thickness
                )
                # Draw text using output image, bottom left bound, font, font scaling factor, color, and line thickness
                cv2.putText(
                    frame, self.model.names[int(labels[i])], rectangle_top_left, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    self.detection_color, self.detection_line_thickness
                )
        # Return marked up frame
        return frame

    # Call Method
    def __call__(self):
        # Set player as camera input
        player = self.stream
        # Ensure camera is opened
        assert player.isOpened()
        # Main operation loop
        while True:
            # Set time to begin FPS calculation
            start_time = time()
            # Read image from camera, received is to ensure the read was successful, frame is the actual image
            received, frame = player.read()
            # Ensure input is available to be read
            assert received
            # Receive results of scoring from model
            results = self.score_frame(frame)
            # Mark up frame
            frame = self.plot_boxes(results, frame)
            # Set time to end FPS calculation
            end_time = time()
            # Perform FPS calculation
            fps = int(1 / (end_time - start_time))
            # Create variable for FPS counter position
            fps_position = (525, 25) #These may have to be changed based on resolution of output video
            # Insert FPS into output video
            cv2.putText(
                frame, f"FPS: {fps}", fps_position, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    self.detection_color, self.detection_line_thickness
                )
            # Show single frame of output video
            cv2.imshow("frame", frame)
            # If key is not pressed in 1ms (Minimal length) move to next frame
            cv2.waitKey(1)


# Initialize created class
MyObjectDetection = RealTimeCameraObjectDetection()
# Call created class
MyObjectDetection()
