import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import tensorflow as tf
from yolo_manager import yolov3


if __name__ == "__main__":
    # Load yolo model
    yolo = yolov3.utils.load_yolo_model()

    # Load env
    env = load_env()

    while True:
        # Step the environment
        image = env.step()
        # original_image = cv2.imread(image_path)

        # Get image from environment, and send to yolo
        detections = yolov3.utils.detect_image_live(yolo, image, input_size=416, CLASSES=yolov3.configs.YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='')

        # Process the detections into the correct state format
        # TODO: Implement

        # Send the state to the RL agent
        # TODO: Implement



