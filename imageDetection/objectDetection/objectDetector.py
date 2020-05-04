import numpy as np
import imutils
import cv2

class ObjectDetector:
    def __init__(self, classifier):
        self.classifier = classifier

    def detect(self, frame):
        grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = self.classifier.detectMultiScale(
            grayscaleFrame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return objects
