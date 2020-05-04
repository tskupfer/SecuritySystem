import numpy as np
import imutils
import cv2

class MotionDetector:
    def __init__(self, weightedAvg=0.5):
        #If weightedAvg is larger, the foreground is considered more than the background
        #If weightedAvg is smaller, the background is considered more than the foreground
        #The default weightedAvg (0.5) gives equal weight to the foreground and background
        self.weightedAvg = weightedAvg
        self.backgroundModel = None

    def updateBackgroundModel(self, frame):
        #Initialize backgroundModel
        if self.backgroundModel is None:
            self.backgroundModel = frame.copy().astype("float")
            return

        #If backgroundModel is initialized update it using the
        #input frame, current backgroundModel, and weightedAvg
        cv2.accumulateWeighted(frame, self.backgroundModel, self.weightedAvg)

    def detect(self, frame, thresholdVal=25):
        #Threshold (partition image as foreground and background) the delta image
        #which is the absolute difference between the backgroundModel and input frame
        threshold = cv2.threshold(
                cv2.absdiff(self.backgroundModel.astype("uint8"), frame), #Delta image
                thresholdVal, #Value that determines if a pixel is motion or not
                255, #Any pixel with a delta image > thresholdVal is set to 255 (white foreground)
                     #otherwise it is set to 0 (black background)
                cv2.THRESH_BINARY
        )[1]

        #Carry out erosions and dilations to remove abnormalities (noise, false-positive motion spots)
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)

        #Detect contours to locate regions in the frame with motion
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        #Initialize tuples using +infinity and -infinity to store where the perimeter of motion is
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        #return no contours are found there is no motion so return None
        if len(contours) == 0:
            return None

        for contour in contours:
            #compute the bounding rectangle of the given contour
            (x, y, width, height) = cv2.boundingRect(contour)

            #update the minimum and maximum bounding region (perimeter of motion)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x+width), max(maxY, y+height))

        #return threshold tuple and bounding perimeter tuple
        return (threshold, (minX, minY, maxX, maxY))
