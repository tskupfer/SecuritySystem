from imageDetection.motionDetection.motionDetector import MotionDetector
from imageDetection.objectDetection.objectDetector import ObjectDetector
from imutils.video import VideoStream
from flask import Response, Flask, render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

#Initialize outputFrame and a lock variable to ensure thread-safe exchanges of
#the output frames (necessary when multiple clients accessing the stream)
outputFrame = None
threadLock = threading.Lock()
i=1

#Initialize flask object
app=Flask(__name__)

#Initialize videoStream and give the camera module time to activate
#videoStream = VideoStream(usePiCamera=1).start()
videoStream = VideoStream(src=0).start()
time.sleep(2.0)

@app.route('/')
def index():
    #Render index.html with flask
    return render_template("index.html")

def detectMotion(minFrameCount):
    #Using global reference to variables to ensure concurrency is supported
    global videoStream, outputFrame, threadLock

    motionDetector = MotionDetector(weightedAvg=0.6)
    totalFramesRead = 0

    while True:
        #Read in next frame, resize it for faster computation (less data),
        #convert to grayscale, and apply Gaussian blurring to reduce noise
        frame = videoStream.read()
        frame = imutils.resize(frame, width=400)
        grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscaleFrame = cv2.GaussianBlur(grayscaleFrame, (7, 7), 0)

        #Add the timestamp to the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
        )

        if totalFramesRead > minFrameCount:
            motion = motionDetector.detect(grayscaleFrame)
            if motion is not None:
                #Extract motion data tuples and draw the
                #rectangle around the motion in the RGB frame
                (threshold, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

        motionDetector.updateBackgroundModel(grayscaleFrame)
        totalFramesRead+=1

        #Get the thread lock to make sure outputFrame isn't
        #being read by another client while trying to update it
        with threadLock:
            outputFrame = frame.copy()

def detectObject():
    #Using global reference to variables to ensure concurrency is supported
    global videoStream, outputFrame, threadLock

    objectDetector = ObjectDetector(cv2.CascadeClassifier("imageDetection/objectDetection/models/facial_recognition_model.xml"))

    while True:
        #Read in next frame, resize it for faster computation (less data),
        #convert to grayscale, and apply Gaussian blurring to reduce noise
        frame = videoStream.read()
        frame = imutils.resize(frame, width=400)

        #Add the timestamp to the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
        )

        objects = objectDetector.detect(frame)

        if len(objects) > 0:
            for (x, y, width, height) in objects:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

        with threadLock:
            outputFrame = frame.copy()

#encodes outputFrame as JPEG
def generateJPEG():
    global outputFrame, threadLock

    while True:
        #Encodes each output frame in a thread-safe environment
        with threadLock:
            #Hardware fault
            if outputFrame is None:
                continue

            #Use JPEG compression to speed up transmission of frames and reduce network load
            (successful, encodedImage) = cv2.imencode('.jpg', outputFrame)

            #Compression failed
            if not successful:
                continue
            #Yield outputFrame in the byte format so web browser can interpret the frame
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video-feed')
def videoFeed():
    return Response(generateJPEG(), mimetype='multipart/x-mixed-replace; boundary=frame')

#main execution thread
if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--ip', type=str, required=True, help='device\'s ip address')
    argParser.add_argument('-o', '--port', type=int, required=True, help='temporary server port (1024-65535)')
    argParser.add_argument('-f', '--frames', type=int, default=32, help='minimum frames needed to build the background model')

    args = vars(argParser.parse_args())

    #motionDetectionThread = threading.Thread(target=detectMotion, args=(args['frames'],))
    #motionDetectionThread.daemon = True
    #motionDetectionThread.start()

    objectDetectionThread = threading.Thread(target=detectObject, args=())
    objectDetectionThread.daemon = True
    objectDetectionThread.start()

    app.run(host=args['ip'], port=args['port'], debug=False, threaded=True, use_reloader=False)

videoStream.stop()
