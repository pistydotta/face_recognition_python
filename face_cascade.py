from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    eyes_qnt = 0
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        eyes_qnt += len(eyes)
    return faces, eyes_qnt
    # cv.imshow('Capture - Face detection', frame)



face_cascade_name = "/home/dotta/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "/home/dotta/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)     
img = cv.imread("faces.jpg")
#-- 2. Read the video stream
faces, eyes = detectAndDisplay(img)
print(len(faces))
print(eyes)
