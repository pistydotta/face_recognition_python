from __future__ import print_function
from lib2to3.pgen2 import grammar
import os
import sys
import cv2 as cv
import time
import boto3
from numpy import eye
import threading

def loadCascadeXmls(face_cascade_path, eyes_cascade_path):
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    if not face_cascade.load(cv.samples.findFile(face_cascade_path)) or not eyes_cascade.load(cv.samples.findFile(eyes_cascade_path)):
        print('--(!)Error loading face or eye cascade')
        exit(0)   
    return face_cascade, eyes_cascade

def getTargetImages(images_path):
    image_qnt = sys.argv[1]
    images = os.listdir(images_path)
    images.sort()
    workingImages = images[0:int(image_qnt)]
    return workingImages

def saveResultsToFile(result_list):
    if (sys.argv[2] == "single"):
        for result in result_list:
            fileName = result.split('\n')[0]
            with open(f"./results/{fileName}.txt", 'w') as f:
                f.write(result)
    else:
        fileName = "results_imagem.txt"
        with open(f"./results/{fileName}", 'w') as f:
            for result in result_list:
                f.write(result)
    

def uploadResultToAws(s3_path):
    s3 = boto3.resource('s3')
    bucket_name = "opencv-pistydotta"
    files = os.listdir('./results')
    for file in files:
        s3.Bucket(bucket_name).upload_file("./results/" + file, f"{s3_path}{file}")

def detectAndDisplay(image_list, images_path, i, j):
    result_list = []
    for i in range(j):
        start = time.time()
        img = cv.imread(f"{images_path}{image_list[i]}")
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.equalizeHist(gray_img)
        faces = face_cascade.detectMultiScale(gray_img)
        eyes_qnt = 0
        for (x,y,w,h) in faces:
            faceROI = gray_img[y:y+h,x:x+w]
            eyes = eyes_cascade.detectMultiScale(faceROI)
            eyes_qnt += len(eyes)
        end = time.time()
        result_list.append(image_list[i] + "\n" + str(start) + "\n" + str(end) + "\n" + str(end-start) + "\n")
    return result_list
    thread.start_new_thread(detectAndDisplay, (image_list, images_path, 0, 4))
def cleanUpResults():
    folder = "./results"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

        
#To run the application use: python3 face-cascade.py images_qnt upload_mode
# upload_mode: (single = results have their own file, else, all results go in just one file)
#If you wouldn't like the results folder to get deleted, comment the function cleanUpResults()
timestamp1 = time.time()
face_cascade_path = "./utils/haarcascade_frontalface_alt.xml"
eyes_cascade_path = "./utils/haarcascade_eye_tree_eyeglasses.xml"
face_cascade, eyes_cascade = loadCascadeXmls(face_cascade_path, eyes_cascade_path)
images_path = os.path.expanduser("~/dev/images/")
image_list = getTargetImages(images_path)
print(image_list)
#exit(0)
#t1 = threading.Thread(target=detectAndDisplay, args=(image_list, images_path, 0, 5))
#t2 = threading.Thread(target=detectAndDisplay, args=(image_list, images_path, 6, 10))
#t3 = threading.Thread(target=detectAndDisplay, args=(image_list, images_path, 11, 15))
#t4 = threading.Thread(target=detectAndDisplay, args=(image_list, images_path, 16, 20))
#t1.start()
#t2.start()
#t3.start()
#t4.start()
#t1.join()
#t2.join()
#t3.join()
#t4.join()
#print(t1)
#print(t2)
result_list = detectAndDisplay(image_list, images_path, 0, 20)
timestamp2 = time.time()
#saveResultsToFile(result_list)
timestamp3 = time.time()
s3_path = "results/local/11/"
#uploadResultToAws(s3_path)
timestamp4 = time.time()
#cleanUpResults()
# print("The time of execution of above program is :", end-start)
print("Images were processed in: " + str(timestamp2 - timestamp1))
print("Time it took to send results to amazon: " + str(timestamp4 - timestamp3))
