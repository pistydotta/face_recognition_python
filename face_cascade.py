from __future__ import print_function
from lib2to3.pgen2 import grammar
import os
import sys
import cv2 as cv
import time
import boto3

def getTargetImages(images_path):
    image_qnt = sys.argv[1]
    images = os.listdir(images_path)
    images.sort()
    workingImages = images[0:int(image_qnt)]
    return workingImages

def detectAndDisplay(image_list, images_path, s3, bucket_name, s3_path):
    tmp = []
    for image in image_list:
        start = time.time()
        img = cv.imread(f"{images_path}{image}")
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.equalizeHist(gray_img)
        faces = face_cascade.detectMultiScale(gray_img)
        eyes_qnt = 0
        for (x,y,w,h) in faces:
            faceROI = gray_img[y:y+h,x:x+w]
            eyes = eyes_cascade.detectMultiScale(faceROI)
            eyes_qnt += len(eyes)
        end = time.time()
        exec_time = end - start
        # fileName = f"{image}_{end}_.txt"
        # print(fileName)
        # with open(f"./results/{fileName}", 'w') as f:
            # f.write("a")
            # f.write(f"{image} took {str(exec_time)} to execute\nThere was {str(len(faces))} faces and {str(eyes_qnt)} eyes on the image")
        tmp.append(f"{image} took {str(exec_time)} to execute\nThere was {str(len(faces))} faces and {str(eyes_qnt)} eyes on the image\n")
    fileName = "results_imagem.txt"
    with open(f"./results/{fileName}", 'w') as f:
        for text in tmp:
            f.write(text)
    s3.Bucket(bucket_name).upload_file("./results/" + fileName, f"{s3_path}{fileName}")
        # print(len(faces))
        # print(eyes_qnt)



start = time.time()
face_cascade_name = "./utils/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "./utils/haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)     
# img = cv.imread("faces.jpg")
images_path = "/home/dotta/dev/images/"
image_list = getTargetImages(images_path)
s3_client = boto3.resource('s3')
bucket_name = "opencv-pistydotta"
s3_path = "results/local/999/"
detectAndDisplay(image_list, images_path, s3_client, bucket_name, s3_path)
end = time.time()
print("The time of execution of above program is :", end-start)
# with open('image1' + str(time.time()) + '.txt', 'w') as f:
#     f.write(str(end-start) + '\ntestando barra n')
# string_teste = "images/penis/image.png"
# imageName = string_teste.split('/')[len(string_teste.split('/'))-1]
# print(imageName)