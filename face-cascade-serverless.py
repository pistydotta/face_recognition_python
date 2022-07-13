import cv2 as cv
import boto3
import time



def load_cascades():
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    face_cascade.load(cv.samples.findFile('./haarcascade_frontalface_alt.xml'))
    eyes_cascade.load(cv.samples.findFile('./haarcascade_eye_tree_eyeglasses.xml'))
    return face_cascade, eyes_cascade

def load_image(bucketName, s3Key):
    tmp_filename='/tmp/my_image.jpg'
    s3 = boto3.resource('s3')
    # BUCKET_NAME = "opencv-pistydotta"
    # S3_KEY = "faces.jpg"
    s3.Bucket(bucketName).download_file(s3Key, tmp_filename)
    return tmp_filename

def process_image(filename, face_cascade, eyes_cascade):
    img = cv.imread(filename)
    eyes_qnt = 0
    frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        eyes_qnt += len(eyes)
    return faces, eyes_qnt
    
def send_results_to_s3(bucketName, imageName, start_time, end_time):
    s3 = boto3.client('s3')
    timestamp = str(time.time())
    fileName = '/tmp/' + imageName + timestamp + '.txt'
    s3FilePath = 'results/serverlessTesting/1/8/' + imageName + '.txt'
    # s3FilePath = 'results/serverlessTesting/999/'
    with open(fileName, 'w') as f:
        f.write(imageName + "\n" + str(start_time) + "\n" + str(end_time) + "\n" + str(end_time-start_time))
    s3.upload_file(fileName, 'opencv-pistydotta', s3FilePath)


def lambda_handler(event, context):
    start_time = time.time()
    face_cascade, eyes_cascade = load_cascades()
    bucketName = event['Records'][0]['s3']['bucket']['name']
    s3Key = event['Records'][0]['s3']['object']['key']
    filename = load_image(bucketName, s3Key)
    faces, eyes = process_image(filename, face_cascade, eyes_cascade)
    end_time = time.time()
    imageName = s3Key.split('/')[len(s3Key.split('/'))-1]
    send_results_to_s3(bucketName, imageName, start_time, end_time)
    # print("Number of faces: " + str(len(faces)))
    # print("Number of eyes: " + str(eyes))
    # print("Execution time: " + str(executionTime))
    
    return {
        'statusCode': 200,
        'body': str(len(faces)) + ' ' + str(eyes),
        'time': executionTime
    }
