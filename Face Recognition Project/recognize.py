

import cv2                        
import numpy as np                
from os import listdir            
from os.path import isfile,join
import pyttsx3
from face_recognition import face_detection_cli

# pyttsx3 used for text-to-speech.
k = pyttsx3.init()
sound = k.getProperty('voices')
k.setProperty('voice',sound[0].id)
k.setProperty('rate',130)
k.setProperty('pitch',200)

# Speak the given text
def speak(text):
    k.say(text)
    k.runAndWait()


# The path to the directory containing the training data is specified.

data_path = "sample/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# The image files are read from the directory and converted to grayscale. The grayscale images and their corresponding labels are stored in two lists.
Training_Data,Labels = [],[]

for i,files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels,dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Congratulations model is TRAINED ... *_*...")

# haarcascade_frontalface_default. xml : Detects faces.

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# The main function face_detector() is defined which takes an image and detects any faces present in it using the face classifier. 
# It returns the original image with rectangles drawn around the detected faces, and the region of interest (ROI) containing the face.

def face_detector(img,size = 0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(200,200))

    return img,roi

# The script captures the video stream from the default camera (0)
cap = cv2.VideoCapture(0)

# A while loop continuously reads frames from the video stream and passes them to the face_detector() function.

while True:
    ret,frame = cap.read()
    image , face = face_detector(frame)


    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            Confidence = int(100 * (1 - (result[1])/300))
            display_string = str(Confidence)+'% confidence it is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if Confidence > 65:
            cv2.putText(image, "HELLO USER", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper",image)
            

        else:
            cv2.putText(image, "CAN'T RECOGNISE", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Face Cropper", image)

    except:
        speak("face not found")
        cv2.putText(image, "Face not FoUnD", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Face Cropper", image)
        pass
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()




