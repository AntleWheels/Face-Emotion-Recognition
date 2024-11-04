import urllib.request
from facial_emotion_recognition import EmotionRecognition
import cv2
import numpy as np

er = EmotionRecognition(device='cpu') # the value in the device == cpu is dependents upon the what we are entering in the serialization .py in the torch library

url = "Your IP for the camera" # this is the public IP of the camera
while True :
    imagePath = urllib.request.urlopen(url) # It is used to open the url
    imageNp = np.array(bytearray(imagePath.read()),dtype=np.uint8)
    frame = cv2.imdecode(imageNp,-1)
    frame =er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow("Frame",frame)
    key =cv2.waitKey(1)
    if key ==27:
        break

cv2.destroyAllWindows()
