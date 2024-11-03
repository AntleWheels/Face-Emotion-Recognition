import urllib.request
from facial_emotion_recognition import EmotionRecognition
import cv2
import numpy as np

er = EmotionRecognition(device='cpu')

url = "http://192.168.174.92:8080/shot.jpg"
while True :
    imagePath = urllib.request.urlopen(url)
    imageNp = np.array(bytearray(imagePath.read()),dtype=np.uint8)
    frame = cv2.imdecode(imageNp,-1)
    frame =er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow("Frame",frame)
    key =cv2.waitKey(1)
    if key ==27:
        break

cv2.destroyAllWindows()
