from facial_emotion_recognition import EmotionRecognition # we are just importing the class
import cv2
er = EmotionRecognition(device='cpu') # we are creating an variable of the class

cam = cv2.VideoCapture(0) # This synrtax is used to acces the camera
while True :
    Success,frame =cam.read() # This synrtax is used to read the camera
    frame =er.recognise_emotion(frame,return_type='BGR') # This synrtax is used to detect the emotion 
    cv2.imshow("Frame",frame)# This synrtax is used to display the frame
    key =cv2.waitKey(1)
    if key ==27:
        break
cam.release()
cv2.destroyAllWindows()
