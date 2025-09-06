import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    #running the web camera
    success,img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
     
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,Lm in enumerate(handLms.landmark):
                #print(id)
                #print(Lm)
                h,w,c = img.shape
                cx,cy=int(Lm.x*w),int(Lm.y*h)
                print(id)
                print(cx,cy)
                #if id==4:
                cv2.circle(img,(cx,cy),10,(250,250,250),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mphands.HAND_CONNECTIONS) 
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(250,250,250),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
