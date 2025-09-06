import cv2
import mediapipe as mp
import time
import math
import numpy as np
import HandTrackingModule as htm
import autopy

#declaring variables
wCam=640
hCam=480
frameR=100 #frame size
smoothing=5
plocX,plocY=0,0
clocX,clocY=0,0



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime=0
detector=htm.handDetector(maxHands=1)
wScr,hScr=autopy.screen.size()
print(wScr,hScr)



while True:
    success, img=cap.read()
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)


    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        #print(x1,y1,x2,y2)


        finger=detector.fingersUp()
       # print (finger)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

        if finger[1]==1 and finger[2]==0:
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            autopy.mouse.move(x3,y3)

            clocX=plocX+(x3-plocX)/smoothing
            clocY=plocY+(y3-plocY)/smoothing

            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY=clocX,clocY

        if finger[1]==1 and finger[2]==1:
            length, img, lineInfo =detector.findDistance(8,12,img) 
            print(length)
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,250,0),cv2.FILLED)
                autopy.mouse.click()



    #setting FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
     
    #displaying the image 
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
