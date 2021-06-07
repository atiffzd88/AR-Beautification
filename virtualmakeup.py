import cv2
import numpy as np
import dlib
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def empty(a):
    pass

cv2.namedWindow('DoMakeup')
cv2.resizeWindow('DoMakeup',640,240)
cv2.createTrackbar('Blue','DoMakeup',0,255,empty)
cv2.createTrackbar('Green','DoMakeup',0,255,empty)
cv2.createTrackbar('Red','DoMakeup',0,255,empty)

def createBox(img,leftEye,rightEye,Lips,scale=5,masked=False,cropped = True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[leftEye,rightEye,Lips],(255,255,255))
        img = cv2.bitwise_and(img,mask)
    if cropped:
        bbox = cv2.boundingRect(leftEye,rightEye,Lips)
        x,y,w,h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask
while True:
    _, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        landmarks = predictor(imgGray,face)
        myPoints =[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
        myPoints = np.array(myPoints)

        imgFace = createBox(img,myPoints[36:41],myPoints[42:47],myPoints[48:67],8,masked=True,cropped=False)

        imgColor = np.zeros_like(imgFace)
        b = cv2.getTrackbarPos('Blue','DoMakeup')
        g = cv2.getTrackbarPos('Green','DoMakeup')
        r = cv2.getTrackbarPos('Red','DoMakeup')
        imgColor[:] = b,g,r
        imgColor = cv2.bitwise_and(imgFace,imgColor)
        imgColor = cv2.GaussianBlur(imgColor,(7,7),10)

        imgColor = cv2.addWeighted(img,1,imgColor,0.4,0)

        cv2.imshow('DoMakeup',imgColor)

        print(myPoints)

    key = cv2.waitKey(1)
    if key == 27:
        break