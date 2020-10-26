import cv2
import numpy as np
import dlib
#POI = ["Nose tip","Chin","Left eye left corner","Right eye right corner","Left Mouth corner","Right mouth corner"]
POI = []
Chin = 8
Nose_Tip = 30
Left_Eye = 36
Right_Eye = 45
Left_Mouth = 48
Right_Mouth = 64

def getPOI(point_number, x, y):
    if(point_number == Chin):
        POI.append([x,y])
    if(point_number == Nose_Tip):
        POI.append([x,y])
    if(point_number == Left_Eye):
        POI.append([x,y])
    if(point_number == Right_Eye):
        POI.append([x,y])
    if(point_number == Left_Mouth):
        POI.append([x,y])
    if(point_number == Right_Mouth):
        POI.append([x,y])

'''path = '/home/pi/Desktop/image.jpg'
img = cv2.imread(path)
'''
while True:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    imgOriginal = img.copy()


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imgCrop = imgGray[0:350, 150:850]
    faces = detector(imgGray,1)
    print(faces)

    for face in faces:
        x1,y1 = face.left(), face.top(),
        x2, y2 = face.right(), face.bottom()
        imgOriginal = cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 3)
        landmarks = predictor(imgGray, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x 
            y = landmarks.part(n).y
            getPOI(n,x,y)
            myPoints.append([x,y])
            cv2.circle(imgOriginal, (x,y), 1, (0,255,255), cv2.FILLED)
            cv2.putText(imgOriginal, str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,0),1)
            
            
        print("Chin", str(POI[0]))
        print("Nose_Tip", str(POI[1]))
        print("Left_Eye", str(POI[2]))
        print("Right_Eye", str(POI[3]))
        print("Left_Mouth", str(POI[4]))
        print("Right_Mouth",str(POI[5]))

    cv2.imshow('Original',imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.waitKey(0)

