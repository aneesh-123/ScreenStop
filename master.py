from picamera import PiCamera
from time import sleep
import cv2
import numpy as np
import dlib
import math
import time

# Variable Definitions
#camera = PiCamera()
x_axis,y_axis,z_axis = 0,1,2
Chin = 8
Nose_Tip = 30
Left_Eye = 36
Right_Eye = 45
Left_Mouth = 48
Right_Mouth = 64

#Function Definitions
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

def calculateAngle(start_point, end_point, y_distance):
    vector_magnitude = math.sqrt(((start_point[0]-end_point[0]) ** 2) + ((start_point[1] - end_point[1]) ** 2))
    z = math.sqrt((vector_magnitude ** 2) - (y_distance ** 2))
    print("vector",vector_magnitude,"z",z, "y", y_distance)
    angle_yz = math.degrees(math.asin(y_distance / vector_magnitude))
    angle_xz = math.degrees(math.acos(z / vector_magnitude))
    print("Angle xz: ", angle_xz, " Angle yz: ", angle_yz)
    return [angle_xz, angle_yz]

#################################################INITIATION OF WHILE LOOP#########################
while True:
    POI = [] # Clear your POI's
    ########################################TAKE PICTURE HERE#########################################
    print("Taking picture in 3 seconds")
    sleep(3)
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    print("Picture taken")
    
    '''path = '/home/pi/Desktop/image.jpg'
    camera.start_preview()
    sleep(5)
    camera.capture(path, use_video_port = True)
    camera.stop_preview()'''

    #####################################FIND FACIAL LANDMARKS OF FACE #########################
    #img = cv2.imread(path)
    img = cv2.resize(img,(650,650))
    imgOriginal = img.copy()

    #Detect and call detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #Convert to Gray Scale and detect faces
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray,2)
    print(faces)
    #Cycle through faces and find x and y of landmarks
    for face in faces:
        print("Found faces")
        x1,y1 = face.left(), face.top(),
        x2, y2 = face.right(), face.bottom()
        imgOriginal = cv2.rectangle(imgOriginal, (x1,y1),(x2,y2), (255,0,0), 3)
        landmarks = predictor(imgGray, face)
        myPoints = []
        
        #Draw the Landmarks and store the values
        for n in range(68):
            x = landmarks.part(n).x 
            y = landmarks.part(n).y
            getPOI(n,x,y)
            myPoints.append([x,y])
            cv2.circle(imgOriginal, (x,y), 1, (0,255,255), cv2.FILLED)
            cv2.putText(imgOriginal, str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,0),1)
            
            
        #Display Points of Interest
        print("Chin", str(POI[0]))
        print("Nose_Tip", str(POI[1]))
        print("Left_Eye", str(POI[2]))
        print("Right_Eye", str(POI[3]))
        print("Left_Mouth", str(POI[4]))
        print("Right_Mouth",str(POI[5]))
        
    #####################################DETERMINE HEAD POSE ##############################
    if(len(faces) > 0):
        size = img.shape
            
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    POI[1],     # Nose tip
                                    POI[0],     # Chin
                                    POI[2],     # Left eye left corner
                                    POI[3],     # Right eye right corne
                                    POI[4],     # Left Mouth corner
                                    POI[5]      # Right mouth corner
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                
                                ])


        # Camera internals

        focal_length = size[1]
        center = (size[0]/2, size[1]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        print ("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        print ("Rotation Vector:\n {0}".format(rotation_vector))
        print ("Translation Vector:\n {0}".format(translation_vector))


        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose


        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (100,0,255), -1)


        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        
        '''
        #A is the nose vector
        #B is the end point of the line coming from the nose vector
        A = [int(image_points[0][0]), int(image_points[0][1]), 0]
        B = [int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]), 1000.0]

        print(A,B)
        #Calculate each respective angle relative to plane
        angle_xy = 90 - math.degrees(math.atan((A[y_axis] - B[y_axis]) / (A[x_axis] - B[x_axis])))
        angle_xz = 90 - math.degrees(math.atan((A[z_axis]-B[z_axis]) / (A[x_axis] - B[x_axis])))
        angle_yz = 90 - math.degrees(math.atan((A[y_axis]-B[y_axis]) / (A[z_axis] - B[z_axis])))
        print(angle_xy,angle_xz,angle_yz)
        '''
        
        y_length = p1[1] - p2[1]
        print("point 1", p1[1], "point 2", p2[1])
        angle = calculateAngle(p1,p2,y_length)
        
        angle_xz = angle[0]
        angle_yz = angle[1]
        #if(angle_yz > 45 and angle_yz < 90):
            #cv2.putText(img, "LOOK AT THE SCREEN!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,100,255),7)
        if(angle_yz < -25):
            cv2.putText(img, "LOOK AT THE SCREEN!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,100,255),7)
        if(angle_xz > 25 or angle_xz < -25):
            cv2.putText(img, "LOOK AT THE SCREEN!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,100,255),7)
            
        cv2.putText(img, "XZ Angle" + str(angle_xz), (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 4)
        cv2.putText(img, "YZ Angle:" + str(angle_yz), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 4)
        cv2.line(img, p1, p2, (255,0,0), 2)

    else:
        cv2.putText(img, "LOOK AT THE SCREEN!", (100,100), cv2.FONT_HERSHEY_SIMPLEX,2, (200,100,255),7)
    # Display image\
    cv2.destroyAllWindows()
    cv2.imshow("Output", img)
    cv2.imshow('Original', imgOriginal)
    cv2.moveWindow('Output', 100, 200)
    cv2.moveWindow('Original', 800,200)
    cv2.waitKey(1)
    sleep(10)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cv2.destroyAllWindows()

    