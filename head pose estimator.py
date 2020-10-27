#!/usr/bin/env python
#import libraries
import cv2
import numpy as np
import math
path = '/home/pi/Desktop/image.jpg'

#Necessary Variables
x,y,z = 0,1,2

# Read Image
im = cv2.imread(path)
print(im.shape)
im = cv2.resize(im,(0,0),None,0.5,0.5)
size = im.shape
print(size)
    
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (391, 166),     # Nose tip
                            (386, 242),     # Chin
                            (335, 128),     # Left eye left corner
                            (421, 119),     # Right eye right corne
                            (354, 195),     # Left Mouth corner
                            (408, 190)      # Right mouth corner
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
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (100,0,255), -1)


p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

#A is the nose vector
#B is the end point of the line coming from the nose vector
A = [int(image_points[0][0]), int(image_points[0][1]), 0]
B = [int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]), 1000.0]

#Calculate each respective angle relative to plane
angle_xy = math.degrees(math.atan((A[y] - B[y]) / (A[x] - B[x])))
angle_xz = math.degrees(math.atan((A[z]-B[z]) / (A[x] - B[x])))
angle_yz = math.degrees(math.atan(abs(A[y]-B[y]) / (A[z] - B[z])))
print(angle_xy,angle_xz,angle_yz)

cv2.line(im, p1, p2, (255,0,0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
