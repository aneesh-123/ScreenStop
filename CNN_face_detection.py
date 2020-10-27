import cv2
import dlib
import argparse
import time
from picamera import PiCamera
from time import sleep

# handle command line arguments
ap = argparse.ArgumentParser()
# ap.add_argument("CNN_face_detection.py")
ap.add_argument('-i', '--img', required=True, help='path to img file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()

camera = PiCamera()
path = '/home/pi/Desktop/img.jpg'
camera.start_preview()
sleep(5)
camera.capture(path)
camera.stop_preview()
img = cv2.imread(path)

# load input img
if img is None:
    print("Could not read input img")
    exit()
    
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

start = time.time()

# apply face detection (hog)
faces_hog = hog_face_detector(img, 1)

end = time.time()
print("Execution Time (in seconds) :")
print("HOG : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
start = time.time()

# apply face detection (cnn)
faces_cnn = cnn_face_detector(img, 1)

end = time.time()
print("CNN : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

     # draw box over face
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

# write at the top left corner of the img
# for color identification
img_height, img_width = img.shape[:2]
cv2.putText(img, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
cv2.putText(img, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output img
cv2.imshow("face detection with dlib", img)
cv2.waitKey()

# save output img 
cv2.imwrite("cnn_face_detection.png", img)

# close all windows
cv2.destroyAllWindows()