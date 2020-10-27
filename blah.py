from picamera import PiCamera
from time import sleep
import cv2
import time
'''##### Image Caputre with Camera########
camera = PiCamera()

while True:
    path = '/home/pi/Desktop/image.jpg'
    camera.start_preview(fullscreen=False,window=(100,200,1000,1000))
    sleep(2)
    camera.capture(path, use_video_port = True)
    camera.stop_preview()
    img = cv2.imread(path)

    cv2.putText(img, "NO FACE DETECTED!", (100,100), cv2.FONT_HERSHEY_SIMPLEX,4, (200,100,255),7)
    # Display image
    cv2.destroyAllWindows()
    cv2.imshow('yo', img)
    sleep(6)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cv2.destroyAllWindows()
'''


# define a video capture object 
'''vid = cv2.VideoCapture(0) 

while(True):
        
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    sleep(2)
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
'''

for x in range(0,10):
    print("taking picture")
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    #cv2.imwrite('test'+str(x),frame)
    cap.release()
    img = cv2.resize(img, (650,650))
    sleep(1)
    print("Picture taken")
    cv2.imshow('frame',img)
    cv2.imshow('frame2',img)
    cv2.moveWindow('frame',100,200)
    cv2.moveWindow('frame2',800,200)
    cv2.waitKey(1)
    #sleep(6)
    #cv2.destroyAllWindows()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.destroyAllWindows()

