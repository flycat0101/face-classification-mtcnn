import cv2
import sys
from PIL import Image

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
  
    #video resource, can be from USB camera
    cap = cv2.VideoCapture(camera_idx)        

    #face classfier
    classfier = cv2.CascadeClassifier("/home/huangcm/work/tensorflow/face_recog/haarcascade_frontalface_alt2.xml")

    #diaplay the rectangle, RGB format
    color = (0, 255, 0)

    while cap.isOpened():
        ok, frame = cap.read() #read one frame data
        if not ok:
            break                    

        #change to grey picture
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect face
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #detect face
            for faceRect in faceRects:  #display rectangle for each face
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 20), (x + w + 10, y + h + 20), color, 2)

        #display picture and wait for 10s to enter from keyboard, 'q' exit application
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
                                         
    #free windown and usb camera
    cap.release()
    cv2.destroyAllWindows() 
                                                                     
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("Capture video", int(sys.argv[1]))
