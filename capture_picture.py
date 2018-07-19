import cv2
import sys
from PIL import Image

def CatchUsbVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
  
    #video resource, can be from USB camera
    cap = cv2.VideoCapture(camera_idx)        

    #face classfier
    classfier = cv2.CascadeClassifier("/home/huangcm/work/tensorflow/face_recog/haarcascade_frontalface_alt2.xml")

    #diaplay the rectangle, RGB format
    color = (0, 255, 0)

    num = 1
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

                #save the current picture
                img_name = '%s/%d.jpg'%(path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)
                num +=1
                if num > (catch_pic_num):
                    break

                #draw the rectangle
                cv2.rectangle(frame, (x - 10, y - 20), (x + w + 10, y + h + 20), color, 2)

                #display the total number of picutres
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)

        #exit when get the enough pictures
        if num > (catch_pic_num):
            break
        #display picture and wait for 10s to enter from keyboard, 'q' exit application
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
                                         
    #free windown and usb camera
    cap.release()
    cv2.destroyAllWindows() 
                                                                     
if __name__ == '__main__':
    catch_pic_num = 1000
    path_name = "/home/huangcm/work/tensorflow/face_recog/data/me"
    if len(sys.argv) != 3:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("Capture video", int(sys.argv[1]), catch_pic_num, sys.argv[2])
