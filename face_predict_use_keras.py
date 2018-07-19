import cv2
import sys
from face_train_use_keras import Model
import gc

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

#    camera_idx = int(sys.argv[1])
    camera_idx = 0

    # load the model
    model = Model()
    model.load_model(file_path = './model/hcm.face.model.h5')
    print("Load model successfully")

    # video resource, can be from USB camera
    cap = cv2.VideoCapture(camera_idx)        

    # load the face classifier
    cascade = cv2.CascadeClassifier("/home/huangcm/work/tensorflow/face_recog/haarcascade_frontalface_alt2.xml")

    # the color of the rectangle, RGB format
    color = (0, 255, 0)

    while cap.isOpened():
        _, frame = cap.read() # read one frame data

        # change to grey picture
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect face
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #detect face
            for faceRect in faceRects:  #display rectangle for each face
                x, y, w, h = faceRect
                #cv2.rectangle(frame, (x - 10, y - 20), (x + w + 10, y + h + 20), color, 2)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                name = 'Xu or others' # default Xu or others
                # if it is me
                if faceID == 0:
                    # text to me
                    name = 'Jerry'
                elif faceID == 1:
                    # text to xiaoliang
                    name = 'Xiaoliang'
                else:
                    pass
                cv2.putText(frame, name,
                        (x + 20, y - 20),           # locate
                        cv2.FONT_HERSHEY_SIMPLEX,   # font
                        1,                          # size
                        (255,0,255),                # color
                        2)                          # line width

        #display picture and wait for 10s to enter from keyboard, 'q' exit application
        cv2.imshow("Face Classification!", frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
                                         
    #free windown and usb camera
    cap.release()
    cv2.destroyAllWindows() 
                                                                     
