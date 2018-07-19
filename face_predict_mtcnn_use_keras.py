import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
from face_train_use_keras import Model
import cv2

if __name__ == '__main__':
    camera_idx = 0
    window_name = 'Face Classification!'
#    cv2.namedWindow(window_name)    

    # load the model
    model = Model()
    model.load_model(file_path = './model/hcm.face.model.h5')

    #video resource, can be from USB camera
    cap = cv2.VideoCapture(camera_idx)

    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    color = (0, 255, 0)

    while cap.isOpened():
        ok, frame = cap.read() #read one frame data
        if not ok:
            break

        #change to grey picture
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
        #detect face
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if bounding_boxes.shape[0] > 0:
            for faceRect in bounding_boxes:
                image = frame[int(faceRect[1]) - 10: int(faceRect[3]) + 10, int(faceRect[0]) - 10: int(faceRect[2]) + 10]
                faceID = model.face_predict(image)

                cv2.rectangle(frame, (int(faceRect[0]), int(faceRect[1])), (int(faceRect[2]), int(faceRect[3])), color)

                name= 'Xu or other' #default other
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
                        (int(faceRect[0]) + 20, int(faceRect[1]) - 20), # locate
                        cv2.FONT_HERSHEY_SIMPLEX,                       # font
                        1,                                              # size
                        (255,0,255),                                    # color
                         2)                                             # line width

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    #free windown and usb camera
    cap.release()
    cv2.destroyAllWindows()
