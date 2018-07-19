# tensorflow-mtcnn

MTCNN is one of the best face detection algorithms.
Here is inference only for MTCNN face detector on Tensorflow, which is based on davidsandberg's facenet project, include the python version and C++ version.

## Face classification for three people

1. detect_face.py - The Algorithms file for MTCNN
2. det1.npy, det2.npy and det3.npy - The models for P-Net, R-Net and O-Net
3. face_detect_mtcnn.py - Detect face with MTCNN
4. face_capture_picture-mtcnn.py - Capture the face picure with MTCNN and save them
5. face_train_use_keras.py - Train these pictures with Keras based on Tensorflow framework
6. face_predict_mtcnn_use_keras.py - Predict or classify any people of these three
7. data.tar.bz2 - the picture data, uncompress it to 'data' directory

## C++

There are two version for C++.

One is to be build inside tensorflow code repository, so that it needs to be copied to the directory tensorflow/example.
please check cpp/tf_embedded/README.md for details.

The other is the standalone one, just needs libtensorflow.so and c_api.h to build and run.
Please check cpp/standalone/README.md for more details

## Python Run
1. install tensorflow first, please refers to https://www.tensorflow.org/install
2. install python packages: opencv, numpy
3. python ./facedetect_mtcnn.py --input input.jpg --output  new.jpg

## Build tensorflow on arm64 board

Please check out the guide [how to build tensorflow on firefly](https://cyberfire.github.io/tensorflow/rk3399/howto%20build%20tensorflow%20on%20firefly.md)

## Credit

### Keras

http://keras-cn.readthedocs.io/en/latest/#keraspython

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN doc

https://blog.csdn.net/tinyzhao/article/details/53236191

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Tensorflow 

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet
From this directory:
  facenet/src/align


