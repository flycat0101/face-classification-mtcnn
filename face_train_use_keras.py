import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

class Dataset:
    def __init__(self, path_name):
        # trann set
        self.train_images = None
        self.train_labels = None

        # valid set
        self.valid_images = None
        self.valid_labels = None

        # test set
        self.test_images  = None
        self.test_labels  = None

        # data path
        self.path_name    = path_name
        
        # dimensionality number
        self.input_shape = None

    # load data set and cross validation
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, nb_classes = 3):
        # load data to memory
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.2, random_state = random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))

        # if the Framework is 'th' (for theano), then the picture range is: channels,rows,cols, otherwise, rows,cols,channels
        # all these codes are needed by Keras lib
        if K.image_dim_ordering() == 'th': # for theano framwork
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:   # for tensorflow framework
             train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
             valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
             test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
             self.input_shape = (img_rows, img_cols, img_channels)

        # output train, validation and test set
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        # use categorical_crossentropy as loss function, so need nb_classes to encode lable with one-hot
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        # pixel to float
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # normalization to 0~1 for pixel
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels  = test_labels


# CNN net model
class Model:
    def __init__(self):
        self.model = None

    # setup model
    def build_model(self, dataset, nb_classes = 3):
        # structure one empty net model, Linear stack model
        self.model = Sequential()

        # add all layer CNN needed, one "add" is one layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = dataset.input_shape))  #1 2D convolution layer
        self.model.add(Activation('relu'))                              #2 activate function

        self.model.add(Convolution2D(32, 3, 3))                         #3 2D convolution layer
        self.model.add(Activation('relu'))                              #4 activate function

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #5 pooling layer
        self.model.add(Dropout(0.25))                                   #6 Dropout layer

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))     #7 2D convolution layer
        self.model.add(Activation('relu'))                              #8 activate function

        self.model.add(Convolution2D(64, 3, 3))                         #9 2D convolution layer
        self.model.add(Activation('relu'))                              #10 activate function

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #11 pooling layer
        self.model.add(Dropout(0.25))                                   #12 Dropout layer

        self.model.add(Flatten())                                       #13 Flatten layer
        self.model.add(Dense(512))                                      #14 Dense layer
        self.model.add(Activation('relu'))                              #15 activate function
        self.model.add(Dropout(0.5))                                    #16 Dropout layer
        self.model.add(Dense(nb_classes))                               #17 Dense layer
        self.model.add(Activation('softmax'))                           #18 classifcation layer

        # output the result
        self.model.summary()

    # train model
    def train(self, dataset, batch_size = 20, nb_epoch = 10, data_augmentation = True):
        sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True) # use SGD+momentum
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy']) # finish model match work

        # not use data promote
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                    dataset.train_labels,
                    batch_size = batch_size,
                    nb_epoch = nb_epoch,
                    validation_data = (dataset.valid_images, dataset.valid_labels),
                    shuffle = True)
        else:   # use data promote
            datagen = ImageDataGenerator(
                    featurewise_center = False,             # whether input data decentration
                    samplewise_center  = False,             # whether input data to 0
                    featurewise_std_normalization = False,  # whether input data standardization
                    samplewise_std_normalization  = False,  # whether input data divide itself's standard deviation
                    zca_whitening = False,                  # whether input data ZCA 
                    rotation_range = 20,                    # input data angle (0 ~ 180)
                    width_shift_range  = 0.2,               # input data horizontal shift (0 ~ 1)
                    height_shift_range = 0.2,               # input data vertical shift (0 ~ 1)
                    horizontal_flip = True,                 # input data horizontal overturn
                    vertical_flip = False)                  # input data vertical overturn

            # all train set features normalization and ZCA handle
            datagen.fit(dataset.train_images)
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                            batch_size = batch_size),
                    samples_per_epoch = dataset.train_images.shape[0],
                    nb_epoch = nb_epoch,
                    validation_data = (dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = './hcm.face.model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # face recognition
    def face_predict(self, image):
        # make sure the Dimensions of the order according to Tensorflow or Theano
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)     # size must be the same IMAGE_SIZE X IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE)) # has different size, just to one picture
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # float and normalization
        image = image.astype('float32')
        image /= 255

        # give the probability of the picture
        result = self.model.predict_proba(image)
        print('result:', result)

        # give the result 0, 1 or 2
        result = self.model.predict_classes(image)
        # print('result too:', result[0])

        # return the result
        return result[0]


if __name__ == '__main__':
    dataset = Dataset('./data/')    
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path = './model/hcm.face.model.h5')

    model.load_model(file_path = './model/hcm.face.model.h5')
    model.evaluate(dataset)
