import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64

#resize the image
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    #get the size of picture
    h, w, _ = image.shape

    #get the longgest edge
    longest_edge = max(h, w)

    #compute how many pixels need to be added
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB color
    BLACK = [0, 0, 0]

    #add the edge, heigh = width
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #resize the picture and return
    return cv2.resize(constant, (height, width))

#readthe tranning set
images = []
labels = []
labelsn = []
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        #start from the initialized path, merge all path recognized
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):    #if directory, recursion call read_path
            read_path(full_path)
        else:   #picture file
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                #see the picture resized
                #cv2.imwrite('1.jpg', image)
                images.append(image)
                labels.append(path_name)

    return images,labels

#read the data
def load_dataset(path_name):
    images,labels = read_path(path_name)

    #64x64 for each picture, RGB: 3x64x64
    images = np.array(images)
    print(images.shape)

    #lable data, 'hcm' directory is my face picture, marks 0, mark "yxl" to 1, others to 2
#    labels = np.array([0 if label.endswith('hcm') else 1 for label in labels])
    for label in labels:
        if label.endswith('hcm'):
            labelsn.append(np.array([0]))
        elif label.endswith('yxl'):
            labelsn.append(np.array([1]))
        else:
            labelsn.append(np.array([2]))

    return images, labelsn

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_dataset(sys.argv[1])
