import cv2
import numpy as np


def preprocessing(input_image, height, width):

    image = cv2.resize(input_image, (width,height))

    image = image.transpose((2, 0, 1))

    image = image.reshape(1, 3, height, width)

    return image


def pose_estimation(input_image):

    preprocessed_image = np.copy(input_image)

    preprocessed_image = preprocessing(preprocessed_image, 256, 456)

    return preprocessed_image


def text_detection(input_image):

    preprocessed_image = np.copy(input_image)

    preprocessed_image = preprocessing(preprocessed_image, 768, 1280)

    return preprocessed_image


def car_meta(input_image):

    preprocessed_image = preprocessing(preprocessed_image, 72, 72)

    return preprocessed_image



