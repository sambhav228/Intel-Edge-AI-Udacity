import cv2
import numpy as np


def handle_pose(output, input_shape):

    heatmaps = output['Mconv7_stage2_L2']
    # print(pose.shape)

    # resize the heatmap back to the size of the input
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])

    # iterate through and resize each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):

    # extract only the first blob output (text/no text classification)
    text_classes = output['model/segm_logits/add']

    # resize this output back to the size of the input
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])

    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    # print(output.keys())
    return out_text


def handle_car(output, input_shape):

    color = output['color'].flatten()
    car_type = output['type'].flatten()

    color_class = np.argmax(color)
    type_class = np.argmax(car_type)

    return color_class, type_class


def handle_output(model_type):

    if model_type == 'POSE':

        return handle_pose

    elif model_type == 'TEXT':

        return handle_text

    elif model_type == 'CAR_META':

        return handle_car

    else:

        return None


def preprocessing(input_image, height, width):

    image = cv2.resize(input_image, (width,height))

    image = image.transpose((2, 0, 1))

    image = image.reshape(1, 3, height, width)

    return image
