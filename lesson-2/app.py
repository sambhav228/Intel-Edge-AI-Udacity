import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]

CAR_TYPES = ["car", "bus", "truck", "van"]


def get_args():

    parser = argparse.ArgumentParser("Basic Edge App with the inference engine")

    # description for commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML File"
    t_desc = "The type of model: POST, TEXT, or CAR_META"

    # add required and optional groups

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    required.add_argument("-c", help=c_desc, required=True)
    required.add_argument("-d", help=d_desc, default="CPU")

    args = parser.parse_args()

    return args


def get_mask(processed_output):

    empty = np.zeros(processed_output.shape)

    mask = np.dstack((empty, processed_output, empty))

    return mask


def create_output_image(model_type, image, output):

    if model_type == 'POSE':

        output = output[:-1]

        for c in range(len(output)):

            output[c] = np.where(output[c] > 0.5, 255, 0)

        output = np.sum(output, axis=0)

        pose_mask = get_mask(output)

        image = image + pose_mask

        return image

    elif model_type == 'TEXT':

        output = np.where(output[1] > 0.5, 255, 0)

        # get semantic mask
        text_mask = get_mask(output)

        # add the mask to the image
        image = image + text_mask

        return image

    elif model_type == 'CAR_META':

        # get the color and car type from their list
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]

        # scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)

        # write the text of color and type onto the image
        image = cv2.putText(image,
                            "Color: {}, Type: {}".format(color, car_type),
                            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            2 * scaler, (255, 255, 255), 3 * scaler)

        return image


def perform_inference(args):

    # performs inference on input image, given a model

    # create a network for using the inference engine
    inference_network = Network()

    # Load the model in the network, and obtain its input image
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)

    # preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # obtain the output of the inference request
    output = inference_network.extract_output()
    process_func = handle_output(args.t)
    processed_output = process_func(output, image.shape)

    # create an output image based on network
    try:
        output_image = create_output_image(args.t, image, processed_output)

        try:
            cv2.imwrite("outputs/{}-output.jpg".format(args.t), output_image)
            # cv2.imwrite('fha.jpg',output_image)
            # cv2.imshow(output_image)
        except:
            print('Cannot write image')
        print('Success')
    except:
        output_image = image
        print('Error')

    # save down the resulting image


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()





