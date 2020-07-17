import argparse

import os

from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def get_args():
    # gets the arguments from the command line
    parser = argparse.ArgumentParser('Load an IR to the inference engine')

    # create descriptions for the command line
    m_desc = "The location of the model XML File"

    # create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    # load the inference engine API
    plugin = IECore()

    # load the IR files into their related class
    model_bin  = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)

    # adding a CPU extension
    plugin.add_extension(CPU_EXTENSION, "CPU")

    # get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="CPU")

    # check for any unsupported layers and let the user know if anything is missing
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]

    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(supported_layers))
        print("Check whether extensions are available to add to IECore")
        exit(1)

    # load the network into the inference engine
    plugin.load_network(net, "CPU")
    print('IR successfully loaded into the inference engine')

    return


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()


