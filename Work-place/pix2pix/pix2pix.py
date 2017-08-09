# from: https://github.com/affinelayer/pix2pix-tensorflow

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

from helper import Dataset


def discriminator(inputs, targets, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        n_layers = 3

        # [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        concatenated_inputs = tf.concat(values=[inputs, targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]

def main():
    train_input_image_dir = '../Data_sets/facades/train/'

    # will return list of tuples [ (inputs, targets), (inputs, targets), ... , (inputs, targets)]
    my_dataset = Dataset(train_input_image_dir)
    one_batch = my_dataset.get_next_batch(30)
    print('number of images: {:d}'.format(len(one_batch)))

    return 0

if __name__ == '__main__':
    main()



