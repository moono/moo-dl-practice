# from: https://github.com/affinelayer/pix2pix-tensorflow

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

from helper import Dataset

def generator(inputs, out_channels, n_first_layer_filter=64, alpha=0.2, reuse=False, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # for convolution weights initializer
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        # prepare to stack layers
        layers = []

        # encoders
        # encoder_1: [batch, 256, 256, 3] => [batch, 128, 128, 64]
        layer1 = tf.layers.conv2d(inputs, filters=n_first_layer_filter, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=False)
        layers.append(layer1)

        layer_specs = [
            n_first_layer_filter * 2,  # encoder_2: [batch, 128, 128, 64] => [batch, 64, 64, 128]
            n_first_layer_filter * 4,  # encoder_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
            n_first_layer_filter * 8,  # encoder_4: [batch, 32, 32, 256] => [batch, 16, 16, 512]
            n_first_layer_filter * 8,  # encoder_5: [batch, 16, 16, 512] => [batch, 8, 8, 512]
            n_first_layer_filter * 8,  # encoder_6: [batch, 8, 8, 512] => [batch, 4, 4, 512]
            n_first_layer_filter * 8,  # encoder_7: [batch, 4, 4, 512] => [batch, 2, 2, 512]
            n_first_layer_filter * 8,  # encoder_8: [batch, 2, 2, 512] => [batch, 1, 1, 512]
        ]
        for out_channels in layer_specs:
            layer = tf.maximum(alpha * layers[-1], layers[-1])
            layer = tf.layers.conv2d(layer, filters=out_channels, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=w_init, use_bias=False)
            layer = tf.layers.batch_normalization(inputs=layer, training=is_training)
            layers.append(layer)

        # decoders
        num_encoder_layers = len(layers)
        layer_specs = [
            (n_first_layer_filter * 8, 0.5),  # decoder_8: [batch, 1, 1, 512] => [batch, 2, 2, 512]
            (n_first_layer_filter * 8, 0.5),  # decoder_7: [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (n_first_layer_filter * 8, 0.5),  # decoder_6: [batch, 4, 4, 512] => [batch, 8, 8, 512]
            (n_first_layer_filter * 8, 0.0),  # decoder_5: [batch, 8, 8, 512] => [batch, 16, 16, 512]
            (n_first_layer_filter * 4, 0.0),  # decoder_4: [batch, 16, 16, 512] => [batch, 32, 32, 256]
            (n_first_layer_filter * 2, 0.0),  # decoder_3: [batch, 32, 32, 256] => [batch, 64, 64, 128]
            (n_first_layer_filter, 0.0),  # decoder_2: [batch, 64, 64, 128] => [batch, 128, 128, 64]
        ]
        for decoder_layer, (n_filters, dropout) in enumerate(layer_specs):
            # handle skip layer
            skip_layer = num_encoder_layers - decoder_layer - 1
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            layer = tf.maximum(alpha * inputs, inputs)
            layer = tf.layers.conv2d_transpose(inputs=layer, filters=n_filters, kernel_size=4, strides=2, padding='same')
            layer = tf.layers.batch_normalization(inputs=layer, training=is_training)

            # handle dropout
            if dropout > 0.0:
               layer = tf.layers.dropout(layer, rate=dropout)

            # stack
            layers.append(layer)

        # decoder_1: [batch, 128, 128, 64] => [batch, 256, 256, out_channels]
        last_layer = tf.maximum(alpha * layers[-1], layers[-1])
        last_layer = tf.layers.conv2d_transpose(inputs=last_layer, filters=out_channels, kernel_size=4, strides=2,
                                                padding='same')
        out = tf.tanh(last_layer)
        layers.append(last_layer)

        return out

def discriminator(inputs, targets, n_first_layer_filter=64, alpha=0.2, reuse=False, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # for convolution weights initializer
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        # concatenate inputs
        # M: batch size
        # Mx256x256x3 + Mx256x256x3 => Mx256x256x6
        concat_inputs = tf.concat(values=[inputs, targets], axis=3)

        # layer_1: [batch, 256, 256, 6] => [batch, 128, 128, 64], without batchnorm
        l1 = tf.layers.conv2d(concat_inputs, filters=n_first_layer_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l1 = tf.maximum(alpha * l1, l1)

        # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128], with batchnorm
        n_filter = n_first_layer_filter * 2
        l2 = tf.layers.conv2d(l1, filters=n_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256], with batchnorm
        n_filter = n_first_layer_filter * 4
        l3 = tf.layers.conv2d(l2, filters=n_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # layer_4: [batch, 32, 32, 256] => [batch, 31, 31, 512], with batchnorm
        n_filter = n_first_layer_filter * 8
        l4 = tf.layers.conv2d(l3, filters=n_filter, kernel_size=2, strides=1, padding='valid',
                              kernel_initializer=w_init, use_bias=False)
        l4 = tf.layers.batch_normalization(inputs=l4, training=is_training)
        l4 = tf.maximum(alpha * l4, l4)

        # layer_5: [batch, 31, 31, 512] => [batch, 30, 30, 1], without batchnorm
        n_filter = 1
        l5 = tf.layers.conv2d(l4, filters=n_filter, kernel_size=2, strides=1, padding='valid',
                              kernel_initializer=w_init, use_bias=False)
        out = tf.sigmoid(l5)

        return out



def main():
    train_input_image_dir = '../Data_sets/facades/train/'

    # will return list of tuples [ (inputs, targets), (inputs, targets), ... , (inputs, targets)]
    my_dataset = Dataset(train_input_image_dir)
    one_batch = my_dataset.get_next_batch(30)
    print('number of images: {:d}'.format(len(one_batch)))

    return 0

def test():
    sess = tf.InteractiveSession()
    t = tf.zeros([64, 64, 64, 128], dtype=tf.float32)

    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # l1 = tf.layers.conv2d(t, filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=w_init, use_bias=False)
    l1 = tf.layers.conv2d_transpose(t, filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=w_init,
                                    use_bias=False)

    print(l1.shape)

    sess.close()
if __name__ == '__main__':
    # main()
    test()



