# prepare packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

# get data sets
from glob import glob
import helper

dataset_dir = '../Data_sets/'

# get data
celebA_dataset = helper.Dataset(glob(os.path.join(dataset_dir, 'img_align_celeba/*.jpg')))
celebA_attr = helper.AttrParserCelebA(os.path.join(dataset_dir, 'celeba_anno/list_attr_celeba.txt'), attr_name='Male')

# our place holders
def model_inputs(image_width, image_height, image_channels, y_dim, z_dim):
    inputs_x = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='inputs_x')
    inputs_y = tf.placeholder(tf.float32, [None, y_dim], name='inputs_y')
    inputs_y_reshaped = tf.placeholder(tf.float32, [None, image_width, image_height, y_dim], name='inputs_y_reshaped')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    
    return inputs_x, inputs_y, inputs_y_reshaped, inputs_z

# generator network structure
def generator(z, y):
    concatenated_inputs = tf.concat(values=[z, y], axis=1)
    print(concatenated_inputs)
    return concatenated_inputs

# reshape y into appropriate input to D
def y_reshaper(y, width, height):
    new_y = np.zeros((y.shape[0], width, height, y.shape[1]))
    
    for i in range(y.shape[0]):
        new_y[i, :, :, :] = y[i,:] * np.ones((width, height, y.shape[1]))
    return new_y

# Hyperparameters
image_width = 64
image_height = 64
image_channels = 3
y_size = 2 # labels: Female or Male
z_size = 100
batch_size = 1

input_x, input_y, input_y_reshaped, input_z = model_inputs(image_width, image_height, image_channels, y_size, z_size)
ggg = generator(input_z, input_y)


fixed_z = np.random.uniform(-1, 1, size=(10, z_size))
fixed_y = np.zeros(shape=[y_size, 10, y_size])
for c in range(y_size):
    fixed_y[c, :, c] = 1


sess = tf.InteractiveSession()



x_ = celebA_dataset.get_next_batch(batch_size)
y_ = celebA_attr.get_next_batch(batch_size)

# also reshape y for input_y_reshaped placeholder
y_reshaped_ = y_reshaper(y_, image_width, image_height)

# Sample random noise for G
z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))

concated = sess.run(ggg, feed_dict={input_z: z_, input_y: y_})
print(concated)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5, 2), sharey=True, sharex=True)
for ax_row, y_ in zip(axes, fixed_y):
    samples = sess.run( generator(input_z, input_y), feed_dict={input_y: y_, input_z: fixed_z})
    for ax, img in zip(ax_row, samples):
        print(img)


sess.close()