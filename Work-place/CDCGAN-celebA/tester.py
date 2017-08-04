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

import sklearn.preprocessing as sklp

attrib_fn = os.path.join(dataset_dir, 'celeba_anno/list_attr_celeba_cropped.txt')
def new_parser(attr_fn):
    with open(attr_fn, 'r') as f:
        # parse number of data
        first_line = f.readline()
        n_data = int(first_line)

        # parse each attribute names & size
        second_line = f.readline()
        attr_names = second_line.split()
        n_attr = len(attr_names)

    attr_data = np.loadtxt(attr_fn, dtype=int, skiprows=2, usecols=range(1, n_attr + 1))

    attr_index = attr_names.index('Male')
    single_attr_data = attr_data[:, attr_index]

    lb = sklp.LabelBinarizer()
    lb.fit([-1, 1])
    onehot = lb.transform(single_attr_data)
    print(onehot)
    print(onehot.shape)
    # # convert that to 0 & 1
    # attr_data = (attr_data + 1) // 2

new_parser(attrib_fn)

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
    return concatenated_inputs

def discriminator(x, y_reshaped):
    concatenated_inputs = tf.concat(axis=3, values=[x, y_reshaped])
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
batch_size = 2

input_x, input_y, input_y_reshaped, input_z = model_inputs(image_width, image_height, image_channels, y_size, z_size)
ggg = generator(input_z, input_y)
ddd = discriminator(input_x, input_y_reshaped)


fixed_z = np.random.uniform(-1, 1, size=(10, z_size))
fixed_y = np.zeros(shape=[y_size, 10, y_size])
for c in range(y_size):
    fixed_y[c, :, c] = 1


sess = tf.InteractiveSession()



x_ = celebA_dataset.get_next_batch(batch_size)
y_ = celebA_attr.get_next_batch(batch_size)

# also reshape y for input_y_reshaped placeholder
y_reshaped_ = y_reshaper(y_, image_width, image_height)
print('y_reshaped_: ', y_reshaped_.shape)
# print(y_reshaped_[:,:,0])
# print(y_reshaped_[:,:,1])

# Sample random noise for G
z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))

concated_g = sess.run(ggg, feed_dict={input_z: z_, input_y: y_})
concated_d = sess.run(ddd, feed_dict={input_x: x_, input_y_reshaped: y_reshaped_})

print('concated_g: ', concated_g.shape)
print(concated_g)

print('concated_d: ', concated_d.shape)
print('batch1')
print(concated_d[0,:,:,3])
print(concated_d[0,:,:,4])
print('batch2')
print(concated_d[1,:,:,3])
print(concated_d[1,:,:,4])

print('printing results')
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5, 2), sharey=True, sharex=True)
for ax_row, y_ in zip(axes, fixed_y):
    samples = sess.run( generator(input_z, input_y), feed_dict={input_y: y_, input_z: fixed_z})
    for ax, img in zip(ax_row, samples):
        print(img)


sess.close()