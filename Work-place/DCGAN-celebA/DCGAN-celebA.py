# prepare packages
import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt

# get data sets
import os
from glob import glob
import helper
dataset_dir = '../Data_sets/'

# Network definitions
# input placeholders
def model_inputs(image_width, image_height, image_channels, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='input_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='input_z')
    
    return inputs_real, inputs_z

# our generator
def generator(z, output_dim, reuse=False, initial_feature_size=1024, alpha=0.2, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):        
        # try different weight initializer
        # w_init = tf.contrib.layers.variance_scaling_initializer()
        # w_init = tf.truncated_normal_initializer(stddev=0.02)
        w_init = tf.contrib.layers.xavier_initializer()
        
        # 1. Fully connected layer (make 4x4x512) & reshape to prepare first layer
        feature_map_size = initial_feature_size
        x1 = tf.layers.dense(inputs=z, 
                             units=4*4*feature_map_size, 
                             activation=None, 
                             use_bias=True, 
                             kernel_initializer=w_init)
        x1 = tf.reshape(tensor=x1, shape=[-1, 4, 4, feature_map_size])
        x1 = tf.layers.batch_normalization(inputs=x1, training=is_training)
        x1 = tf.maximum(alpha * x1, x1)
        
        # 2. deconvolutional layer (make 8x8x256)
        feature_map_size = feature_map_size // 2
        x2 = tf.layers.conv2d_transpose(inputs=x1, 
                                        filters=feature_map_size, 
                                        kernel_size=5, 
                                        strides=2, 
                                        padding='same', 
                                        activation=None, 
                                        kernel_initializer=w_init)
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training)
        x2 = tf.maximum(alpha * x2, x2)
        
        # 3. deconvolutional layer (make 16x16x128)
        feature_map_size = feature_map_size // 2
        x3 = tf.layers.conv2d_transpose(inputs=x2, 
                                        filters=feature_map_size, 
                                        kernel_size=5, 
                                        strides=2, 
                                        padding='same', 
                                        activation=None, 
                                        kernel_initializer=w_init)
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training)
        x3 = tf.maximum(alpha * x3, x3)
        
        # 4. deconvolutional layer (make 32x32x64)
        feature_map_size = feature_map_size // 2
        x4 = tf.layers.conv2d_transpose(inputs=x3, 
                                        filters=feature_map_size, 
                                        kernel_size=5, 
                                        strides=2, 
                                        padding='same', 
                                        activation=None, 
                                        kernel_initializer=w_init)
        x4 = tf.layers.batch_normalization(inputs=x4, training=is_training)
        x4 = tf.maximum(alpha * x4, x4)
        
        # 4. Output layer, 64x64x3
        logits = tf.layers.conv2d_transpose(inputs=x4, 
                                            filters=output_dim, 
                                            kernel_size=5, 
                                            strides=2, 
                                            padding='same', 
                                            activation=None,
                                            kernel_initializer=w_init)
        out = tf.tanh(logits)
    return out

# our discriminator
def discriminator(x, reuse=False, initial_filter_size=64, alpha=0.2, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # input is 64x64x3
        
        # try different weight initializer
        # w_init = tf.contrib.layers.variance_scaling_initializer()
        # w_init = tf.truncated_normal_initializer(stddev=0.02)
        w_init = tf.contrib.layers.xavier_initializer()
        
        # 1. make 32x32x64
        filters = initial_filter_size
        x1 = tf.layers.conv2d(inputs=x, 
                              filters=filters, 
                              kernel_size=5, 
                              strides=2, 
                              padding='same', 
                              activation=None, 
                              kernel_initializer=w_init)
        x1 = tf.maximum(alpha * x1, x1)
        
        # 2. make 16x16x128
        filters = filters * 2
        x2 = tf.layers.conv2d(inputs=x1, 
                              filters=filters, 
                              kernel_size=5, 
                              strides=2, 
                              padding='same', 
                              activation=None, 
                              kernel_initializer=w_init)
        x2 = tf.layers.batch_normalization(inputs=x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)
        
        # 3. make 8x8x256
        filters = filters * 2
        x3 = tf.layers.conv2d(inputs=x2, 
                              filters=filters, 
                              kernel_size=5, 
                              strides=2, 
                              padding='same', 
                              activation=None, 
                              kernel_initializer=w_init)
        x3 = tf.layers.batch_normalization(inputs=x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)
        
        # 4. make 4x4x512
        filters = filters * 2
        x4 = tf.layers.conv2d(inputs=x3, 
                              filters=filters, 
                              kernel_size=5, 
                              strides=2, 
                              padding='same', 
                              activation=None, 
                              kernel_initializer=w_init)
        x4 = tf.layers.batch_normalization(inputs=x4, training=True)
        x4 = tf.maximum(alpha * x4, x4)
        
        # 5. flatten the layer
        flattend_layer = tf.reshape(tensor=x4, shape=[-1, 4*4*filters])
        logits = tf.layers.dense(inputs=flattend_layer, 
                                 units=1, 
                                 activation=None, 
                                 use_bias=True, 
                                 kernel_initializer=w_init)
        out = tf.sigmoid(logits)
    return out, logits

# loss definition
def model_loss(input_real, input_z, output_dim, alpha=0.2, smooth=0.1):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)
    
    d_real_label = tf.ones_like(d_logits_real) * (1 - smooth)
    d_fake_label = tf.zeros_like(d_logits_real)
    d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_real_label) )
    d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_fake_label) )
    g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)) )

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss

# optimizer definition
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

# model builder class
class DCGAN:
    def __init__(self, data_shape, z_size, learning_rate, alpha=0.2, beta1=0.5, smooth=0.1):
        tf.reset_default_graph()
        
        self.z_size = z_size

        self.input_real, self.input_z = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_size)
        
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, data_shape[3], alpha=alpha, smooth=smooth)
        
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)

# image helper function
def save_generator_output(sess, n_images, input_z, out_channel_dim, image_mode, img_str):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
    
    samples = sess.run(
        generator(input_z, out_channel_dim, reuse=True, is_training=False),
        feed_dict={input_z: example_z})
    
    images_grid = helper.images_square_grid(samples, image_mode)
    plt.imshow(images_grid, cmap=cmap)
    plt.savefig(img_str)

# actual training function
def train(net, epochs, batch_size, get_batches, data_shape, data_image_mode, show_n_images, print_every=30):
    losses = []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_images in get_batches(batch_size):
                steps += 1
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, net.z_size))

                # Run optimizers
                _ = sess.run(net.d_opt, feed_dict={net.input_real: batch_images, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: batch_images})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: batch_images})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))
            
            # save generated images on every epochs
            image_fn = './assets/DCGAN-celebA-epoch-{:d}.png'.format(e)
            save_generator_output(sess, show_n_images, net.input_z, data_shape[3], data_image_mode, image_fn)                    
                    
    return losses


# hyper parameters
z_size = 256
learning_rate = 0.0002
batch_size = 128
epochs = 30
alpha = 0.2
beta1 = 0.5
smooth = 0.1
show_n_images = 25

# get data
celebA_dataset = helper.Dataset(glob(os.path.join(dataset_dir, 'img_align_celeba/*.jpg')))

# Create the network
net = DCGAN(celebA_dataset.shape, z_size, learning_rate, alpha=alpha, beta1=beta1, smooth=smooth)

# start training
losses = train(net, epochs, batch_size, celebA_dataset.get_batches, celebA_dataset.shape, celebA_dataset.image_mode, show_n_images)

# create animated gif from result images
images = []
for e in range(epochs):
    image_fn = './assets/DCGAN-celebA-epoch-{:d}.png'.format(e)
    images.append( imageio.imread(image_fn) )
imageio.mimsave('./assets/DCGAN-celebA-by-epochs.gif', images, fps=5)

# plot losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show('hold')