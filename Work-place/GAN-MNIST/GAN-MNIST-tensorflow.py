import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import pickle as pkl
import time
from pathlib import Path

# our place holders
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    
    return inputs_real, inputs_z

# gennrator network structure
def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.2):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        h1 = tf.layers.dense(z, n_units, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU
        
        h2 = tf.layers.dense(h1, n_units//4, activation=None, kernel_initializer=w_init)
        h2 = tf.maximum(alpha * h2, h2)
        
        # Logits and tanh(-1~1) output
        logits = tf.layers.dense(h2, out_dim, activation=None, kernel_initializer=w_init)
        out = tf.tanh(logits)
        
        return out

def discriminator(x, n_units=128, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        h1 = tf.layers.dense(x, n_units, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU
        
        h2 = tf.layers.dense(h1, n_units//4, activation=None, kernel_initializer=w_init)
        h2 = tf.maximum(alpha * h2, h2)

        # Logits and sigmoid(0~1) output
        logits = tf.layers.dense(h2, 1, activation=None, kernel_initializer=w_init)
        out = tf.sigmoid(logits)
        
        return out, logits


'''
hyper parameters
'''
# Size of input image to discriminator
input_size = 28 * 28 # 28x28 MNIST images flattened
# Size of latent vector to generator
z_size = 100
# learning rate
learning_rate = 0.002
# Leak factor for leaky ReLU
alpha = 0.2
# Label smoothing 
smooth = 0.0
beta1 = 0.5

# wipe out previous graphs and make us to start building new graph from here
tf.reset_default_graph()

# Create our input placeholders
inputs_real, inputs_z = model_inputs(real_dim=input_size, z_dim=z_size)

# Generator network here(g_model is the generator output)
g_model = generator(z=inputs_z, out_dim=input_size, reuse=False, alpha=alpha)

# Disriminator network here
d_model_real, d_logits_real = discriminator(x=inputs_real, reuse=False, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(x=g_model, reuse=True, alpha=alpha)

# Calculate losses
real_labels = tf.ones_like(d_logits_real) * (1 - smooth) # label smoothing
d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=real_labels) )

fake_labels = tf.zeros_like(d_logits_real)
d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=fake_labels) )

d_loss = d_loss_real + d_loss_fake

gen_labels = tf.ones_like(d_logits_fake)
g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=gen_labels) )

# Optimizers
# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes

'''
Training Session
'''
batch_size = 128
epochs = 100
samples = []
losses = []

# prepare saver
model_fn = './tf-model.ckpt'
saver = tf.train.Saver()

# use fixed z for displaying
fixed_z = np.random.uniform(-1, 1, size=(16, z_size))

# check if saved model file exits
if Path(model_fn+'.meta').is_file():
    # use saved model
    with tf.Session() as sess:
        saver.restore(sess, model_fn)
        print("Model restored.")

        gen_samples = sess.run( generator(inputs_z, input_size, reuse=True), feed_dict={inputs_z: fixed_z} )

        fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), gen_samples):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        #plt.savefig('./assets/last_generated.png')
        plt.show('hold')
else:
    # train new one
    # get data sets
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../Data_sets/MNIST_data')

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                
                # Get images, reshape and rescale to pass to D
                batch_images = batch[0].reshape((batch_size, input_size))
                batch_images = batch_images*2 - 1
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                
                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images, inputs_z: batch_z})
                _ = sess.run(g_train_opt, feed_dict={inputs_z: batch_z})
            
            # At the end of each epoch, get the losses and print them out
            train_loss_d = sess.run(d_loss, {inputs_real: batch_images, inputs_z: batch_z})
            train_loss_g = g_loss.eval({inputs_z: batch_z})
                
            print("Epoch {}/{}...".format(e+1, epochs),
                  "Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}".format(train_loss_g))    
            # Save losses to view after training
            losses.append((train_loss_d, train_loss_g))
            
            # Sample from generator as we're training for viewing afterwards
            # sample_z = np.random.uniform(-1, 1, size=(16, z_size))
            gen_samples = sess.run( generator(inputs_z, input_size, reuse=True), feed_dict={inputs_z: fixed_z} )
            samples.append(gen_samples)

        save_path = saver.save(sess, model_fn)
        print("Model saved in file: {}".format(save_path))

    end_time = time.time()
    total_time = end_time - start_time
    print('Elapsed time: ', total_time)
    # 100 epochs in 197.95 sec

    fig1, ax1 = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('./assets/losses.png')

    fig2, axes2 = view_samples(-1, samples)
    plt.savefig('./assets/last_generated.png')


    rows, cols = 10, 6
    fig3, axes3 = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes3):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.savefig('./assets/generation_via_epochs.png')

# plt.show('hold')