# prepare packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print('turning on interative mode')
plt.ion()

# get data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../Data_sets/MNIST_data', one_hot=True)


# our place holders
def model_inputs(x_dim, y_dim, z_dim):
	'''
	:param x_dim: real input size to discriminator. For MNIST 784
	:param y_dim: label input size to discriminator & generator. For MNIST 10
	:param z_dim: latent vector input size to generator. ex) 100
	'''
	inputs_x = tf.placeholder(tf.float32, [None, x_dim], name='inputs_x')
	inputs_y = tf.placeholder(tf.float32, [None, y_dim], name='inputs_y')
	inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
	
	return inputs_x, inputs_y, inputs_z


# gennrator network structure
def generator(z, y, out_dim, n_hidden_units=128, reuse=False,  alpha=0.01):
	'''
	:param z: placeholder of latent vector
	:param y: placeholder of labels
	'''
	with tf.variable_scope('generator', reuse=reuse):
		# weight initializer
		w_init = tf.contrib.layers.xavier_initializer()

		# concatenate inputs
		concatenated_inputs = tf.concat(axis=1, values=[z, y])

		h1 = tf.layers.dense(concatenated_inputs, n_hidden_units, activation=None, kernel_initializer=w_init)
		h1 = tf.maximum(alpha * h1, h1)
		
		h2 = tf.layers.dense(h1, n_hidden_units//4, activation=None, kernel_initializer=w_init)
		h2 = tf.maximum(alpha * h2, h2)
		
		# Logits and tanh (-1~1) output
		logits = tf.layers.dense(h2, out_dim, activation=None)
		out = tf.tanh(logits)
		
		return out


def discriminator(x, y, n_hidden_units=128, reuse=False, alpha=0.01):
	'''
	:param x: placeholder of real or fake inputs
	:param y: placeholder of labels
	'''
	with tf.variable_scope('discriminator', reuse=reuse):
		# weight initializer
		w_init = tf.contrib.layers.xavier_initializer()

		# concatenate inputs
		concatenated_inputs = tf.concat(axis=1, values=[x, y])

		h1 = tf.layers.dense(concatenated_inputs, n_hidden_units, activation=None, kernel_initializer=w_init)
		h1 = tf.maximum(alpha * h1, h1)
		
		# Logits and sigmoid (0~1) output
		logits = tf.layers.dense(h1, 1, activation=None, kernel_initializer=w_init)
		out = tf.sigmoid(logits)
		
		return out, logits


# Size of input image to discriminator (28x28 MNIST images flattened)
x_size = 28 * 28
# Size of labels (classes)
y_size = 10
# Size of latent vector to generator
z_size = 100
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# learning rate
learning_rate = 0.002
# Leak factor for leaky ReLU
alpha = 0.02
# Label smoothing 
smooth = 0.1


# Build Network

# wipe out previous graphs and make us to start building new graph from here
tf.reset_default_graph()

# Create our input placeholders
inputs_x, inputs_y, inputs_z = model_inputs(x_dim=x_size, y_dim=y_size, z_dim=z_size)

# Generator network here(g_model is the generator output)
g_model = generator(z=inputs_z, y=inputs_y, out_dim=x_size, n_hidden_units=g_hidden_size, reuse=False, alpha=alpha)

# Disriminator network here
d_model_real, d_logits_real = discriminator(x=inputs_x, y=inputs_y, n_hidden_units=d_hidden_size, reuse=False, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(x=g_model, y=inputs_y, n_hidden_units=d_hidden_size, reuse=True, alpha=alpha)

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

d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)


# Training session
lucky_number_to_generate = 7
batch_size = 100
epochs = 2
samples = []
losses = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in range(epochs):
		for ii in range(mnist.train.num_examples//batch_size):
			x_, y_ = mnist.train.next_batch(batch_size)
			
			# Get images, reshape and rescale to pass to D
			x_ = x_.reshape((batch_size, 784))
			x_ = x_*2 - 1
			
			# Sample random noise for G
			z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))
			
			# Run optimizers
			_ = sess.run(d_train_opt, feed_dict={inputs_x: x_, inputs_y: y_, inputs_z: z_})
			_ = sess.run(g_train_opt, feed_dict={inputs_z: z_, inputs_y: y_})
		
		# At the end of each epoch, get the losses and print them out
		train_loss_d = sess.run(d_loss, {inputs_x: x_, inputs_y: y_, inputs_z: z_})
		train_loss_g = g_loss.eval({inputs_z: z_, inputs_y: y_})
			
		print("Epoch {}/{}...".format(e+1, epochs),
			  "Discriminator Loss: {:.4f}...".format(train_loss_d),
			  "Generator Loss: {:.4f}".format(train_loss_g))    
		# Save losses to view after training
		losses.append((train_loss_d, train_loss_g))
		
		# Sample from generator as we're training for viewing afterwards
		sample_z = np.random.uniform(-1, 1, size=(16, z_size))
		sample_y = np.zeros(shape=[16, y_size])
		lucky_number_index = lucky_number_to_generate - 1
		sample_y[:, lucky_number_index] = 1
		gen_samples = sess.run( generator(z=inputs_z, y=inputs_y, out_dim=x_size, n_hidden_units=g_hidden_size, reuse=True),
								feed_dict={inputs_z: sample_z, inputs_y: sample_y})
		samples.append(gen_samples)


# display training loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show('hold')
#plt.pause(0.01)

# generate generator samples
def view_samples(epoch, samples):
	fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
	for ax, img in zip(axes.flatten(), samples[epoch]):
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.imshow(img.reshape((28,28)), cmap='Greys_r')
		plt.show('hold')
		#plt.pause(0.01)

	return fig, axes

# view last sample
_ = view_samples(-1, samples)


# rows, cols = 10, 6
# fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

# for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
# 	for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
# 		ax.imshow(img.reshape((28,28)), cmap='Greys_r')
# 		ax.xaxis.set_visible(False)
# 		ax.yaxis.set_visible(False)

