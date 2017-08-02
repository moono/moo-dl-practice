# get packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data generators
class DataDistribution(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, x):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)
        # return np.random.normal(self.mu, self.sigma, N)

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

# Model
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    
    return inputs_real, inputs_z

# pre_D(x)
def pre_trainer(x, n_hidden=32, reuse=False, alpha=0.02):
    with tf.variable_scope('pre', reuse=reuse):
        # weight initializer 
        w_init = tf.contrib.layers.xavier_initializer()
        
        h1 = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # leacky ReLU
        
        # logits and sigmoid output
        logits = tf.layers.dense(h1, 1, activation=None, kernel_initializer=w_init)
        out = tf.sigmoid(logits)
        return out

# G(z)
def generator(z, out_dim=1, n_hidden=32, reuse=False, alpha=0.02):
    with tf.variable_scope('generator', reuse=reuse):
        # weight initializer 
        w_init = tf.contrib.layers.xavier_initializer()

        h1 = tf.layers.dense(z, n_hidden, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # leacky ReLU
        
        # logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None, kernel_initializer=w_init)
        out = tf.layers.tanh(logits)
        return out


# D(x)
def discriminator(x, n_hidden=32, reuse=False, alpha=0.02):
    with tf.variable_scope('discriminator', reuse=reuse):
        # weight initializer 
        w_init = tf.contrib.layers.xavier_initializer()
        
        h1 = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # leacky ReLU
        
        # logits and sigmoid output
        logits = tf.layers.dense(h1, 1, activation=None, kernel_initializer=w_init)
        out = tf.sigmoid(logits)
        return out, logits

def model_loss(smooth):
    # Create our input placeholders
    inputs_x, inputs_z = model_inputs(real_dim=1, z_dim=1)

    # Build the model
    g_model = generator(z=inputs_z, out_dim=1, n_hidden=n_hidden, reuse=False)
    d_model_real, d_logits_real = discriminator(x=inputs_x, n_hidden=n_hidden, reuse=False)
    d_model_fake, d_logits_fake = discriminator(x=g_model, n_hidden=n_hidden, reuse=True)

    # Calculate losses
    # discriminator
    real_labels = tf.ones_like(d_logits_real) * (1 - smooth)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=real_labels))

    fake_labels = tf.zeros_like(d_logits_real)
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=fake_labels))

    # discriminator's loss is sum of real and fake
    d_loss = d_loss_real + d_loss_fake

    # generator
    g_labels = tf.ones_like(d_logits_fake)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=g_labels))

    return d_loss, g_loss

def model_optimizer(d_loss, g_loss, learning_rate, beta1=0.5):
    # optimization
    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    # d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    # g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    
    return d_train_opt, g_train_opt

class Gaussian1dGAN(object):
    def __init__(self, mu, sigma, data_range, n_hidden, learning_rate, smooth):
        self.mu, self.sigma = mu, sigma

        self.real_data_gen = DataDistribution(mu, sigma)
        self.fake_data_gen = GeneratorDistribution(data_range)

        # clear tf graphs
        tf.reset_default_graph()
        self.d_loss, self.g_loss = model_loss(smooth)
        self.d_train_opt, self.g_train_opt = model_optimizer(self.d_loss, self.g_loss, learning_rate)

# Parameters
mu, sigma = 1., 1.5
data_range = 5
learning_rate = 0.03
input_size = 1 # Size of input
z_size = 1     # Size of latent vector to generator
n_hidden = 32  # Sizes of hidden layers
smooth = 0.1   # Smoothing 

net = Gaussian1dGAN(mu, sigma, data_range, n_hidden, learning_rate, smooth)

# def perform_pre_training():


