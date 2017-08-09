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

    def sample(self, N):
        return np.random.normal(self.mu, self.sigma, N)

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

# plotting functions
def test_samples(D, G, inputs_real, inputs_z, session, data, gen, sample_range, batch_size, num_points=10000, num_bins=100):
    # 1. decision boundary
    xs = np.linspace(-sample_range, sample_range, num_points)
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        x_ = xs[batch_size * i:batch_size * (i + 1)]
        x_ = np.reshape(x_, [batch_size, 1])
        db[batch_size * i:batch_size * (i + 1)] = session.run(D, {inputs_real: x_})

    # bins for computing histogram
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # 2. real data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    if G is not None:
        # 3. generated samples
        zs = gen.sample(num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // batch_size):
            z_ = zs[batch_size * i:batch_size * (i + 1)]
            z_ = np.reshape(z_, [batch_size, 1])
            g[batch_size * i:batch_size * (i + 1)] = session.run(G, {inputs_z: z_})
        pg, _ = np.histogram(g, bins=bins, density=True)
    else:
        pg = None
    return db, pd, pg

def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    ax.set_xlim(-sample_range, sample_range)
    plt.plot(p_x, pd, label='real data')
    if pg is not None:
        plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

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
        out = tf.tanh(logits)
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

def model_loss(input_size, z_size, n_hidden, smooth):
    # Create our input placeholders
    inputs_x, inputs_z = model_inputs(real_dim=input_size, z_dim=z_size)

    # Build the model
    g_model = generator(z=inputs_z, out_dim=input_size, n_hidden=n_hidden, reuse=False)
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

    return inputs_x, inputs_z, d_model_real, g_model, d_loss, g_loss

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
    
    return d_train_opt, g_train_opt, d_vars

class Gaussian1dGAN(object):
    def __init__(self, mu, sigma, data_range, input_size, z_size, n_hidden, learning_rate, smooth):
        self.mu, self.sigma, self.data_range = mu, sigma, data_range

        self.input_size, self.z_size = input_size, z_size

        # clear tf graphs
        tf.reset_default_graph()

        self.inputs_real, self.inputs_z, self.d_model_real, self.g_model, self.d_loss, self.g_loss = model_loss(input_size, z_size, n_hidden, smooth)
        self.d_train_opt, self.g_train_opt, self.d_vars = model_optimizer(self.d_loss, self.g_loss, learning_rate)

def perform_pre_training(mu, sigma, data_range, input_size, learning_rate, beta1, epochs):
    tf.reset_default_graph()

    # Create our input placeholders
    inputs_pre = tf.placeholder(tf.float32, [None, input_size], name='inputs_pre')
    labels_pre = tf.placeholder(tf.float32, [None, input_size], name='labels_pre')

    # Build the model
    d_model_pre = pre_trainer(x=inputs_pre, reuse=False)

    # Calculate losses
    d_loss_pre = tf.reduce_mean(tf.square(d_model_pre - labels_pre))

    # optimization
    t_vars = tf.trainable_variables()
    pre_vars = [var for var in t_vars if var.name.startswith('pre')]

    pre_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss_pre, var_list=pre_vars)

    # clear normal distribution
    N = 300
    x_ = np.linspace(-data_range, data_range, N)
    y_ = norm.pdf(x_, loc=mu, scale=sigma)
    
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            x_ = np.reshape(x_, [N, input_size])
            y_ = np.reshape(y_, [N, input_size])
            
            # Run optimizers
            _, train_loss_pre = sess.run([pre_train_opt, d_loss_pre], feed_dict={inputs_pre: x_, labels_pre: y_})
            
            # Save losses to view after training
            losses.append(train_loss_pre)
            
            if (e+1) % 100 == 0:
                print("Pre-training Epoch {}/{}...".format(e+1, epochs), "Loss: {:.4f}...".format(train_loss_pre))
            
            # at last step
            if e == epochs-1:
                # test_samples(D, G, inputs_real, inputs_z, session, data, gen, sample_range, batch_size, num_points=10000, num_bins=100)
                pre_like_data = DataDistribution(mu, sigma)
                samps = test_samples(d_model_pre, None, inputs_pre, None, sess, pre_like_data, None, data_range, 150)
                
                # copy the learned weights over into a tmp array
                weights_P = sess.run(pre_vars)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses, label='pre-trained')
    plt.title("Pre-Training Losses")
    plt.legend()

    plot_distributions(samps, data_range)

    return weights_P

def perform_training(net, epochs, pre_trained, batch_size):
    data = DataDistribution(net.mu, net.sigma)
    gen = GeneratorDistribution(net.data_range)

    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # copy weights from pre-training over to new D network
        for i, v in enumerate(net.d_vars):
            sess.run(v.assign(pre_trained[i]))

        for e in range(epochs):
            x_ = data.sample(batch_size * net.input_size)  # sampled m-batch from p_data
            x_ = np.reshape(x_, [batch_size, net.input_size])
            z_ = gen.sample(batch_size * net.z_size)  # sample m-batch from noise prior
            z_ = np.reshape(z_, [batch_size, net.z_size])

            # Run optimizers
            _ = sess.run(net.d_train_opt, feed_dict={net.inputs_real: x_, net.inputs_z: z_})
            _ = sess.run(net.g_train_opt, feed_dict={net.inputs_z: z_})

            # At the end of each epoch, get the losses and print them out
            train_loss_d = sess.run(net.d_loss, {net.inputs_real: x_, net.inputs_z: z_})
            train_loss_g = net.g_loss.eval({net.inputs_z: z_})

            # Save losses to view after training
            losses.append((train_loss_d, train_loss_g))

            if (e + 1) % 100 == 0:
                print("Epoch {}/{}...".format(e + 1, epochs),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}".format(train_loss_g))
            # save last training status for viewing
            if e == epochs - 1:
                samps = test_samples(net.d_model_real, net.g_model, net.inputs_real, net.inputs_z, sess, data, gen, net.data_range, batch_size)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()

    plot_distributions(samps, net.data_range)

def main():
    # Parameters
    mu, sigma = 1., 1.5
    data_range = 5
    learning_rate = 0.03
    batch_size = 150
    input_size = 1  # Size of input
    z_size = 1  # Size of latent vector to generator
    n_hidden = 32  # Sizes of hidden layers
    # alpha = 0.2
    smooth = 0.1  # Smoothing
    beta1 = 0.5
    pre_epochs = 100
    epochs = 1000

    pretrained = perform_pre_training(mu, sigma, data_range, input_size, learning_rate=0.03, beta1=beta1, epochs=pre_epochs)

    # def __init__(self, mu, sigma, data_range, input_size, z_size, n_hidden, learning_rate, smooth):
    net = Gaussian1dGAN(mu, sigma, data_range, input_size, z_size, n_hidden, learning_rate, smooth)

    perform_training(net, epochs, pretrained, batch_size)

if __name__ == '__main__':
    main()