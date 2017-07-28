import torch
from torch.nn import init
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt 
import numpy as np 


def my_weight_init(m):
    # classname = m.__class__.__name__
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight.data)
        # torch.nn.init.constant(m.bias.data, 0)
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)

class Generator(torch.nn.Module):
    def __init__(self, input_size, n_hidden=128, n_output=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.fc1 = torch.nn.Linear(input_size, n_hidden, bias=True)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, n_hidden//4, bias=True)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, n_output, bias=True)

        for m in self.modules():
            my_weight_init(m)
            

    def forward(self, input):
        out = torch.nn.functional.leaky_relu(self.fc1(input), negative_slope=self.alpha)
        out = torch.nn.functional.leaky_relu(self.fc2(out), negative_slope=self.alpha)
        out = torch.nn.functional.tanh(self.fc3(out))

        return out

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, n_hidden=128, n_output=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.fc1 = torch.nn.Linear(input_size, n_hidden, bias=True)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, n_hidden//4, bias=True)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, n_output, bias=True)

        for m in self.modules():
            my_weight_init(m)

    def forward(self, input):
        out = torch.nn.functional.leaky_relu(self.fc1(input), negative_slope=self.alpha)
        out = torch.nn.functional.leaky_relu(self.fc2(out), negative_slope=self.alpha)
        out = torch.nn.functional.sigmoid(self.fc3(out))

        return out


'''
Parameters
'''
x_size = 28 * 28
z_size = 100
n_hidden = 128
# n_classes = 10
epochs = 200
batch_size = 128
learning_rate = 0.0002
alpha = 0.2
beta1 = 0.5
print_every = 50

# data_loader normalize [0, 1] ==> [-1, 1]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../Data_sets/MNIST_data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# build network
G = Generator(z_size, n_hidden=n_hidden, n_output=x_size, alpha=alpha)
D = Discriminator(x_size, n_hidden=n_hidden, n_output=1, alpha=alpha)
G.cuda()
D.cuda()

# optimizer
BCE_loss = torch.nn.BCELoss()
G_opt = torch.optim.Adam( G.parameters(), lr=learning_rate, betas=[beta1, 0.999] )
D_opt = torch.optim.Adam( D.parameters(), lr=learning_rate, betas=[beta1, 0.999] )

'''
Start training
'''
step = 0
samples = []
losses = []
for e in range(epochs):
    for x_, _ in train_loader:
        step += 1
        '''
        Train in Discriminator
        '''
        # reshape input image
        x_ = x_.view(-1, x_size)
        current_batch_size = x_.size()[0]

        # create labels for loss computation
        y_real_ = torch.ones(current_batch_size)
        y_fake_ = torch.zeros(current_batch_size)

        # make it cuda Tensor
        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        # run real input on Discriminator
        D_result_real = D(x_)
        D_loss_real = BCE_loss(D_result_real, y_real_)

        # run Generator input on Discriminator
        z1_ = torch.Tensor(current_batch_size, z_size).uniform_(-1, 1)
        z1_ = Variable(z1_.cuda())
        x_fake = G(z1_)
        D_result_fake = D(x_fake)
        D_loss_fake = BCE_loss(D_result_fake, y_fake_)

        D_loss = D_loss_real + D_loss_fake

        # optimize Discriminator
        D.zero_grad()
        D_loss.backward()
        D_opt.step()
        
        '''
        Train in Generator
        '''
        z2_ = torch.Tensor(current_batch_size, z_size).uniform_(-1, 1)
        y_ = torch.ones(current_batch_size)
        z2_, y_ = Variable(z2_.cuda()), Variable(y_.cuda())
        G_result = G(z2_)
        D_result_fake2 = D(G_result)
        G_loss = BCE_loss(D_result_fake2, y_)

        G.zero_grad()
        G_loss.backward()
        G_opt.step()

        if step % print_every == 0:
            losses.append((D_loss.data[0], G_loss.data[0]))

            print("Epoch {}/{}...".format(e+1, epochs),
                "Discriminator Loss: {:.4f}...".format(D_loss.data[0]),
                "Generator Loss: {:.4f}".format(G_loss.data[0])) 
    # Sample from generator as we're training for viewing afterwards
    sample_z = torch.Tensor(16, z_size).uniform_(-1, 1)
    sample_z = Variable(sample_z.cuda())
    gen_samples = G(sample_z)
    current_epoch_samples = []
    for k in range(16):
        current_epoch_samples.append(gen_samples[k, :].cpu().data.numpy())
    samples.append(current_epoch_samples)

fig1, ax1 = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes

samples = np.array(samples)
print(samples.shape)
fig2, axes2 = view_samples(-1, samples)

rows, cols = 10, 6
fig3, axes3 = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes3):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

plt.show('hold')