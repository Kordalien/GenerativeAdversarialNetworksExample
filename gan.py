import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sin
from math import cos
import matplotlib.pyplot as plt


#This is the function that we're trying to learn to approximate
# A gan learns to generate Output which is associated with a distribution, so in ou case
# this is equivalent to the histogram of values from this function. E.g. if you sample this function with
# the uniform noise prior Z, certain values occur with a given frequency. The Gan will be learning to emulat those
# value frequencies.
two_pi = 2*3.14159
func_to_learn =  np.vectorize(lambda x: 5*sin(x*two_pi)+3*cos(x*2*two_pi)+2*cos(x/2*two_pi))
#func_to_learn =  np.vectorize(lambda x: 5*sin(x*two_pi))

epochs = 10000
print_every_n = 200

# Because we're trying to learn a function f: Double => Double, we need 1 input
# Then we'll need arbitrary hidden units
# and 1 output:

g_in = 1
g_hidden = 200
g_out = 1

# We need 1 output (p_real)
# Note that the discriminator a set of generated inputs and decides about them, so d_in controls the number of samples it considers.

d_in = 200
d_hidden = 100
d_out = 1

#Dsicriminator training rounds per training round
d_steps = 2
#Generator training rounds per training round
g_steps = 1

#Learning rates
g_eta = 1e-4
d_eta = 1e-4

#Models
generator = nn.Sequential(
    nn.Linear(g_in, g_hidden),
    nn.Sigmoid(),
    nn.Linear(g_hidden, g_hidden),
    nn.ELU(),
    nn.Linear(g_hidden,g_hidden),
    nn.ReLU(),
    nn.Linear(g_hidden, g_out)
)

discriminator = nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.ELU(),
    nn.Linear(d_hidden,d_hidden//2),
    nn.ELU(),
    nn.Linear(d_hidden//2,d_hidden//4),
    nn.ELU(),
    nn.Linear(d_hidden//4,d_out),
    nn.Sigmoid()
)

d_optimizer = optim.Adam(discriminator.parameters(), lr = d_eta, betas=(0.9,0.999))
g_optimizer = optim.Adam(generator.parameters(), lr = g_eta, betas=(0.9,0.999))
#Accroding to the paper, training generator is just BCE where everyone is in class 0
criterion = nn.BCELoss() #just the standard loss function proposed by the paper.

samples = [None] * int(epochs/print_every_n)
x_samples = np.arange(0,1, .1) # take 10 samples over the functions range


print('start training')
#Begin trianing
for epoch in range(epochs):
    if epoch % print_every_n == 0:
        print('training round {}'.format(epoch))
    #Update the discriminator
    for d_step in range(d_steps):
        #Run disciminiator with values from generator
        r_samples = torch.tensor(func_to_learn(np.random.rand(1,d_in)), dtype = torch.float)
        r_scores = discriminator(r_samples)
        #Run disciminiator with values from data
        g_samples = generator(torch.tensor(np.random.rand(d_in,g_in), dtype = torch.float))
        #g_samples = generator(torch.randn(100,1))
        g_scores = discriminator(g_samples.t()) # note the transpose, since the output is a column instead of row vector...

        #Train D using the rule: Loss = Sum_i=1^n log(D(x^i)) + log(1-d(G(Z^i))
        r_loss = criterion(r_scores, torch.ones(1))
        g_loss = criterion(g_scores, torch.zeros(1))    

        if epoch % print_every_n == 0:
            print("r_loss: {}, g_loss: {}".format(r_loss, g_loss))
        #Clear the gradient so we can perform the backwards pass:
        d_optimizer.zero_grad()
        
        #Compute the backpropogation of losses with respect to the classes 
        r_loss.backward()
        g_loss.backward()

        #update the network
        d_optimizer.step()
    #Update the generator        
    for g_step in range(g_steps):
        g_optimizer.zero_grad()

        #Sample a batch
        g_samples = generator(torch.tensor(np.random.rand(d_in,g_in), dtype = torch.float))
        g_scores = discriminator(g_samples.t())
        #rank it by the discriminators standards
        g_loss = criterion(g_scores, torch.ones(1)) #Note the paper uses a descending optimzier here, we swap that for the ascending case which computes the same request, but wihtout returning the optimizer
        if epoch % print_every_n == 0:
            print("loss in the generator: {}".format(g_loss))
        #and optimize under G.

        g_loss.backward()
        
        g_optimizer.step()
    if epoch % print_every_n == 0:
        samples[int(epoch/print_every_n)] = generator(torch.tensor(x_samples, dtype = torch.float).view(len(x_samples),1)).detach().numpy()

def plot():
    print('start plotting')
    #For now allocate 4 plots
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)

    # Generate Plot 0: the function to learn
    # take 50 samples over the functions range
    detailed_x_samples = np.arange(0,1, .02)
    y_samples = func_to_learn(detailed_x_samples)
    g_y_samples = generator(torch.tensor(detailed_x_samples, dtype = torch.float).view(len(detailed_x_samples),1)).detach().numpy()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('F(x) vs G[enerated](x)')
    ax1.scatter(detailed_x_samples,y_samples)
    ax1.scatter(detailed_x_samples,g_y_samples)
    # Generate Plot #1: Histogram of values generated by our transforming function of the range of values we care about, [0,1)
    data_values = func_to_learn(np.random.rand(5000))
    gen_values = generator(torch.tensor(np.random.rand(5000,1), dtype=torch.float)).detach().numpy()
    data = np.column_stack((data_values,gen_values))
    ax2.hist(data, np.arange(data.min(), data.max(), 0.1), label =('real','generated'))
    ax2.legend()
    ax2.set_title('Relative Frequency values')
    ax2.set_xlabel('Sampled Value')
    ax2.set_ylabel('Frquency')
    #show the evoloution of the generator
    for sample in samples:
        ax3.scatter(x_samples,sample)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('G(x) over time')
    plt.show()

plot()
