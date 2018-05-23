import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sin
from math import cos
import matplotlib.pyplot as plt


'''
Executes training under the generative adversarial framework.
generator and discriminator are tuples containing a neural network, the number of rounds to train per epoch, and the optimizer to train with.
f_sampler and z_sampler are functions which sample f, the unknown function, and z, the noise prior that will be used to generate samples.

Let x be a sample drawn from the space (i.e. an image or a value from a distribution or w/e)
Then f_sampler should return 1 value of x,
while z_sampler should return 1 value of z, which is the vector which can be passed to g.

the discriminator should accept individual values of x and make a ruling on them.
'''
def train(generator, discriminator, samplers, minibatch_size, epochs, sampler=None):
    (g, g_steps, g_opt) = generator
    (d, d_steps, d_opt) = discriminator
    (f_sampler, z_sampler) = samplers
    real = torch.ones(minibatch_size)
    fake = torch.zeros(minibatch_size)
    if sampler:
        (sample_rate, callback) = sampler
        
    criterion = nn.BCELoss() # standard loss as proposed by the paper
    
    for epoch in range(epochs):
        for d_step in range(d_steps):
            #Draw minibatch real samples from the system 
            real_samples = torch.tensor(np.stack([f_sampler() for x in range(minibatch_size)],0), dtype = torch.float)
            real_scores = d(real_samples)

            #draw minibatch samples from z
            priors = torch.tensor(np.stack([z_sampler() for x in range(minibatch_size)], 0), dtype = torch.float)
            #and convert them to minibatch samples from x.
            generated_samples = g(priors)
            generated_scores = d(generated_samples)

            #figure out how well the discriminator did
            real_loss = criterion(real_scores.view(minibatch_size), real)
            generated_loss = criterion(generated_scores.view(minibatch_size), fake)

            #and train it
            d_opt.zero_grad()
            real_loss.backward()
            generated_loss.backward()
            d_opt.step()
            
        for g_step in range(g_steps):
            #draw minibatch samples from z
            priors = torch.tensor(np.stack([z_sampler() for x in range(minibatch_size)], 0), dtype = torch.float)
            #and convert them to minibatch samples from x.
            generated_samples = g(priors)
            generated_scores = d(generated_samples)

            #Calculate the entropy as if these values were real
            #Maximizing this value === minimizing the value if they were false
            generated_loss = criterion(generated_scores.view(minibatch_size), real)

            g_opt.zero_grad()
            generated_loss.backward()
            g_opt.step()

        if(sampler and epoch % sample_rate == 0):
            callback(epoch, epoch//sample_rate)
            

