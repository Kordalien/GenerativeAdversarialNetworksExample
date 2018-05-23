import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sin
from math import cos
import matplotlib.pyplot as plt
import gan

# We're going to try to learn the distribution of values in an arbitrary function between 0 and 1.
# This means G will be generating a sample from that distribution, and D will try to guess if the generated sample came from the function
# or from G.
# Then we'll repeat the expirmeent, but instead G will be trained to take N samples, and produce N samples from the distribution
#And see how the entworks perform.
two_pi = 2*3.14159
func_to_learn =  np.vectorize(lambda x: 5*sin(x*two_pi)+3*cos(x*2*two_pi)+2*cos(x/2*two_pi))
#func_to_learn =  np.vectorize(lambda x: 5*sin(x*two_pi))

epochs = 10000
experiment_1_in = 1
experiment_2_in = 1000
experiment_3_in = 200

#Expiriment 1: the generator and discriminator each act on 1 sample from the distribution
discriminator_exp_1 = nn.Sequential(
    nn.Linear(experiment_1_in, 100),
    nn.ELU(),
    nn.Linear(100,50),
    nn.ELU(),
    nn.Linear(50,25),
    nn.ELU(),
    nn.Linear(25,1),
    nn.Sigmoid()
)
generator_exp_1 = nn.Sequential(
    nn.Linear(experiment_1_in, 200),
    nn.Sigmoid(),
    nn.Linear(200, 200),
    nn.ELU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200, experiment_1_in)
)

#Expiriment 2: the genrator and discriminator work against a set of samples from the distribution
discriminator_exp_2 = nn.Sequential(
    nn.Linear(experiment_2_in, 100),
    nn.ELU(),
    nn.Linear(100,500),
    nn.ELU(),
    nn.Linear(500,25),
    nn.ELU(),
    nn.Linear(25,1),
    nn.Sigmoid()
)
generator_exp_2 = nn.Sequential(
    nn.Linear(experiment_2_in, 200),
    nn.Sigmoid(),
    nn.Linear(200, 200),
    nn.ELU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200, experiment_2_in)
)

#Using the same discriminator, train a generator which does a 1 to 1 mapping (1 input to 1 output) but modify it so that it applies itself to distribution number of samples. (e.g. we apply g 1 sample at a time to 200 samples...
class GeneratorExp3(nn.Module):
    def __init__(self):
        super(GeneratorExp3, self).__init__()

        self.model =  nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    #Input will be of the form [n, 200], the coputational model is [200n] (one at a time) the output is [n,200]
    # we want to execute on 
    def forward(self, z):
        (rows,cols) = z.shape
        return self.model(z.view(rows*cols,1)).view(rows,cols)

discriminator_exp_3 = nn.Sequential(
    nn.Linear(experiment_3_in, 100),
    nn.ELU(),
    nn.Linear(100,50),
    nn.ELU(),
    nn.Linear(50,25),
    nn.ELU(),
    nn.Linear(25,1),
    nn.Sigmoid()
)

generator_exp_3 = GeneratorExp3()

#Learning rates
g_eta = 1e-4
d_eta = 1e-4

'''
Utility method which is used to train a gan using a prespecified trainer and automating wiring for things like epochs.
'''
def run_expiriment(generator, discriminator, samplers, minibatch, sampler):
    d_optimizer = optim.Adam(discriminator.parameters(), lr = d_eta, betas=(0.9,0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr = g_eta, betas=(0.9,0.999))
    gan.train(
        (generator, 1, g_optimizer),
        (discriminator, 2, d_optimizer),
        samplers, minibatch, epochs,
        sampler
    )

'''
Sampling:
To collect some information about the distribution we're finding, we sample it.
Mean/variance tell us a little bit. It's possible we can describe a better sampler by say,
taking an oodle of samples and subtracting the 
'''
def stats_gen(generator, z_sampler, samples_to_take):
    z_samples =  torch.tensor(np.stack([z_sampler() for x in range(samples_to_take)],0), dtype = torch.float)
    samples  = generator(z_samples).detach().numpy()
    samples = samples.reshape(np.prod(a.shape))
    mean = np.mean(samples)
    variance = np.var(samples)
    
    
    
#Expiriment 1 needs 1 sample from each so:
print("begin exp 1")
run_expiriment(generator_exp_1,
               discriminator_exp_1,
               (lambda : func_to_learn(np.random.rand(experiment_1_in)),
                lambda : np.random.rand(experiment_1_in)),
               50,
               (500, lambda epoch, sample_n: print(epoch))
)
'''
For exp 1 we should be tracking a bnch of stuff...
Distribution of values in G over time as:
average, mean, median, variance,
'''
print("begin exp 2")
run_expiriment(generator_exp_2,
               discriminator_exp_2,
               (lambda : func_to_learn(np.random.rand(experiment_2_in)),
                lambda : np.random.rand(experiment_2_in)),
               4,
               (500, lambda epoch, sample_n: print("Epoch {}".format(epoch)))
)
print("begin exp 3")
run_expiriment(generator_exp_3,
               discriminator_exp_3,
               (lambda : func_to_learn(np.random.rand(experiment_3_in)),
                lambda : np.random.rand(experiment_3_in)),
               1,
               (500, lambda epoch, sample_n: print("Epoch {}".format(epoch)))
)


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3) 
detailed_x_samples = np.arange(0,1, .02) 
y_samples = func_to_learn(detailed_x_samples)
exp_1_samples = generator_exp_1(torch.tensor(detailed_x_samples, dtype = torch.float).view(len(detailed_x_samples),1)).detach().numpy() 
exp_3_samples = generator_exp_3(torch.tensor(detailed_x_samples, dtype = torch.float).view(len(detailed_x_samples),1)).detach().numpy() 
ax1.set_xlabel('X') 
ax1.set_ylabel('Y') 
ax1.set_title('F(x) vs G[enerated](x)s') 
ax1.scatter(detailed_x_samples,y_samples, label = 'real', alpha = 0.5) 
ax1.scatter(detailed_x_samples,exp_1_samples, label = 'generated exp 1', alpha = 0.5)
ax1.scatter(detailed_x_samples,exp_3_samples, label = 'generated exp 3' , c = 'r', alpha =0.5)

hist_samples_to_draw = 20000
data_values = func_to_learn(np.random.rand(hist_samples_to_draw)) 
gen_values_exp_1 = generator_exp_1(torch.tensor(np.random.rand(hist_samples_to_draw,1), dtype=torch.float)).detach().numpy() 
gen_values_exp_2 = generator_exp_2(torch.tensor(np.random.rand(hist_samples_to_draw//experiment_2_in,experiment_2_in), dtype=torch.float)).detach().numpy().reshape((hist_samples_to_draw//experiment_2_in)*experiment_2_in,1) 
gen_values_exp_3 = generator_exp_3(torch.tensor(np.random.rand(hist_samples_to_draw,1), dtype=torch.float)).detach().numpy() 
data = np.column_stack((data_values,gen_values_exp_1, gen_values_exp_2, gen_values_exp_3))

ax2.hist(data_values, np.arange(data.min(), data.max(), 0.1), label ='real', alpha = 0.5, density = True) 
ax2.hist(gen_values_exp_1, np.arange(data.min(), data.max(), 0.1), label ='generated exp 1', alpha = 0.5, density = True) 
ax2.hist(gen_values_exp_2, np.arange(data.min(), data.max(), 0.1), label ='generated exp 2', alpha = 0.5, density = True) 
ax2.hist(gen_values_exp_3, np.arange(data.min(), data.max(), 0.1), label ='generated exp 3', alpha = 0.5, density = True) 
ax2.set_title('Relative Frequency values') 
ax2.set_xlabel('Sampled Value') 
ax2.set_ylabel('Frquency') 


ax3.hist(data, histtype='barstacked', label = ('real', 'generated exp 1', 'generated exp 2', 'generated exp 3'), alpha = 0.5, density = True)
ax3.set_title('Stacked Relative Frequency values') 
ax3.set_xlabel('Sampled Value') 
ax3.set_ylabel('Frquency')

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center')


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3) 
ax1.set_xlabel('X') 
ax1.set_ylabel('Y') 
ax1.set_title('F(x) vs G[enerated](x)s') 
ax1.scatter(detailed_x_samples,y_samples, label = 'real', alpha = 0.5) 
ax1.scatter(detailed_x_samples,exp_3_samples, label = 'generated exp 3', c ='g',  alpha =0.5)

data = np.column_stack((data_values, gen_values_exp_2, gen_values_exp_3))
ax2.hist(data_values, np.arange(data.min(), data.max(), 0.1), label ='real', alpha = 0.5, density = True) 
ax2.hist(gen_values_exp_2, np.arange(data.min(), data.max(), 0.1), label ='generated exp 2', alpha = 0.5, density = True) 
ax2.hist(gen_values_exp_3, np.arange(data.min(), data.max(), 0.1), label ='generated exp 3', alpha = 0.5, density = True) 
ax2.set_title('Relative Frequency values') 
ax2.set_xlabel('Sampled Value') 
ax2.set_ylabel('Frquency') 


ax3.hist(data, histtype='barstacked', label = ('real', 'generated exp 2', 'generated exp 3'), alpha = 0.5, density = True)
ax3.set_title('Stacked Relative Frequency values') 
ax3.set_xlabel('Sampled Value') 
ax3.set_ylabel('Frquency')

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center')
plt.show() 
