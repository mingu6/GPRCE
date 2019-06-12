# TO DO: Vectorise w.r.t. x
# move to CUDA
# visualisation tools

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm       # progress bar
#from torch.utils.tensorboard import SummaryWriter


def main():
    n_samples = 5
    n_layers = 50
    width = [10, 50, 100, 200, 500, 1000]
    nt = 100

    ylim = [-2, 2]

    #writer = SummaryWriter()

    # start rotating from (0, 1) on unit circle
    th_plot = np.linspace(0, np.pi * 2, nt)
    x_plot = np.zeros([nt, 2])
    for i in range(nt):
        th = th_plot[i]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        x = np.matmul(R, np.array([1, 0]))
        x_plot[i] = x

    fig, axes = plt.subplots(nrows=2, ncols=len(width))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('No. of layers = ' + str(n_layers))
    axesf = axes.flatten()

    for i in tqdm(range(len(width))):
        w = width[i]
        # indep normal prior
        nn = RCENet(F, n_layers=n_layers, width=w, activation='relu', n_samples=n_samples)
        # evaluate f(x) for samples
        f = nn.generate_samples(x_plot) 

        ax = axesf[i] 
        ax.plot(th_plot, f)
        ax.set_ylabel(r'$NN(\theta)$')
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_title('Gaussian - width ' + str(w))
        #ax.set_ylabel(r'$f(\theta)$')

        # funky prior
        nn1 = RCENet(F1, n_layers=n_layers, width=w, activation='relu', n_samples=n_samples)
        # evaluate f(x) for samples
        f1 = nn1.generate_samples(x_plot) 
        
        ax1 = axesf[len(width) + i]
        ax1.plot(th_plot, f1)
        ax1.set_ylabel(r'$NN(\theta)$')
        ax1.set_xlabel(r'$\theta$')
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_title('RCE - width ' + str(w))
        #ax1.set_xlabel(r'$\theta$')
        #ax1.set_ylabel(r'$f(\theta)$')
    #writer.add_figure('plot', fig)
    #writer.close()
    plt.show()

# generate F(A, B, C, D) for collection of samples from A, B, C, D
# input: an array of samples from A (float) B (col vec) C (row vec) D (matrix)
# out: samples from RCE prior W = F(A, B, C, D)

# Gaussian prior, takes input of uniform RVs, generates iid standard Gaussian weights
def F(A, B, C, D):
    # extract no. of rows and cols in W from D
    nrows, ncols = D.shape[1], D.shape[2]
    # tile A to fill W for vectorised operations
    A = A.repeat(nrows, ncols, 1).permute(2, 0, 1)
    # tile B
    B = B.repeat(1, ncols).view(-1, ncols, nrows).permute(0, 2, 1)
    # tile C
    C = C.repeat(1, nrows).view(-1, nrows, ncols)
    #F = torch.FloatTensor(norm.ppf(D))
    return D 
    #return F

# RCE prior
def F1(A, B, C, D):
    # extract no. of rows and cols in W from D
    nrows, ncols = D.shape[1], D.shape[2]
    # tile A to fill W for vectorised operations
    A = A.repeat(nrows, ncols, 1).permute(2, 0, 1)
    # tile B
    B = B.repeat(1, ncols).view(-1, ncols, nrows).permute(0, 2, 1)
    # tile C
    C = C.repeat(1, nrows).view(-1, nrows, ncols)
    #F = 0.5 * (torch.FloatTensor(- A.numpy() + 2 * norm.ppf(B) * norm.ppf(C) + 3 * norm.ppf(D)))
    #F = torch.FloatTensor(- A + 2 * B * C + 3 * D) # A, B, C, D are generated as normal r.v.s instead of inverting normal cdf from uniform for numerical stability 
    F = D - 0.01
    #F = torch.FloatTensor(norm.ppf(D))
    return F
    

class RCENet:
    def __init__(self, F, n_layers=1, width=10, activation='tanh', n_samples=100):
        # initialise activation function
        if activation == 'tanh':
            self.phi = torch.tanh
        elif activation == 'relu':
            self.phi = torch.relu
        else:
            raise Exception('youre a towel')
        self.n_layers = n_layers
        self.width = width
        self.n_samples = n_samples
        self.F = F
        
    # generate samples from W given dimensions
    def W_samples(self, nrows, ncols):    
        n_samples = self.n_samples
        #A = torch.rand(n_samples)
        #B = torch.rand(n_samples, nrows)
        #C = torch.rand(n_samples, ncols)
        #D = torch.rand(n_samples, nrows, ncols)
        A = torch.randn(n_samples)
        B = torch.randn(n_samples, nrows)
        C = torch.randn(n_samples, ncols)
        D = torch.randn(n_samples, nrows, ncols)
        return self.F(A, B, C, D)
    
    # for a vector of x, generate samples from f(x) assuming RCE distn on weights
    # in: x -> list of x values
    # out: n_samples of f(x)
    ### TO DO: vectorise this!!!
    def generate_samples(self, x):
        # init parameters
        n_samples = self.n_samples
        n_layers = self.n_layers
        width = self.width
        # list stores values of outputs at each layer
        x_temp_list = torch.zeros([len(x), n_samples, width, 1])
        # process layers sequentially, edge cases for final and init layers
        for i in tqdm(range(n_layers)):
            if i == 0:
                nrows = width
                if len(x.shape) > 1:
                    ncols = x.shape[1]
                else:
                    ncols = 1
            else:
                nrows = width
                ncols = width
            # for each layer sample A, B, C, D to yield W_i
            # find Wx for each x -> must use SAME WEIGHT VECTOR for each x, so eval W and then all x
            if i == 0:
                Wsamples = 1 / np.sqrt(len(x) / 2) * torch.distributions.Normal(0, 1).sample([n_samples, nrows, ncols])
            else:
                Wsamples = 1 / np.sqrt(width / 2) * self.W_samples(width, width)
            
            # iterate over values of x
            for j in range(len(x)):
                # tile x for evaluation of f(x) for all samples (vectorised)
                if i == 0:
                    x_temp = torch.FloatTensor(np.atleast_1d(x[j])).repeat(n_samples).view(n_samples, ncols, 1) # initial value of x is provided
                else:
                    x_temp = x_temp_list[j] # otherwise, take previous layer output
                # apply linear transform from W
                x_temp = torch.matmul(Wsamples, x_temp)
                # apply activation
                x_temp = self.phi(x_temp)
                x_temp_list[j] = x_temp

        # last layer processing (linear transformation only)        
        # apply linear transform to final layer output to turn into scalar
        Wsamples = 1 / np.sqrt(width / 2) * self.W_samples(1, width)
        x_out = torch.zeros([len(x), n_samples])
        for i in range(len(x)):
            x_out[i] = torch.matmul(Wsamples, x_temp_list[i]).view(n_samples)
        return x_out.numpy()

if __name__ == "__main__":
    main()