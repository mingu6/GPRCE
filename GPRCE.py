# TO DO: Vectorise w.r.t. x
# move to CUDA
# visualisation tools

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm       # progress bar


def main():
    n_samples = 5
    n_layers = 50
    width = 10000
    nt = 100

    # start rotating from (0, 1) on unit circle
    th_plot = np.linspace(0, np.pi * 2, nt)
    x_plot = np.zeros([nt, 2])
    for i in range(nt):
        th = th_plot[i]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        x = np.matmul(R, np.array([1, 0]))
        x_plot[i] = x

    # indep normal prior
    nn = RCENet(F, n_layers=n_layers, width=width, activation='relu', n_samples=n_samples)
    # evaluate f(x) for samples
    f = nn.generate_samples(x_plot) 

    plt.plot(th_plot, f)
    plt.title('Neal prior')
    plt.xlabel('theta')
    plt.ylabel('f(theta)')
    plt.show()

    # funky prior
    nn1 = RCENet(F1, n_layers=n_layers, width=width, activation='relu', n_samples=n_samples)
    # evaluate f(x) for samples
    f1 = nn1.generate_samples(x_plot) 

    plt.plot(th_plot, f1)
    plt.title('RCE prior')
    plt.xlabel('theta')
    plt.ylabel('f(theta)')
    plt.show()

# generate F(A, B, C, D) for collection of samples from A, B, C, D
# input: an array of samples from A (float) B (col vec) C (row vec) D (matrix)
# out: samples from RCE prior W = F(A, B, C, D)

# Gaussian prior
def F(A, B, C, D):
    # extract no. of rows and cols in W from D
    nrows, ncols = D.shape[1], D.shape[2]
    # tile A to fill W for vectorised operations
    A = A.repeat(nrows, ncols, 1).permute(2, 0, 1)
    # tile B
    B = B.repeat(1, ncols).view(-1, ncols, nrows).permute(0, 2, 1)
    # tile C
    C = C.repeat(1, nrows).view(-1, nrows, ncols)
    #F = 1 / 4 * torch.FloatTensor(norm.ppf(A) + norm.ppf(B) + norm.ppf(C) + norm.ppf(D))
    F = torch.FloatTensor(norm.ppf(D))
    return F

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
    F = torch.FloatTensor(-0 * A.numpy() + norm.ppf(B) * norm.ppf(C) + norm.ppf(D))
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
        A = torch.rand(n_samples)
        B = torch.rand(n_samples, nrows)
        C = torch.rand(n_samples, ncols)
        D = torch.rand(n_samples, nrows, ncols)
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
        # initialise input into network
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
                Wsamples = 1 / np.sqrt(width) * torch.distributions.Normal(0, 1).sample([n_samples, nrows, ncols])
            else:
                Wsamples = 1 / np.sqrt(width) * torch.distributions.Normal(0, 1).sample([n_samples, nrows, ncols])
            
            for j in range(len(x)):
                # tile x for evaluation of f(x) for all samples (vectorised)
                if i == 0:
                    x_temp = torch.FloatTensor(np.atleast_1d(x[j])).repeat(n_samples).view(n_samples, ncols, 1)
                else:
                    x_temp = x_temp_list[j]
                #if j == 0:
                #    print(Wsamples[0].shape, x_temp[0].shape)
                # apply linear transform from W
                x_temp = torch.matmul(Wsamples, x_temp)
                # apply activation
                x_temp = self.phi(x_temp)
                x_temp_list[j] = x_temp
                
        # apply linear transform to final layer output to turn into scalar
        Wsamples = 1 / np.sqrt(width) * self.W_samples(1, width)
        x_out = torch.zeros([len(x), n_samples])
        for i in range(len(x)):
            x_out[i] = torch.matmul(Wsamples, x_temp_list[i]).view(n_samples)
        return x_out.numpy()

if __name__ == "__main__":
    main()