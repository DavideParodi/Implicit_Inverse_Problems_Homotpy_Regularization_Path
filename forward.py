import os
import argparse
import time
import matplotlib.pyplot as plt
#import matplotlib
from scipy.integrate import solve_ivp
import scipy
import numpy as np
# import autograd.numpy as np
# from autograd import grad
#import jax.numpy as np
#from jax import grad
#import torch
import math


#Import of basis function from the inverse library
from inverse import Phi_basis_function

def eqdiff(t, u, Phi, m):
    dim_state_variable = u.shape[0]
    dudt = np.zeros(dim_state_variable)
    for k in range(0, dim_state_variable):
        for j in range(0, Phi.dim_basis):
                dudt[k] = dudt[k] + Phi.basis_functions[j](u) * m[j,k]
    return dudt

def gen_truedata(t, m, c_i, noise, perc_noise):
    #dimensioni soluzione
    n_sample = t.shape[0]
    m_sizes = m.shape
    dim_state_variable = m_sizes[1] #numero di traiettorie
    #Generazione istanza della classe
    Phi = Phi_basis_function(dim_state_variable)
    #Calcolo soluzione in avanti
    d_sol = solve_ivp(lambda t, u, Phi, m: eqdiff(t, u, Phi, m), [t[0], t[-1]], c_i, t_eval=t, args=(Phi, m), method='LSODA')
    d = d_sol.y.T
    #Aggiunta del noise
    d_noise = create_noisy_data(d, noise, perc_noise)
    return d_noise

def create_noisy_data(d, noise, perc_noise):
    #Input:
    #d: clean data
    #noise: true or false (true means there is noise)
    #perc_noise: percentage of noise
    if noise:
        #d_noise = d + perc_noise * np.random.normal(0, perc_noise)
        d_noise = d + np.random.normal(np.zeros(d.shape), perc_noise/100)
    else:
        d_noise = d

    return d_noise
