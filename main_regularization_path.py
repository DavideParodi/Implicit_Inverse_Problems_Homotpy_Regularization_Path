import io
import os
import argparse
import time
import numpy as np 
#import autograd.numpy as np
#from autograd import grad
#import jax.numpy as np
#from jax import grad

import pandas

import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
plt.ion() 
from scipy.integrate import solve_ivp
import scipy

import sys
import math
import datetime


#Importo my libraries
import forward
import inverse
import plot


#System
parser = argparse.ArgumentParser('ODE learing')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='scipy_solver')
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)

#Plot
parser.add_argument('--plot', default=False) #if true it plot datas

#Problem Hyperparameters (hyperparameters which determine the problem)
parser.add_argument('--dimension', default=1) #Number of trajectories
parser.add_argument('--start_time', default=0.) #Start time
parser.add_argument('--end_time', default=6.) #End time 
parser.add_argument('--n_sample', default=100) #Time samples
parser.add_argument('--noise', default=True) #Noise on datas: if True there is noise
parser.add_argument('--perc_noise', default=np.array([1., 10., 20.])) #Quantity of noise on data

#Inverse Problem Algorithm hyperparameters (hyperparameters which determine the algorithm)
parser.add_argument('--regularizer', type=str, choices=['norm1', 'norm2'], default='norm1') #Type of regularization
parser.add_argument('--alpha_reg', default=0.) #Regularization Parameter: if 0 -> no regularization
parser.add_argument('--random_init', default=True) #If true a completely random parameters initialization is done
parser.add_argument('--interval_init', default=0.) #If random_init is True parameters are randlomly initialized between -interval_init, interval_init
parser.add_argument('--perc_from_gt', default=0.) #If random_init is False perturbation on infinite norm of groud truth parameters
parser.add_argument('--tau', default=10**(-3)) #Gradient Step Size
parser.add_argument('--n_iter_Adj', default=1000) #Number of Adjoint method iterations
parser.add_argument('--max_iter_NR', default=50) #Number of Newton-Raphson Iterations
parser.add_argument('--max_iter_LW', default=100) #N max iter Landweber
parser.add_argument('--max_iter_BT', default=100) #N max iter back-tracking
parser.add_argument('--armijo', default=False) #If true you use the Armijo rule to find the gradient step
parser.add_argument('--n_check_es', default=5) #Number of iterations to be checked for early stopping
parser.add_argument('--backtracking', default=False) #If true use backtracking on classical gradient step
parser.add_argument('--energy_factor', default=1.1) #Energy factor for controlling solution with respect data

args = parser.parse_args() 

#OUTPUT DIRECTORY 
directory_output = '/Users/davide/Università/Post_Laurea/Dottorato/M3/ODE_Learning_Codes/My_ODE_Learning/Graph_and_Results/regularization_path/risultati_prova_articolo/'
if not os.path.exists(directory_output):
    os.makedirs(directory_output)

#Additional Hyperparameters
alpha_reg_vect = np.logspace(0,-6,100)
epsilon = np.finfo(float).eps
n_prove = 3 #number of trials for each level of noise
c_i = np.array([0.2]) #Initial Condition

base_str = 'base_1'
directory_base = directory_output + 'tau=' + str(args.tau) + '/' + base_str + '/'
if not os.path.exists(directory_base):
    os.makedirs(directory_base)

dim_state_variable = args.dimension #number of trajectories
#create istance of class Phi_basis_function
Phi_basis_function = inverse.Phi_basis_function(dim_state_variable) #<- change basis in inverse 

#base 1: cos(u), cos(2u),...,cos(6u)
m_gt_1 = np.array([1., -1., 0., 0., 0., 0.]).T #gt_01
m_gt_2 = np.array([ -1.5, 1.5, -1.5, 1, -1, 0]).T #gt_01

m_gt_matrix = np.vstack([m_gt_1, m_gt_2])

for gt in np.arange(m_gt_matrix.shape[0])+1:
    m_gt = m_gt_matrix[gt-1]
    if gt<10:
        gt_str = 'gt_0' + str(gt)
    else:
        gt_str = 'gt_' + str(gt)
    
    directory_gt = directory_base + gt_str + '/'
    if not os.path.exists(directory_gt):
        os.makedirs(directory_gt)

    #forward clean data
    t = np.linspace(args.start_time, args.end_time, args.n_sample) #times
    if len(m_gt.shape)==1:
        m_gt = m_gt.reshape(m_gt.shape[0], 1)
    clean_data = forward.gen_truedata(t, m_gt, c_i, False, 0.)
    #save clean data
    filename = 'clean_data.npy'
    filepath = os.path.join(directory_gt, filename)
    np.save(filepath, clean_data)

    for perc_noise in args.perc_noise:
        directory_noise = directory_gt + '/noise=' + str(perc_noise) + '/'
        if not os.path.exists(directory_noise):
            os.makedirs(directory_noise)

        for prova in np.arange(n_prove)+1:
            if prova<10:
                str_prova = '0' + str(prova)
            else:
                str_prova = str(prova)

            directory_prova = directory_noise + 'prova_' + str_prova
            if not os.path.exists(directory_prova):
                os.makedirs(directory_prova)

            #noisy data
            t = np.linspace(args.start_time, args.end_time, args.n_sample) #tempi
            data = forward.gen_truedata(t, m_gt, c_i, args.noise, perc_noise)
            filename = 'noisy_data.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, data)
            t = t.reshape(t.shape[0], 1) #Reshape time vector: Tx1

            #Definition of data loss function depending on noisy data
            h_d = inverse.data_loss(data) 

            #Initialization
            #1) parameters
            m_init = inverse.init_parameters_random(args.interval_init, Phi_basis_function.dim_basis, dim_state_variable)
            m_init = m_init.reshape(m_init.shape[0],1)

            #2) solution
            init_sol = np.ones([args.n_sample, dim_state_variable]) * c_i #inizializzo soluzione a valore costante pari a init_guess: qui non serve BT
            v_init = np.zeros([args.n_sample, dim_state_variable])
            
            #3) adjoint state variables
            lambda_mul_init = np.zeros([args.n_sample, dim_state_variable])

            #1st cycle initializations
            m_0 = m_init
            u_sol_0 = init_sol
            v_0 = v_init
            lambda_mul_0 = lambda_mul_init
            iter_alpha = 0

            #varying regularization path
            m_regularization_path = np.zeros([len(alpha_reg_vect), Phi_basis_function.dim_basis, dim_state_variable]) #parameters
            u_sol_regularization_path = np.zeros([len(alpha_reg_vect), args.n_sample, dim_state_variable]) #solution
            main_loss_regularization_path = np.zeros([len(alpha_reg_vect), 1]) #main loss function
            data_loss_regularization_path = np.zeros([len(alpha_reg_vect), 1]) #data loss function
            g_regularization_path = np.zeros([len(alpha_reg_vect), 1]) #regularization term

            #REGULARIZATION PATH
            for alpha_reg in alpha_reg_vect:
                args.alpha_reg = alpha_reg
                # regularization 
                regularization = inverse.regularization(args.alpha_reg, args.regularizer)
                tau_iter = args.tau #gradient step
                loss = np.zeros([args.n_iter_Adj, 1]) #difference solution-data
                
                ##################################################################################################################################
                # Ciclo Adjoint (interno)
                # Step 1: Compute u solution with NR
                # Step 2: Computer lambda solution of Adjoint system
                # Step 3: Compute gradient Grad_Obj
                # Step 4: Update parameters m with gradient step

                # ADJOINT FOR CYCLE
                m = m_0 #initialization parameters from preview step
                u_sol = u_sol_0 #initialization solution from preview step
                loss[0] = np.linalg.norm(u_sol_0 - data) ** 2 #we need this for early stopping check
                v = v_0 
                lambda_mul = lambda_mul_0
                # Step 3: we start from here because steps 1 and 2 are already done
                Obj = inverse.compute_Obj(Phi_basis_function, data, u_sol_0, m_0, t, lambda_mul_0, regularization)
                Grad_Obj = inverse.compute_gradient(Phi_basis_function, u_sol_0, m_0, t, lambda_mul_0, regularization)

                for i in range(1, args.n_iter_Adj):

                    # Steps 4 and 1: Update parameters and back to step 1
                    m, u_sol, v, tau_iter = inverse.update_sol_and_param_NR(u_sol, v, m, tau_iter, Grad_Obj, c_i, t, Phi_basis_function, args.max_iter_NR, data, args.max_iter_LW, args.backtracking, 0, args.max_iter_BT, args.energy_factor, regularization)
                    loss[i] = np.linalg.norm(u_sol - data) ** 2
                    # Step 2
                    lambda_mul = inverse.adjoint_state_solver(u_sol, m, data, t, Phi_basis_function, lambda_mul, args.max_iter_LW)
                    
                    # Step 3
                    Obj = inverse.compute_Obj(Phi_basis_function, data, u_sol, m, t, lambda_mul, regularization)
                    Grad_Obj = inverse.compute_gradient(Phi_basis_function, u_sol, m, t, lambda_mul, regularization)

                    #print('Alpha_iter: ' + str(iter_alpha) + '. Iterazione: ' + str(i)) #control
                    
                    if inverse.early_stopping(loss, i, args.n_check_es):
                        #print('Early stopped: ' + str(i)) #control
                        break


                #Save obtained values and warm restart initialization
                m_0 = m
                u_sol_0 = u_sol
                v_0 = v
                lambda_mul_0 = lambda_mul

                m_regularization_path[iter_alpha, :, :] = m
                u_sol_regularization_path[iter_alpha, :, :] = u_sol
                main_loss_regularization_path[iter_alpha] = np.abs(Obj)
                data_loss_regularization_path[iter_alpha] = h_d.data_loss_function(u_sol)
                g_regularization_path[iter_alpha] = regularization.regularizer_function(m)
                iter_alpha = iter_alpha + 1
                print('iter alpha: '+str(iter_alpha))


            #Saving regularization paths
            #Parameters
            filename = 'm_regularization_path.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, m_regularization_path)

            #Solution
            filename = 'u_sol_regularization_path.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, u_sol_regularization_path)

            #Loss Function
            filename = 'main_loss_regularization_path.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, main_loss_regularization_path)

            #data Loss Function
            filename = 'data_loss_regularization_path.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, data_loss_regularization_path)

            #regularization term 
            filename = 'g_regularization_path.npy'
            filepath = os.path.join(directory_prova, filename)
            np.save(filepath, g_regularization_path)

            print('Finished:')
            print(gt_str + ', noise=' +str(perc_noise) +', prova=' + str_prova)
            
        print('Finished noise= '+str(perc_noise))












