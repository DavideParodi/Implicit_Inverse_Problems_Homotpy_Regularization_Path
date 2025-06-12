

import io
import os
import argparse
import time
import numpy as np 
#import autograd.numpy as np
#from autograd import grad
#import jax.numpy as np
#from jax import grad


#import torch
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
plt.ion() 
from scipy.integrate import solve_ivp
import scipy

from matplotlib import cm
from matplotlib.colors import Normalize

import sys
import math
import datetime

#Importo le mie librerie
import forward
import inverse
import plot



#Definizione Args
#System
parser = argparse.ArgumentParser('ODE learing')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='scipy_solver')
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)

#Plot
parser.add_argument('--plot', default=False) #if true it plot datas

#Problem Hyperparameters (iperparametri che determinano il problema)
parser.add_argument('--dimension', default=1) #Number of trajectories
parser.add_argument('--start_time', default=0.) #Start time
parser.add_argument('--end_time', default=6.) #End time rimettere 6
parser.add_argument('--n_sample', default=100) #Time samples, rimettere 100
parser.add_argument('--noise', default=True) #Noise on datas: if True there is noise
parser.add_argument('--perc_noise', default=np.array([1., 10., 20.])) #Quantity of noise on data

#Inverse Problem Algorithm hyperparameters (Iperparametri che determinano l'algoritmo)
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

directory_input = '/Users/davide/Università/Post_Laurea/Dottorato/M3/ODE_Learning_Codes/My_ODE_Learning/Graph_and_Results/regularization_path/risultati_prova_articolo/'
directory_output = '/Users/davide/Università/Post_Laurea/Dottorato/M3/ODE_Learning_Codes/My_ODE_Learning/Graph_and_Results/regularization_path/risultati_prova_articolo/Generate_Graphs/'

n_prove = 3
dim_state_variable = args.dimension 
t = np.linspace(args.start_time, args.end_time, args.n_sample)
c_i = np.array([0.2])

base_str = 'base_1'
tau_str = str(args.tau)
#Esperimenti
directory_input_general = directory_input + 'tau=' + tau_str + '/' + base_str + '/'
#Output cartella
directory_output_general = directory_output + 'tau=' + tau_str + '/' + base_str + '/'
if not os.path.exists(directory_output_general):
    os.makedirs(directory_output_general)


#regularization parameter space
alpha_reg_vect = np.logspace(0,-6,100)

#base 1: cos(u), cos(2u),...,cos(6u)
m_gt_1 = np.array([1., -1., 0., 0., 0., 0.]).T #SPARSO
m_gt_2 = np.array([-1.5, 1.5, -1.5, 1., -1., 0.]).T #gt_02 #NON SPARSO

m_gt_matrix = np.vstack([m_gt_1, m_gt_2])

table_diff_m_mean = np.zeros([m_gt_matrix.shape[0],len(args.perc_noise)])
table_diff_m_std = np.zeros([m_gt_matrix.shape[0],len(args.perc_noise)])
table_diff_u_mean = np.zeros([m_gt_matrix.shape[0],len(args.perc_noise)])
table_diff_u_std = np.zeros([m_gt_matrix.shape[0],len(args.perc_noise)])


#ciclo sui ground truth
for gt in np.arange(m_gt_matrix.shape[0])+1:

    m_gt = m_gt_matrix[gt-1]
    if gt<10:
        gt_str = 'gt_0' + str(gt)
    else:
        gt_str = 'gt_' + str(gt)

    directory_input_gt = directory_input_general + gt_str + '/'
    directory_output_gt = directory_output_general + gt_str + '/'
    if not os.path.exists(directory_output_gt):
        os.makedirs(directory_output_gt)
    clean_data = np.load(os.path.join(directory_input_gt+'clean_data.npy'))

    diff_m = np.zeros([args.perc_noise.size,n_prove])
    diff_u = np.zeros([args.perc_noise.size,n_prove])

    #ciclo sul noise
    for n in np.arange(args.perc_noise.size):
        str_noise = 'noise=' + str(args.perc_noise[n]) + '/'
        directory_output_prove_indip = directory_output_general+gt_str+ '/'+str_noise
        if not os.path.exists(directory_output_prove_indip):
            os.makedirs(directory_output_prove_indip)

        for prova in np.arange(n_prove)+1:
            if prova<10:
                str_prova = 'prova_0' + str(prova)
            else:
                str_prova = 'prova_' + str(prova)
            
            directory_input_prova = os.path.join(directory_input_gt+str_noise+str_prova+'/')

            directory_output = directory_output_prove_indip+str_prova+'/'
            if not os.path.exists(directory_output):
                os.makedirs(directory_output)
            
            #Load delle variabili
            data = np.load(os.path.join(directory_input_prova, 'noisy_data.npy'))
            u_sol_regularization_path = np.load(os.path.join(directory_input_prova, 'u_sol_regularization_path.npy'))
            m_regularization_path = np.load(os.path.join(directory_input_prova, 'm_regularization_path.npy'))
            main_loss_regularization_path = np.load(os.path.join(directory_input_prova, 'main_loss_regularization_path.npy'))
            data_loss_regularization_path = np.load(os.path.join(directory_input_prova, 'data_loss_regularization_path.npy'))

            #GRAFICI
            #1) Synthetic Data VS Noisy Data
            plt.figure()
            string = "synthetic data"
            plt.plot(t, clean_data, 'c', linewidth=3, alpha=.9, label=string)
            string = "data with noise = " + str(args.perc_noise[n]/100)
            plt.plot(t, data, '.', markersize=8, alpha=.9, label=string)

            title = "Synthetic Data vs Noisy Data no title x label y label"
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xticks(fontsize=18) 
            plt.yticks(fontsize=18)
            plt.xlabel('time', fontsize=20, labelpad=3)
            plt.ylabel('data', fontsize=20, labelpad=5)
            plt.legend(fontsize=14)
            plt.tight_layout()
            y_min_data, y_max_data = plt.gca().get_ylim()
            plt.show()

            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600)
            plt.close('all') 


            #2) SOLUTION REGULARIZATION PATH PLOT
            plt.figure()
            A = range(100)
            norm = Normalize(vmin=0, vmax=len(A) - 1)
            colors = cm.jet(norm(range(len(A))))

            for i in A:
                plt.plot(t, u_sol_regularization_path[i, :, 0], color=colors[i])

            plt.ylim(y_min_data, y_max_data)

            title = "Solution reg. path - Noise = " + str(args.perc_noise[n]/100)
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('time',fontsize=20, labelpad=3)
            plt.ylabel('values', fontsize=20, labelpad=5)
            plt.tight_layout()
            plt.show()
 
            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600)
            plt.close('all')           

 
            #3) PARAMETERS REGULARIZATION PATH PLOT
            colors = []
            plt.figure()
            dim_basis = m_gt.size
            for i in range(0, dim_basis):
                string = "m[" + str(i) + "]"
                line, = plt.plot(np.arange(alpha_reg_vect.size), m_regularization_path[:, i,0], alpha=.5, label=string)
                colors.append(line.get_color())

            x_position = len(alpha_reg_vect)  # Ultima posizione sull'asse delle x (100)
            for i, value in enumerate(m_gt):
                plt.scatter(x_position - 1, value, color=colors[i], marker='x')

            #alpha_reg_vect_str = alpha_reg_vect.astype(str)
            #plt.xticks(ticks=x, labels=alpha_reg_vect_str)
            title = "Parameters reg. path - Noise = " + str(args.perc_noise[n]/100)
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xticks(fontsize=18) 
            plt.yticks(fontsize=18)
            plt.xlabel('regularization parameter', fontsize=20, labelpad=3)
            plt.ylabel('parameters', fontsize=20, labelpad=5)
            plt.legend(fontsize=18, bbox_to_anchor=(1, 1))
            plt.legend(loc="upper left")
            plt.legend(fontsize=11)   #plt.legend(fontsize=12) per base 4, plt.legend(fontsize=14) negli altri casi
            # Visualizza l'immagine finale
            plt.tight_layout()
            plt.show()

            title = "Parameters reg. path with legend - Noise = " + str(args.perc_noise[n]/100) + " with legend"
            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600, bbox_inches='tight')
            plt.close('all') 


            #4) RELATIVE ERROR REGULARIZATION PATH PARAMETERS || m_alpha - m_gt ||2^2/||m_gt||2^2
            rel_error_param = np.zeros([m_regularization_path.shape[0], 1])
            for i in A:
                rel_error_param[i] = np.linalg.norm(m_regularization_path[i, :, 0] - m_gt) / np.linalg.norm(m_gt)

            plt.figure()
            plt.plot(alpha_reg_vect, rel_error_param, linewidth=3, alpha=.7, color='b')
            plt.xscale('log')
            plt.gca().invert_xaxis()
            title = "Relative Error reg. path - Noise = " + str(args.perc_noise[n] / 100)
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xlabel('regularization parameter', fontsize=20, labelpad=3)
            plt.ylabel('relative error', fontsize=20, labelpad=5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # Salvataggio
            title = "Relative Error reg. path - Noise = " + str(args.perc_noise[n] / 100) + " alpha x label"        
            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600, bbox_inches='tight')
            plt.close('all')


            #5) MAIN LOSS REGULARIZATION PATH
            plt.figure()
            plt.plot(alpha_reg_vect, main_loss_regularization_path, linewidth=3, alpha=.7, color='b')
            plt.xscale('log')
            plt.gca().invert_xaxis()
            title = "Loss reg. path - Noise = " + str(args.perc_noise[n] / 100)
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xlabel('regularization parameter', fontsize=20, labelpad=3)
            plt.ylabel('loss function', fontsize=20, labelpad=5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
  
            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600, bbox_inches='tight')
            plt.close('all')


            #6) DATA LOSS REGULARIZATION PATH
            #a) con titolo ed assi
            plt.figure()
            plt.plot(alpha_reg_vect, data_loss_regularization_path, linewidth=3, alpha=.7, color='b')
            plt.xscale('log')
            plt.gca().invert_xaxis()
            title = "Data loss reg. path - Noise = " + str(args.perc_noise[n] / 100)
            plt.suptitle(title, fontsize=20, y=0.95)
            plt.xlabel('regularization parameter', fontsize=20, labelpad=3)
            plt.ylabel('data loss function', fontsize=20, labelpad=5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
  
            filename = f"{title}.png"
            filepath = os.path.join(directory_output, filename)
            plt.savefig(filepath, dpi=600, bbox_inches='tight')
            plt.close('all')


            ##################################################### fine grafici

            #seleziono il valore di errore relativo minimo alpha* in questo cammino di regolarizzazione
            min_norm_reg_param = 1000000000000 #norma minima: inizializzo ad un numero molto grande di modo che inizi l'algoritmo e prenda il primo come minimo
            min_norm_reg_param_position = 0 #posizione della norma minima
            for pos in range(m_regularization_path.shape[0]):
                if np.linalg.norm(m_regularization_path[pos,:].squeeze() - m_gt)**2<min_norm_reg_param:
                    min_norm_reg_param = np.linalg.norm(m_regularization_path[pos,:].squeeze() - m_gt)**2
                    min_norm_reg_param_position = pos
            alpha_best = alpha_reg_vect[min_norm_reg_param_position] 
            RE_parameters_best = min_norm_reg_param/(np.linalg.norm(m_gt)**2)
            RE_solution_best = np.sum((u_sol_regularization_path[min_norm_reg_param_position,:] - clean_data)**2)/(np.linalg.norm(m_gt)**2) #5

            diff_m[n, prova-1] = RE_parameters_best
            diff_u[n, prova-1] = RE_solution_best

            print(prova)


        #Calcolo media e deviazione standard
        table_diff_m_mean[gt-1, n] = np.mean(diff_m[n,:])
        table_diff_m_std[gt-1, n] = np.std(diff_m[n,:])
        table_diff_u_mean[gt-1, n] = np.mean(diff_u[n,:])
        table_diff_u_std[gt-1, n] = np.std(diff_u[n,:])

    
    #QUI GT FISSATO e HA GIRATO SU TUTTI I NOISE
    if gt == 1:
        palette = sns.color_palette("Blues", n_colors=3)
    elif gt == 2:
        palette = sns.color_palette("Reds", n_colors=3)
    else:
        raise ValueError("gt!")

    #VIOLIN PLOT PARAMETERS
    df = pd.DataFrame({
        'Noise Level': np.repeat(['0.01', '0.1', '0.2'], n_prove),
        'Relative Error': np.concatenate([
            diff_m[0, :].squeeze(),
            diff_m[1, :].squeeze(),
            diff_m[2, :].squeeze()
        ])
    })
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Noise Level', y='Relative Error', data=df, hue='Noise Level', palette=palette, legend=False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title(f'Parameters Relative Error - Ground Truth {gt}', fontsize=20)
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15)

    # Visualizza l'immagine finale
    plt.tight_layout()
    plt.show()
    title = "Violin plot Parameters Relative Error - gt = " + str(gt)
    filename = f"{title}.png"
    filepath = os.path.join(directory_output_general, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close('all') 


    #VIOLIN PLOT SOLUTIONS
    df = pd.DataFrame({
        'Noise Level': np.repeat(['0.01', '0.1', '0.2'], n_prove),
        'Relative Error': np.concatenate([
            diff_u[0, :].squeeze(),
            diff_u[1, :].squeeze(),
            diff_u[2, :].squeeze()
        ])
    })
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Noise Level', y='Relative Error', data=df, hue='Noise Level', palette=palette, legend=False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title(f'Solutions Relative Error - Ground Truth {gt}', fontsize=20)
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15)

    # Visualizza l'immagine finale
    plt.tight_layout()
    plt.show()
    title = "Violin plot Solutions Relative Error - gt = " + str(gt)
    filename = f"{title}.png"
    filepath = os.path.join(directory_output_general, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close('all') 





#Saving tables
filename = 'table_error_m_mean.npy'
filepath = os.path.join(directory_output_general, filename)
np.save(filepath, table_diff_m_mean)
print('Table error on parameters')


filename = 'table_error_m_std.npy'
filepath = os.path.join(directory_output_general, filename)
np.save(filepath, table_diff_m_std)
print('Table std on parameters')


filename = 'table_error_u_mean.npy'
filepath = os.path.join(directory_output_general, filename)
np.save(filepath, table_diff_u_mean)

filename = 'table_error_u_std.npy'
filepath = os.path.join(directory_output_general, filename)
np.save(filepath, table_diff_u_std)

             







