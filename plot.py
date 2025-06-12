import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
import scipy
import numpy as np
import math



def plot_data(t, d, perc_noise):
    dim_state_variable = d.shape[1]

    for i in range(0, dim_state_variable):
        string = "s" + str(i + 1)
        plt.plot(t, d[:, i], alpha=.5, label=string)

    title = "Noisy Data, %Noise = " + str(float(perc_noise)) + "%"
    plt.suptitle(title, fontsize=20)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('data', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)

def plot_scatter(t, d, perc_noise):
        dim_state_variable = d.shape[1]
        plt.figure()
        for i in range(0, dim_state_variable):
            string = "data_" + str(i + 1)
            plt.scatter(t, d[:, i], alpha=.5, label=string)

        title = "Noisy Sampled Data, %Noise = " + str(float(perc_noise)) + "%"
        plt.suptitle(title, fontsize=20)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('data', fontsize=20)
        plt.legend(loc="upper right", fontsize=20)
        plt.show()

def plot_function_iter(t, u_sol, i):
    dim_state_variable = u_sol.shape[1]
    plt.figure()
    for h in range(0, dim_state_variable):
        string = "s" + str(h + 1)
        plt.plot(t, u_sol[:, h], alpha=.5, label=string)

    title = "Soluzione iterazione" + str(i)
    plt.suptitle(title, fontsize=20)
    plt.xlabel('time', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)

def plot_sol_vs_dati(t, u_sol, d):
    dim_state_variable = d.shape[1]
    plt.figure()
    for h in range(0, dim_state_variable):
        string = "soluzione-s" + str(h + 1)
        plt.plot(t, u_sol[:, h], alpha=.5, label=string)
        string = "dati-s" + str(h + 1)
        plt.plot(t, d[:, h], alpha=.5, label=string)

    title = "Soluzione ottenuta vs Dati"
    plt.suptitle(title, fontsize=20)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('data', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)


def plot_diff_iterations(diff_u_sol_iter, diff_u_sol_iter_NR):
    n_iter_Adj = diff_u_sol_iter.shape[0]
    dim_state_variable = diff_u_sol_iter.shape[1]
    plt.figure()
    for h in range(0, dim_state_variable):
        plt.subplot(1, dim_state_variable, h+1)
        string = "|| (u_ivp - u)(" + str(h) +") ||"
        plt.plot(diff_u_sol_iter[:,h], alpha=.5, label=string)
        string_1 = "|| (u_NR - u)(" + str(h) +") ||"
        plt.plot(diff_u_sol_iter_NR[:,h], alpha=.5, label=string_1)
        plt.legend(loc="upper right", fontsize=20)
    title = "Differenze soluzioni - dati. Numero iterazioni: "+str(n_iter_Adj)
    plt.suptitle(title, fontsize=20)
    plt.xlabel('iter', fontsize=20)
    plt.ylabel('|| u - d ||', fontsize=20)
    plt.show()



def histogram_parameters(m_0, m, m_gt):
    dim_state_variable = m_gt.shape[1]
    for h in range(0, dim_state_variable):
        plt.subplot(dim_state_variable, 1, h+1)
        categorie = []
        # Dati per l'istogramma
        for i in range(m_gt.shape[0]):
            parameter_name = 'm['+str(i)+','+str(h)+']'
            categorie.append(parameter_name)

        valori1 = m_0[:,h]
        valori2 = m[:,h]
        valori3 = m_gt[:,h]
        larghezza_colonne = 0.2     # larghezza delle colonne
        x = range(len(categorie))     # Posizione delle barre sull'asse x
        #plot
        plt.bar(x, valori1.squeeze(), width=larghezza_colonne, label='m0')
        plt.bar([i + larghezza_colonne for i in x], valori2.squeeze(), width=larghezza_colonne, label='m')
        plt.bar([i + 2*larghezza_colonne for i in x], valori3.squeeze(), width=larghezza_colonne, label='mgt')
        plt.axhline(0., color='k', linestyle='-', linewidth=0.5)
        plt.xlabel('Parameters labels')
        plt.ylabel('Parameters values')
        plt.title('m0 vs m vs m_gt, variable'+str(h))
        plt.xticks([i + larghezza_colonne for i in x], categorie)
        plt.legend()
