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
import sys
import math
import plot
import forward



class Phi_basis_function:
    #classe delle funzioni di base di dimensione T-1 x D.
    def __init__(self, dimension):

        if dimension==1:
            self.basis_functions = [
                #lambda u: 1.,  # cost
                #lambda u: u,  # u
                #lambda u: u ** 2,  # u^2
                lambda u: np.cos(u),  # cos(u)
                #lambda u: np.sin(u),  # sin(u)
                lambda u: np.cos(2 * u),  # cos(2*u)
                #lambda u: np.sin(2 * u),  # sin(2*u)
                lambda u: np.cos(3 * u),  # cos(3*u)
                #lambda u: np.sin(3 * u),  # sin(3*u)
                lambda u: np.cos(4 * u),  # cos(4*u)
                #lambda u: np.sin(4 * u),  # sin(4*u)
                lambda u: np.cos(5 * u),  # cos(5*u)
                #lambda u: np.sin(5 * u),  # sin(6*u)
                lambda u: np.cos(6 * u),   # cos(6*u)
                #lambda u: np.sin(6 * u)  # sin(6*u)
            ]
            self.der_basis_functions = [
                #lambda u: 0.,  # d1/du
                #lambda u: 1.,  # du/du
                #lambda u: 2 * u,  # du^2/du
                lambda u: -np.sin(u),  # dcos(u)/du
                #lambda u: np.cos(u),  # dsin(u)/du
                lambda u: -2 * np.sin(2 * u),  # dcos(2*u)/du
                #lambda u: 2 * np.cos(2 * u),  # dsin(2u)/du
                lambda u: -3 * np.sin(3 * u),  # dcos(3*u)/du
                #lambda u: 3 * np.cos(3 * u),  # dsin(3u)/du
                lambda u: -4 * np.sin(4 * u),  # dcos(4*u)/du
                #lambda u: 4 * np.cos(4 * u),  # dsin(4u)/du
                lambda u: -5 * np.sin(5 * u),  # dcos(5*u)/du
                #lambda u: 5 * np.cos(5 * u),  # dsin(5u)/du
                lambda u: -6 * np.sin(6 * u),  # dcos(6*u)/du
                #lambda u: 6 * np.cos(6 * u),  # dsin(6u)/du
            ]
        else:
            self.basis_functions = [
                lambda u: 1., #cost
                #lambda u: u.flat[0], #u[0]
                #lambda u: u.flat[1], #u[1]
                #lambda u: u.flat[0]**2, #u[0]^2
                #lambda u: u.flat[0]*u.flat[1], #u[0]*u[1]
                #lambda u: u.flat[1]**2, #u[1]^2
                #lambda u: np.sin(u.flat[0]), #sin(u[0])
                #lambda u: np.sin(u.flat[1]), #sin(u[1])
                lambda u: np.cos(u.flat[0]), #cos(u[0])
                lambda u: np.cos(u.flat[1]), #cos(u[1])
                lambda u: np.cos(2 * u.flat[0]),  # cos(2u[0])
                lambda u: np.cos(2 * u.flat[1])  # cos(2u[1])
                ]
            self.der_basis_functions = [
                lambda u: 0., #d1/du_0
                lambda u: 0., #d1/du_1
                #lambda u: 1., #du_0/du_0
                #lambda u: 0., #du_0/du_1
                #lambda u: 0.,  # du_1/du_0
                #lambda u: 1.,  # du_1/du_1
                #lambda u: 2*u.flat[0],  # du_0^2/du_0
                #lambda u: 0.,  # du_0^2/du_1
                #lambda u: u.flat[1],  # d(u_0*u_1)/du_0
                #lambda u: u.flat[0],  # d(u_0*u_1)/du_1
                #lambda u: 0.,  # du_1^2/du_0
                #lambda u: 2*u.flat[1],  # du_1^2/du_1
                #lambda u: np.cos(u.flat[0]),  # d(sen(u_0)/du_0
                #lambda u: 0.,  # d(sen(u_0)/du_1
                #lambda u: 0.,  # d(sen(u_1)/du_0
                #lambda u: np.cos(u.flat[1]),  # d(sen(u_1)/du_1
                lambda u: - np.sin(u.flat[0]),  # d(cos(u_0)/du_0
                lambda u: 0.,  # d(cos(u_0)/du_1
                lambda u: 0.,  # d(cos(u_1)/du_0
                lambda u: -np.sin(u.flat[1]),  # d(cos(u_1)/du_1
                lambda u: - 2*np.sin(2*u.flat[0]),  # d(cos(2*u_0)/du_0)
                lambda u: 0.,  # d(cos(u_0)/du_1
                lambda u: 0.,  # d(cos(u_1)/du_0
                lambda u: - 2* np.sin(2*u.flat[1]),  # d(cos(2*u_1)/du_1
                ]
        #Number of used basis function
        self.dim_basis = len(self.basis_functions)
    def Phi_basis_matrix(self, u):
        # Valutazione della u nelle funzioni di base (u0,...,u_T-2)
        n_sample = u.shape[0]
        values = np.zeros((n_sample, self.dim_basis))
        for j in range(0, self.dim_basis):
            for t in range(0, n_sample):
                values[t,j] = self.basis_functions[j](u[t,:])
        return values

    def dPhi_basis_matrix(self, u):
        # Valutazione della u nelle derivate delle funzioni di base (u0,...,u_T-2)
        n_sample = u.shape[0]
        dim_state_variable = u.shape[1]
        values = np.zeros((n_sample, dim_state_variable*self.dim_basis))
        for j in range(0, self.dim_basis):
            for h in range(0, dim_state_variable):
                for i in range(0, n_sample):
                    values[i,j*dim_state_variable + h] = self.der_basis_functions[j*dim_state_variable + h](u[i,:])
        return values

class regularization:
    def __init__(self, alpha, regularizer):

        self.alpha = alpha
        self.regularizer = regularizer

        if regularizer == "norm1":
            # Norm1 regularized
            epsilon_reg = 0.005  # Regularizer of norm1
            self.regularizer_function = lambda m: np.sum(np.sqrt(m**2 + epsilon_reg**2))
            self.der_regularizer_function = lambda m: m/np.sqrt(m**2 + epsilon_reg**2)
        else:
            # Norm2
            self.regularizer_function = lambda m: 1 / 2 * np.linalg.norm(m.flatten(), ord=2) ** 2
            self.der_regularizer_function = lambda m: m

class data_loss:
    def __init__(self, d):

        self.data_loss_function = lambda u: 1/2 * np.linalg.norm(u - d)**2
        self.der_data_loss_function = lambda u: u - d

def init_parameters_simulation(m, percentage_from_m):
    #INPUT:
    #m: gt parameter since it is known in simulation
    #percentage_from_m: represents the percentage from which initialize m parameters
    #OUTPUT:
    #m0: initialized parameters
    if len(m.shape)==1:
        m = np.expand_dims(m, axis=1)
    m_sizes = m.shape
    m_inf_norm = np.max(np.abs(m))
    m0 = m + m_inf_norm * percentage_from_m/100 * (2 * np.random.rand(*m_sizes) -1)
    return m0

def init_parameters_random(interval_init, dim_basis, dim_state_variable):
    # INPUT:
    # interval_init: estremo dell'intervallo in cui inizializzare i parametri
    # dim_state_variable: dimensione della variabile di stato
    # dim_basis: dimensione delle variabili di stato
    # OUTPUT:
    # m0: parametri inizializzati
    m0 = np.zeros([dim_basis, dim_state_variable])
    m0 = np.random.rand(*m0.shape) * 2 * interval_init - np.ones(m0.shape) * interval_init
    return m0

def compute_dF_du(u_sol, time, Phi_basis_function, m, solve_Adj):
    # a) Calcolare dF/du matrice a 4 indici TxUxTxU
    # b) Trasformare (dF/du) in una matrice a 2 indici (UxT)x(TxU)

    n_sample = len(time)
    dim_state_variable = u_sol.shape[1]
    dim_basis = Phi_basis_function.dim_basis
    dF_du = np.zeros([n_sample, dim_state_variable, n_sample, dim_state_variable])  # TxUxTxU
    dF_du_sol_matrix = np.zeros([dim_state_variable * n_sample, n_sample * dim_state_variable])  # (UxT)x(TxU)

    #  a) Calcolare dF/du matrice a 4 indici TxUxTxU
    dPhi_matrix = Phi_basis_function.dPhi_basis_matrix(u_sol[:-1, :])  # Matrice dPhi_du delle funzioni di base calcolate nella soluzione
    for t in range(0, n_sample):
        if t == 0:
            for h in range(0, dim_state_variable):
                dF_du[t, h, t, h] = 1.
        else:
            for h in range(0, dim_state_variable):
                # Caso a=t
                dF_du[t, h, t, h] = 1.
                # Caso a=t-1
                for b in range(0, dim_state_variable):
                    for j in range(0, dim_basis):
                        dF_du[t, h, t - 1, b] = dF_du[t, h, t - 1, b] - dPhi_matrix[t - 1, j * dim_state_variable + b] * m[j, h] * (time[t].item() - time[t - 1].item())
                    if b == h:
                        dF_du[t, h, t - 1, b] = dF_du[t, h, t - 1, b] - 1.

    # b) Trasformare (dF/du) in una matrice a 2 indici (TxU)x(TxU)
    for h in range(0, dim_state_variable):
        for t in range(0, n_sample):
            for b in range(0, dim_state_variable):
                for a in range(0, n_sample):
                    dF_du_sol_matrix[t + h * n_sample, a + b * n_sample] = dF_du[t, h, a, b]
    #Si traspone diversamente a seconda che si sta risolvendo NR o Adj
    if solve_Adj:
        #Se risolvo l'aggiunto traspongo
        dF_du_sol_matrix = dF_du_sol_matrix.T

    return dF_du_sol_matrix

def compute_F(c_i, u_sol, time, Phi_basis_function, m):
    # a) Calcolare F nell'attuale soluzione u (TxU)
    # b) Trasformare F in un vettore (UxT)x1
    n_sample = len(time)
    dim_state_variable = u_sol.shape[1]
    dim_basis = Phi_basis_function.dim_basis
    F_sol = np.zeros([n_sample, dim_state_variable])  # TxU

    # a) Calcolare F nell'attuale soluzione u
    Phi_matrix = Phi_basis_function.Phi_basis_matrix(
        u_sol[:-1, :])  # Matrice delle funzioni di base calcolate nell'attuale soluzione
    for t in range(0, n_sample):
        for h in range(0, dim_state_variable):
            if t == 0:
                F_sol[t, h] = u_sol[t, h] - c_i[h]
            else:
                F_sol[t, h] = (u_sol[t, h] - u_sol[t - 1, h])
                for j in range(0, dim_basis):
                    F_sol[t, h] = F_sol[t, h] - Phi_matrix[t - 1, j] * m[j, h] * (time[t].item() - time[t - 1].item())

    # b) Trasformare F in un vettore [TxU]x1
    F_sol_vec = F_sol.T.reshape(dim_state_variable * n_sample, 1)  # (TxU)x1


    return F_sol_vec

def Landweber_iteration(A, y, v_init, n_iter):
    # INPUT:
    # A: matrice sistema lineare
    # y: termine noto (dimensione corretta)
    # v_init: vettore v della soluzione precedente
    # n_iter = numero iterazioni Landweber
    #OUTPUT
    # v: soluzione con tecnica Landweber

    w = 1. / np.linalg.norm(A, 2) ** 2
    v_old = v_init
    for i in range(0, n_iter):
        v = v_old - w * np.dot(A.T, np.dot(A, v_old) - y)
        #print(i)
        # if (np.linalg.norm(v - v_old) <= 10 ** (-3)):  #if (np.linalg.norm(v - v_old) <= 10 ** (-6) * np.linalg.norm(v_old)):
        #     print('break landweber: '+str(i))
        #     break
        
        v_old = v

    return v

def Newton_Raphson(u_sol_init, time, Phi_basis_function, m, max_iter_NR, data, v_prec, max_iter_LW, conv=False):

    #INPUT
    # u_sol_init: inizializzazione della soluzione. Alle iterazioni successive inizializza con la soluzione precedente
    # time: tempi in cui calcolare la soluzione
    # Phi_basis_function: istanza della classe delle funzioni di base
    # m: parametri attuali
    # max_iter_NR: numero di iterazioni di Newton Raphson
    # data: dati reali (utili solo se voglio stoppare l'algoritmo non a convergenza)
    # conv: se True richiedo che NR termina quando arriva a convergenza
    #OUTPUT
    # u_sol: soluzione calcolata nei tempi t con condizione iniziale c_i

    # Algoritmo:
    # 1) Calcolare dF/du nella soluzione
    # 2) Calcolare F nella soluzione
    # 3) Risolvere: (dF/du) * v = F ->  v = (dF/du)^-1 * F.
    # 4) Aggiorno la corrente soluzione u come u - v

    #Inizializzazione variabili soluzione
    n_sample = len(time)
    dim_state_variable = u_sol_init.shape[1]
    u_sol = u_sol_init
    u_sol_vec = u_sol.T.reshape([n_sample*dim_state_variable, 1]) #TxU
    norm_F_iter = np.zeros([max_iter_NR, 1])  #norma delle F

    for i in range(0, max_iter_NR):
        # 1) Calcolare dF/du nella soluzione
        dF_du = compute_dF_du(u_sol, time, Phi_basis_function, m, False)  # (TxU)x(TxU)
        # 2) Calcolare F nella soluzione
        F_sol = compute_F(data[0,:], u_sol, time, Phi_basis_function, m) # (TxU)x1 questa ok
        # 3) Risolvere: (dF/du) * v = F ->  v = (dF/du)^-1 * F. #Lo si può fare con LW o con funzione solve
        #v_vec = Landweber_iteration(dF_du, F_sol, v_prec.T.reshape([v_prec.shape[0] * v_prec.shape[1], 1]), max_iter_LW) ATTENZIONE: DECOMMENTARE v = v_prec
        v_vec = np.linalg.solve(dF_du, F_sol)
        # 4) Aggiorno la corrente soluzione u come u - v
        u_sol_vec = u_sol_vec - v_vec # (TxU)x1
        u_sol = u_sol_vec.reshape([dim_state_variable, n_sample]).T #matrice TxU
        v = v_vec.reshape([dim_state_variable, n_sample]).T #TxU
        
        #v_prec = v solo se uso LW

        norm_F_iter[i] = np.linalg.norm(F_sol)

        if norm_F_iter[i] < 10**(-6):
            #print('NR converged at iter' + str(i))
            break
        if i==max_iter_NR-1:
            print('NR ha svolto tutte le iterazioni: ' + str(max_iter_NR))

        #u_sol_prec = u_sol
        #Se ho un vettore che deve essere reshaped in matrice il reshape prende il vettore e inizia a riempire la prima riga della matrice e così via.
        # Quando voglio scrivere a * dim_state_variable + b così significa che "b" è la
        # variabile che cicla internamente mentre a è quella che cicla esternamente. Quindi dopo che b ha ciclato
        # a si dovrà spostare della dimensione di b e quindi a deve essere moltiplicato per dim_state_variable. In generale se avessimo
        #i_1 -> dim_1, i_2 -> dim_2, i_n -> dim_n
        #for i_1, for i_2,...., for i_n (leggere dal fondo l'espressione riga sotto)
        # A[i_1*dim_2*..*dim_n +...+ i_(n-2)*dim_(n-1)*dim_n + i_(n-1)*dim_n + i_n] = A[i_1, i_2,...,i_n]

    #plt.plot(range(0,i), norm_F_iter[0:i, 0])

    return u_sol, v

def adjoint_state_solver(u_sol, m, d, time, Phi_basis_function, lambda_mul_prec, max_iter_LW):
    #Algoritmo
    #1) Calcolare dF_du nella soluzione
    #2) Calcolare dh_du nella soluzione
    #3) Risolvere sistema dF_du_sol_matrix * lambda = dh_du_sol
    #4) Trasformare lambda in matrice TxU

    n_sample = d.shape[0]
    dim_state_variable = d.shape[1]

    #1) Calcolare dF_du nella soluzione
    dF_du = compute_dF_du(u_sol, time, Phi_basis_function, m, True) #2 indici: dimensioni (UxT)x(TxU)

    #2) Calcolare dh_du nella soluzione
    dh_du = data_loss(d).der_data_loss_function
    dh_du_u_sol = dh_du(u_sol) #TxU (matrice)
    dh_du_u_sol_vec = dh_du_u_sol.T.reshape(dim_state_variable * n_sample, 1)  # (TxU)x1

    #3) Risolvere sistema dF_du_sol_matrix * lambda = dh_du_sol
    #Con iterazione di Landweber
    lambda_mul_vec = Landweber_iteration(dF_du, dh_du_u_sol_vec, lambda_mul_prec.T.reshape([lambda_mul_prec.shape[0] * lambda_mul_prec.shape[1], 1]), max_iter_LW)
    #Con linalg.solve
    #lambda_mul_vec = np.linalg.solve(dF_du, dh_du_u_sol_vec)
    #4) Trasformare lambda_mul_vec vettore [TxU]x1
    lambda_mul = lambda_mul_vec.reshape([dim_state_variable, n_sample]).T #matrice TxU

    return lambda_mul

def compute_Obj(Phi_basis_function, d, u_sol, m, time, lambda_mul, regularization):

    #1) Calcolare dh_du nella soluzione
    h_d = data_loss(d).data_loss_function
    h_d_u_sol = h_d(u_sol) #TxU (matrice)

    #2) Calcolo alpha * dg_dm(m) dimensione D x U
    alpha = regularization.alpha
    g_m = regularization.regularizer_function(m)

    #3) Calcolo F
    F_sol = compute_F(d[0,:], u_sol, time, Phi_basis_function, m)

    #4) Calcolo <lambda_mul, F>
    lambda_mul_F = np.sum(np.multiply(lambda_mul, F_sol))

    #4) Calcolo Loss
    Main_Loss_Obj = h_d_u_sol + alpha * g_m - lambda_mul_F

    return Main_Loss_Obj
    
def g_step_BB(m_iter_i, m_iter_i_1, Grad_Obj_iter_i, Grad_Obj_iter_i_1):
    #DA CONTROLLARE
    diff1 = m_iter_i - m_iter_i_1
    diff2 = Grad_Obj_iter_i - Grad_Obj_iter_i_1
    tau = np.sum(np.dot(diff1,diff2), 'all')/np.norm(diff2)
    return tau

def compute_gradient(Phi_basis_function, u_sol, m, time, lambda_mul, regularization):

    #1) Calcolo alpha * dg_dm(m) dimensione D x U
    alpha = regularization.alpha
    dg_dm = regularization.der_regularizer_function

    #2) Calcolo matrice prodotto scalare lambda_dF_dm dimensioni D x U
    dim_state_variable = u_sol.shape[1]
    dim_basis = Phi_basis_function.dim_basis
    n_sample = u_sol.shape[0]
    lambda_dFdm = np.zeros([dim_basis, dim_state_variable])
    Phi_matrix = Phi_basis_function.Phi_basis_matrix(u_sol[:-1, :]) #valutazione delle prime T-1 componenti di u nelle funzioni di base

    for j in range(0, dim_basis):
        for b in range(0, dim_state_variable):
            for t in range(1, n_sample):
                lambda_dFdm[j, b] = lambda_dFdm[j, b] - lambda_mul[t,b]*Phi_matrix[t-1, j]*(time[t, 0] - time[t - 1, 0])

    Grad_Obj = alpha * dg_dm(m) - lambda_dFdm

    return Grad_Obj

def update_parameters(m, tau, Grad_Obj, regularization):
    # INPUT
    # m: parametri attuali da aggiornare dimensione DxU
    # tau: passo del gradiente iniziale
    # Grad_Obj: Gradiente del funzionale obiettivo, dimensione DxU
    #OUTPUT
    # parametri aggiornati

    #Gradient Step
    m_new = m - tau * Grad_Obj #Aggiorno m

    if regularization.alpha != 0: #se sto regolarizzando
        m_new = hard_thresholding(m_new, regularization.alpha/2) #applico regola di hard thresholding
        #print('Hard thresholded')

    return m_new

def hard_thresholding(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

def armijo_rule(u_sol, m, data, time, regularization, Phi_basis_function, Grad_Obj, max_iter_NR, v_prec, max_iter_LW, max_iter_BT):
    #Questa funzione serve per determinare lo step del gradiente ottimale con la regola di Armijo (classica regola Armijo)
    sigma = 0.1 #10**(-4) #contrast la grandezza di GradObj
    beta = 0.5 #ci dà la potenza che moltiplica il peso
    h_d = data_loss(data).data_loss_function
    h_d_u_sol = h_d(u_sol) #scalare, data loss nell'attuale soluzione
    alpha = regularization.alpha
    g = regularization.regularizer_function
    g_m = g(m) #scalare, regolarizzazione nell'attuale soluzione

    l = 0
    inequality_satisfied = True
    while inequality_satisfied and l<=max_iter_BT:
        m_new = update_parameters(m, np.power(beta, l), Grad_Obj)
        u_sol_new, v_new = Newton_Raphson(u_sol, time, Phi_basis_function, m_new, max_iter_NR, data, v_prec, max_iter_LW)
        h_d_u_sol_new = h_d(u_sol_new)
        g_m_new = g(m_new)
        if h_d_u_sol_new + alpha*g_m_new <= h_d_u_sol + alpha*g_m - np.power(beta, l) * sigma * np.linalg.norm(Grad_Obj):
            break
        l += 1

    return m_new, u_sol_new, v_new, np.power(beta, l)

def update_sol_and_param_NR(u_sol, v, m, tau, Grad_Obj, c_i, t, Phi_basis_function, max_iter_NR, data, max_iter_LW, BT, n_iter_BT, max_iter_BT, energy_factor, regularization):
    #1) aggiorno parametri m e soluzione corrispondente u
    #2) controllo sulla soluzione ottenuta
    #a) se la soluzione non esplode: return
    #b) se la soluzione esplode o non esiste: si richiama la funzione con step gradiente tau/2

    m_new = update_parameters(m, tau, Grad_Obj, regularization)
    u_sol_new, v_new = Newton_Raphson(u_sol, t, Phi_basis_function, m_new, max_iter_NR, data, v, max_iter_LW)
    return m_new, u_sol_new, v_new, tau

def update_sol_and_param_NR_BT(u_sol, v, m, tau, Grad_Obj, c_i, t, Phi_basis_function, max_iter_NR, data, max_iter_LW, BT, n_iter_BT, max_iter_BT, energy_factor, regularization):
    #1) aggiorno parametri m e soluzione corrispondente u
    #2) controllo sulla soluzione ottenuta
    #a) se la soluzione non esplode: return
    #b) se la soluzione esplode o non esiste: si richiama la funzione con step gradiente tau/2

    m_new = update_parameters(m, tau, Grad_Obj, regularization)
    u_sol_new, v_new = Newton_Raphson(u_sol, t, Phi_basis_function, m_new, max_iter_NR, data, v, max_iter_LW)

    if BT:
        #Entro qui solo se BT = variabile booleana che vale 1 se voglio fare BackTracking, 0 altrimenti
        if (exploded_sol(u_sol_new, data, energy_factor) and n_iter_BT<max_iter_BT):
            return update_sol_and_param_NR(u_sol, v, m, tau*0.9, Grad_Obj, c_i, t, Phi_basis_function, max_iter_NR, data, max_iter_LW, BT, n_iter_BT+1, max_iter_BT, energy_factor)
        elif (np.linalg.norm(u_sol-u_sol_new) < 10**(-4) * np.linalg.norm(u_sol) and n_iter_BT<max_iter_BT):
            return update_sol_and_param_NR(u_sol, v, m, tau*1.1, Grad_Obj, c_i, t, Phi_basis_function, max_iter_NR, data, max_iter_LW, BT, n_iter_BT+1, max_iter_BT, energy_factor)
        if n_iter_BT==max_iter_BT:
            print("Superato numero max di Backtracking.")
            sys.exit(1)

    return m_new, u_sol_new, v_new, tau

def update_sol_and_param_solve_ivp(u_sol, m, tau, Grad_Obj, c_i, t, Phi_basis_function, data, BT, n_iter_BT, max_iter_BT, energy_factor, regularization):
    #1) aggiorno parametri m e soluzione corrispondente u
    #2) controllo sulla soluzione ottenuta
    #a) se la soluzione non esplode: return
    #b) se la soluzione esplode o non esiste: si richiama la funzione con step gradiente tau/2

    m_new = update_parameters(m, tau, Grad_Obj, regularization)
    u_sol_new = solve_ivp(lambda t, u, Phi, m: forward.eqdiff(t, u, Phi, m), [t[0], t[-1]], c_i, t_eval=t.squeeze(), args=(Phi_basis_function, m),method='LSODA')
    u_sol_new = u_sol_new.y.T

    if BT:
        #Entro qui solo se BT = variabile booleana che vale 1 se voglio fare BackTracking, 0 altrimenti
        if (exploded_sol(u_sol_new, data, energy_factor) and n_iter_BT<max_iter_BT):
            return update_sol_and_param_solve_ivp(u_sol, m, tau*0.9, Grad_Obj, c_i, t, Phi_basis_function, data, BT, n_iter_BT+1, max_iter_BT, energy_factor)         
        elif (np.linalg.norm(u_sol-u_sol_new) < 10**(-4) * np.linalg.norm(u_sol) and n_iter_BT<max_iter_BT):
            return update_sol_and_param_solve_ivp(u_sol, m, tau*1.1, Grad_Obj, c_i, t, Phi_basis_function, data, BT, n_iter_BT+1, max_iter_BT, energy_factor)    
        if n_iter_BT==max_iter_BT:
            print("Superato numero max di Backtracking.")
            sys.exit(1)

    return m_new, u_sol_new, tau

def early_stopping(loss, i, n_check_es):
    if i>=n_check_es: #qui posso controllare almeno le prime n_check_es iterazioni
        check = loss[i-n_check_es:i]
        diff_matrix = np.abs(check[:, None] - check).squeeze() #matrice di tutte le possibili differenze del vettore
        if np.all(diff_matrix < 10**(-3)):
            return True
        else:
            return False
    return False

def exploded_sol(u_sol, data, energy_factor):
    #Questa funzione controlla che il vettore u_sol non contenga componenti Nan o inf e controlla che l'energia di u_sol sia vicina a quella dei dati data
    #INPUT:
    # u_sol: soluzione da testare
    # data: dati del problema
    #OUTPUT
    # True se la soluzione esplode, False altrimenti
    exploded_sol_vs_data = False
    dim_state_variable = u_sol.shape[1]
    for h in range(0, dim_state_variable):
        if (np.linalg.norm(u_sol[:,h]) > energy_factor * np.linalg.norm(data[:,h])): #controllo su ogni traiettoria
            exploded_sol_vs_data = True

    if ((np.isnan(u_sol).any()) or (np.isinf(u_sol).any()) or exploded_sol_vs_data):
        return True

    return False

def ordine_di_grandezza(numero):
    if numero == 0:
        return 0
    elif np.abs(numero)<1:
        ordine = 0
        while abs(numero) < 1:
            numero *= 10
            ordine -= 1
    else:
        log_10 = math.log10(abs(numero))
        ordine = int(log_10)

    return ordine



















# def Landweber_iteration(A, y, n_iter):
#     # INPUT:
#     # A: matrice sistema lineare
#     # y: termine noto
#     #OUTPUT
#     # x: soluzione con tecnica Landweber
#     # A.shape[1]
#
#     x = np.zeros((A.shape[1],1))
#     list_x = np.zeros([n_iter, x.shape[0], x.shape[1]])
#     U, S, V = np.linalg.svd(A)
#     w = 1/np.abs(np.max(S))**2 #è lo step di Landwber: per regola di convergenza deve essere < max_val_sing^2/2. Allora divido per 4
#     for i in range(n_iter):
#         list_x[i, :] = x
#         x_old = x
#         x = x - w * np.matmul(np.transpose(A), np.matmul(A, x) - y)
#         if (np.linalg.norm(x - x_old) <= 10**(-4) * np.linalg.norm(x)):
#             x = list_x[int(np.floor(3*i/4)), :]
#             print('break landweber: '+str(i)+ '. Solution taken position: '+str(int(np.floor(3*i/4))))
#             break
#     return x

# def backtracking(u_sol, m, n_iter_adj, max_iter_BT):

#             #Here if at least one iteration: Back-Tracking
#             print('\n Re-Compute the solution with Back-Tracking')
#             tau_adaptive = tau_adaptive/2
#             m = prev_m - tau_adaptive * Grad_Obj
#             # The first "column" contains Identity
#             for h_1 in range(0, args.dim_state_variable):
#                 for k_1 in range(0, args.dim_state_variable):
#                     if h_1 == k_1:
#                         m[h_1, 0, k_1] = 1
#                     else:
#                         m[h_1, 0, k_1] = 0
#             u_sol, dF_u_sol_matrix, sol_nan_inf = my_solver_Newton_Raphson(t, Phi, m, c_i)
#
#         n_iter_Back_Tracking += 1
    #
    # if n_iter_Back_Tracking==args.max_n_back_track:
    #     u_sol[:,:] = 0



#import ctypes
# from ctypes import CDLL, POINTER, c_double, c_int, c_void_p, c_size_t, byref
# cpp_library = CDLL("./utils.so")
#
# def solver_Prova_funzionante(c_i, t, Phi, m):
#     #INPUT
#     # c_i: condizione iniziale
#     # t: tempi in cui calcolare la soluzione
#     # Phi: istanza della classe delle funzioni di base
#     # m: parametri attuali
#     #OUTPUT
#     # u_sol: soluzione calcolata nei tempi t con condizione iniziale c_i
#
#     n_sample = len(t)
#     dim_state_variable = len(c_i)
#     u_sol_sizes = np.array([n_sample, dim_state_variable])
#
#     #Import function
#     generate_solution = cpp_library.generate_solution
#     generate_solution.argtypes = [POINTER(c_double), c_int, POINTER(c_double), c_int]
#     generate_solution.restype = POINTER(c_double * n_sample * dim_state_variable)
#     dtype = np.float64
#     c_i_ptr = c_i.ctypes.data_as(POINTER(c_double))
#     t_ptr = t.ctypes.data_as(POINTER(c_double))  # transform the np array into a pointer
#
#     u_sol_ptr = generate_solution(c_i_ptr, dim_state_variable, t_ptr, n_sample)
#     u_sol = np.frombuffer(u_sol_ptr.contents)
#
#     u_sol = u_sol.reshape(u_sol_sizes)
#     return u_sol











