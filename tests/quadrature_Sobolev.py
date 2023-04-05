################################################
################################################

## Script for Sobolev RKHS simulations
## Be aware that this simulation can take up to 2 hours

################################################
################################################


import sys
sys.path.insert(0, '..')
## Import useful classic functions
import matplotlib.pyplot as plt
import numpy as np
import cmath
import math
import scipy.integrate as integrate
from scipy.special import zeta, polygamma, factorial
## Import the Bernoulli kernel
from kernels.meta import *
from kernels.Bernoulli import *
## Import the samplers
from samplers.Herding import *
from samplers.BQ import *
from samplers.CUE_sampler import *
from samplers.Uniform_sampler import *
## Import the evaluations functions
from env.evaluations import *


### Activate a flag for saving the results files
SAVE_NPY_FILES_FLAG = 1 # 1 is yes 0 is no
SAVE_PDF_FILES_FLAG = 1 # 1 is yes 0 is no


### The mean function for the uniform distribution on [0,1]
def mu_function(x):
    return 1

### The number of samples for randomized quadratures
N_exp = 50


### The list of the number of nodes in each simulation
N_nodes_list = [5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50]
N_nodes_list_shifted = [i-1 for i in N_nodes_list]
N_nodes_max = max(N_nodes_list)
N_nodes_list_len = len(N_nodes_list)


### Define the error arrays
CUE_error_array = np.zeros((N_exp,N_nodes_list_len))
CUE_unbiased_error_array = np.zeros((N_exp,N_nodes_list_len))
iid_lvs_error_array = np.zeros((N_exp,N_nodes_list_len))
reg_iid_lvs_error_array_1 = np.zeros((N_exp,N_nodes_list_len))
reg_iid_lvs_error_array_2 = np.zeros((N_exp,N_nodes_list_len))
reg_iid_lvs_error_array_3 = np.zeros((N_exp,N_nodes_list_len))


### Define the kernel
kernel_order = 1
my_kernel_0 = Bernoulli_Kernel_Function(kernel_order,0)


### Define the regularization parameters for LVSQ
lambda_reg_list = [0.0001,0.1,0.2,0.3]


### Get DPP and LVSQ sequences
for N_ in list(range(N_nodes_list_len)):
    for _ in list(range(N_exp)):
        ### Generate the sequences
        print("DPP " + str(N_nodes_list[N_]) + " nodes, sample number " + str(_))
        CUE_seq = CUE_seq_generator(N_nodes_list[N_])
        print("LVSQ " + str(N_nodes_list[N_]) + " nodes, sample number " + str(_))
        Uniform_seq = Uniform_seq_generator(N_nodes_list[N_])
        ### Calculate the weights of the two quadratures
        CUE_kernel_matrix = my_kernel_0.get_kernel_matrix(CUE_seq)
        Uniform_kernel_matrix = my_kernel_0.get_kernel_matrix(Uniform_seq)
        f_values_0_CUE = np.ones(N_nodes_list[N_])
        f_values_0_Uniform = np.ones(N_nodes_list[N_])
        ###
        alpha_CUE = Calculate_alpha_subspace_kernel_regression(CUE_kernel_matrix,f_values_0_CUE)
        alpha_CUE_unbiased = 1/N_nodes_list[N_]*np.ones(N_nodes_list[N_])
        alpha_Uniform = Calculate_alpha_ridge_kernel_regression(Uniform_kernel_matrix,f_values_0_Uniform,lambda_reg_list[0])
        alpha_Uniform_reg_1 = Calculate_alpha_ridge_kernel_regression(Uniform_kernel_matrix,f_values_0_Uniform,lambda_reg_list[1]*N_nodes_list[N_])
        alpha_Uniform_reg_2 = Calculate_alpha_ridge_kernel_regression(Uniform_kernel_matrix,f_values_0_Uniform,lambda_reg_list[2]*N_nodes_list[N_])
        alpha_Uniform_reg_3 = Calculate_alpha_ridge_kernel_regression(Uniform_kernel_matrix,f_values_0_Uniform,lambda_reg_list[3]*N_nodes_list[N_])
        ###
        RKHS_error_CUE =  evaluate_quadrature_on_unit_norm_function(alpha_CUE,CUE_seq,f_values_0_CUE,my_kernel_0)
        RKHS_error_CUE_unbiased =  evaluate_quadrature_on_unit_norm_function(alpha_CUE_unbiased,CUE_seq,f_values_0_CUE,my_kernel_0)
        RKHS_error_iid_lvs =  evaluate_quadrature_on_unit_norm_function(alpha_Uniform,Uniform_seq,f_values_0_Uniform,my_kernel_0)
        RKHS_error_iid_lvs_reg_1 =  evaluate_quadrature_on_unit_norm_function(alpha_Uniform_reg_1,Uniform_seq,f_values_0_Uniform,my_kernel_0)
        RKHS_error_iid_lvs_reg_2 =  evaluate_quadrature_on_unit_norm_function(alpha_Uniform_reg_2,Uniform_seq,f_values_0_Uniform,my_kernel_0)
        RKHS_error_iid_lvs_reg_3 =  evaluate_quadrature_on_unit_norm_function(alpha_Uniform_reg_3,Uniform_seq,f_values_0_Uniform,my_kernel_0)
        ###
        CUE_error_array[_,N_] = RKHS_error_CUE
        CUE_unbiased_error_array[_,N_] = RKHS_error_CUE_unbiased
        iid_lvs_error_array[_,N_] = RKHS_error_iid_lvs
        reg_iid_lvs_error_array_1[_,N_] = RKHS_error_iid_lvs_reg_1
        reg_iid_lvs_error_array_2[_,N_] = RKHS_error_iid_lvs_reg_2
        reg_iid_lvs_error_array_3[_,N_] = RKHS_error_iid_lvs_reg_3


### Get the average error for DPPKQ and LVSQ
CUE_mean_error_array = np.mean(CUE_error_array, axis = 0)
CUE_unbiased_mean_error_array = np.mean(CUE_unbiased_error_array, axis = 0)
iid_lvs_mean_error_array = np.mean(iid_lvs_error_array, axis=0)
###
reg_iid_lvs_mean_error_array_1 = np.mean(reg_iid_lvs_error_array_1, axis=0)
reg_iid_lvs_mean_error_array_2 = np.mean(reg_iid_lvs_error_array_2, axis=0)
reg_iid_lvs_mean_error_array_3 = np.mean(reg_iid_lvs_error_array_3, axis=0)


### Get Herding sequence
Herding_seq = herding_get_sequence_Sobolev(N_nodes_max,my_kernel_0,mu_function,250)
Herding_N_nodes_list = list(range(N_nodes_max))
Herding_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    Herding_kernel_matrix = get_kernel_matrix_Bernoulli_kernel(kernel_order,Herding_seq[0:n+1])
    f_values_Herding =  np.ones(n+1)
    alpha_Herding =  1/(n+1)*np.ones(n+1)   
    RKHS_error_Herding = evaluate_quadrature_on_unit_norm_function(alpha_Herding,Herding_seq[0:n+1],f_values_Herding,my_kernel_0)
    Herding_error_array[n] = RKHS_error_Herding
###
del Herding_N_nodes_list[0]
Herding_N_nodes_list.insert(N_nodes_max-1,N_nodes_max)


### Get Bayesian Quadrature sequence
BQ_seq = BQ_get_sequence_Sobolev(N_nodes_max,my_kernel_0,mu_function,250)
BQ_N_nodes_list = list(range(N_nodes_max))
BQ_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    BQ_kernel_matrix = get_kernel_matrix_Bernoulli_kernel(kernel_order,BQ_seq[0:n+1])
    f_values_BQ =  np.ones(n+1)
    alpha_BQ =  Calculate_alpha_subspace_kernel_regression(BQ_kernel_matrix,f_values_BQ)   
    RKHS_error_BQ = evaluate_quadrature_on_unit_norm_function(alpha_BQ,BQ_seq[0:n+1],f_values_BQ,my_kernel_0)
    BQ_error_array[n] = RKHS_error_BQ
###
del BQ_N_nodes_list[0]
BQ_N_nodes_list.insert(N_nodes_max-1,N_nodes_max)


### Get the rates of the sequences
CUE_rate = round(-get_correlation(np.log(N_nodes_list)/np.log(10),np.log(CUE_mean_error_array)/np.log(10)),1)
iid_rate = round(-get_correlation(np.log(N_nodes_list)/np.log(10),np.log(iid_lvs_mean_error_array)/np.log(10)),1)
iid_rate_1 = round(-get_correlation(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_1)/np.log(10)),1)
iid_rate_2 = round(-get_correlation(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_2)/np.log(10)),1)
GQ_CUE_rate = round(-get_correlation(np.log(N_nodes_list)/np.log(10),np.log(CUE_unbiased_mean_error_array)/np.log(10)),1)
Herding_rate = round(-get_correlation(np.log(Herding_N_nodes_list)/np.log(10),np.log(Herding_error_array)/np.log(10)),1)
BQ_rate = round(-get_correlation(np.log(BQ_N_nodes_list)/np.log(10),np.log(BQ_error_array)/np.log(10)),1)


### Plot number 1
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(CUE_mean_error_array)/np.log(10), label= "DPPKQ: "+str(CUE_rate),linestyle='-', marker='.',color='#003399')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(CUE_unbiased_mean_error_array)/np.log(10), label= "DPPUQ: "+str(GQ_CUE_rate),linestyle='-', marker='.',color='#0066ff')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(Herding_error_array[N_nodes_list_shifted])/np.log(10), label= "Herding: "+str(Herding_rate),linestyle='-', marker='.',color = '#cc3300')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(BQ_error_array[N_nodes_list_shifted])/np.log(10), label= "BQ: "+str(BQ_rate),linestyle='-', marker='.', color = '#993333')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(iid_lvs_mean_error_array)/np.log(10) , label = "LVSQ "+r'($\lambda = 0$)'+": "+str(iid_rate),linestyle='-', marker='.',color = '#003300')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_1)/np.log(10) , label = "LVSQ "+r'($\lambda = 0.1$)',linestyle='-', marker='.',color = '#006600')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_2)/np.log(10) , label = "LVSQ "+r'($\lambda = 0.2$)',linestyle='-', marker='.', color='#339933')
###
plt.legend( loc='lower left', bbox_to_anchor=(0,0), fontsize=10)
plt.xlabel(r'$\log_{10} (\mathrm{N})$', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel(r'$\log_{10}$'+"(Squared error)", fontsize=18)
plt.yticks(fontsize=18)
if SAVE_PDF_FILES_FLAG ==1:
    plt.savefig('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_fig_1.pdf')
plt.show()


### Plot number 2
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(CUE_mean_error_array)/np.log(10), label= "DPPKQ: "+str(CUE_rate),linestyle='-', marker='.',color='#003399')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(CUE_unbiased_mean_error_array)/np.log(10), label= "DPPUQ: "+str(GQ_CUE_rate),linestyle='-', marker='.',color='#0066ff')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(Herding_error_array[N_nodes_list_shifted])/np.log(10), label= "Herding: "+str(Herding_rate),linestyle='-', marker='.',color = '#cc3300')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(BQ_error_array[N_nodes_list_shifted])/np.log(10), label= "BQ: "+str(BQ_rate),linestyle='-', marker='.', color = '#993333')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(iid_lvs_mean_error_array)/np.log(10) , label = "LVSQ "+r'($\lambda = 0$)'+": "+str(iid_rate),linestyle='-', marker='.',color = '#003300')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_1)/np.log(10) , label = "LVSQ "+r'($\lambda = 0.1$)',linestyle='-', marker='.',color = '#006600')
plt.plot(np.log(N_nodes_list)/np.log(10),np.log(reg_iid_lvs_mean_error_array_2)/np.log(10) , label = "LVSQ "+r'($\lambda = 0.2$)',linestyle='-', marker='.', color='#339933')
###
plt.legend( loc='lower left', bbox_to_anchor=(0,0), fontsize=12)
plt.xlabel(r'$\log_{10} (\mathrm{N})$', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel(r'$\log_{10}$'+"(Squared error)", fontsize=18)
plt.yticks(fontsize=18)
if SAVE_PDF_FILES_FLAG ==1:
    plt.savefig('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_fig_2.pdf')
plt.show()


### Save files
if SAVE_NPY_FILES_FLAG == 1:
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_DPPKQ.npy', CUE_mean_error_array)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_DPPUQ.npy', CUE_unbiased_mean_error_array)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ0.npy', iid_lvs_mean_error_array)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ1.npy', reg_iid_lvs_mean_error_array_1)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ2.npy', reg_iid_lvs_mean_error_array_2)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_N_List.npy', N_nodes_list)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_Herding.npy', Herding_error_array)
    np.save('Sobolev/Sobolev_kernel_order_'+str(kernel_order)+'_BQ.npy', BQ_error_array)

