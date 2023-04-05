################################################
################################################

## Script for Gaussian kernel simulations
## Be aware that this simulation can take up to 15 mins

################################################
################################################


import sys
sys.path.insert(0, '..')
## Import useful classic functions
import numpy as np
import cmath
import math
import scipy.integrate as integrate
from scipy.special import zeta, polygamma, factorial
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
## Import the Gaussian kernel
from kernels.meta import *
from kernels.Gaussian_kernel import *
## Import the samplers
from samplers.HE_sampler import *
from samplers.Herding import *
from samplers.BQ import *
## Import the evaluations functions
from env.evaluations import *


### Activate a flag for saving the results files
SAVE_TXT_FILES_FLAG = 1 # 1 is yes 0 is no
SAVE_PDF_FILES_FLAG = 1 # 1 is yes 0 is no


### The number of samples for randomized quadratures
N_exp = 50


### The list of the number of nodes in each simulation
N_nodes_max = 50
N_nodes_list = list(range(N_nodes_max))
del N_nodes_list[0]
N_nodes_list_len = len(N_nodes_list)


### Define the error arrays
HE_error_array = np.zeros((N_exp,N_nodes_list_len))
HE_unbiased_error_array = np.zeros((N_exp,N_nodes_list_len))


### Define the kernel
kernel_scale = 0.3
a = 0.25
b = 1/(2*np.power(kernel_scale,2))
c = np.sqrt(np.power(a,2)+2*a*b)
my_kernel_0 = Gaussian_Kernel_Function(kernel_scale,0)


### The mean element for the Gaussian measure 
my_mean_element = mean_element(a,kernel_scale)


### Get the DPP sequences
for N_ in list(range(N_nodes_list_len)):
    for _ in list(range(N_exp)):
        ## Generate the sequences
        print("DPP " + str(N_nodes_list[N_]) + " nodes, sample number " + str(_))
        HE_seq = HE_seq_generator(N_nodes_list[N_],4*c)
        ## Calculate the weights of the quadrature
        HE_kernel_matrix = get_kernel_matrix_Gaussian_kernel(kernel_scale,HE_seq)
        f_values_0_HE = my_mean_element(HE_seq)
        alpha_HE = Calculate_alpha_subspace_kernel_regression(HE_kernel_matrix,f_values_0_HE)
        RKHS_error_HE =  evaluate_quadrature_on_unit_norm_function(alpha_HE,HE_seq,f_values_0_HE,my_kernel_0)
        HE_error_array[_,N_] = RKHS_error_HE


### Get the average error for DPPKQ 
HE_mean_error_array = np.mean(HE_error_array, axis = 0)


### Get theoretical upper bound for DPPKQ
DPP_loose_th_upper_bound_list = DPP_loose_th_upper_bound(N_nodes_list,a,kernel_scale)


### Get Monte Carlo sequences
MC_seq = np.random.normal(0, 1, N_nodes_max)
MC_N_nodes_list = list(range(N_nodes_max))
MC_error_array = np.zeros((N_exp,N_nodes_max))
MC_mean_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    print("Monte Carlo " + str(N_nodes_list[N_]) + " nodes, sample number " + str(_))
    for _ in list(range(N_exp)):
        MC_seq = np.random.normal(0, 1, n+1)
        MC_kernel_matrix = my_kernel_0.get_kernel_matrix(MC_seq)
        f_values_MC =  my_mean_element(MC_seq)
        alpha_MC =  1/(n+1)*np.ones(n+1)   
        MC_error_array[_,n] = evaluate_quadrature_on_unit_norm_function(alpha_MC,MC_seq,f_values_MC,my_kernel_0)
    MC_mean_error_array[n] = np.mean(MC_error_array[:,n], axis = 0)


### Get Herding sequence
Herding_seq = herding_get_sequence(N_nodes_max,my_kernel_0,my_mean_element,500)
Herding_N_nodes_list = list(range(N_nodes_max))
Herding_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    Herding_kernel_matrix = my_kernel_0.get_kernel_matrix(Herding_seq[0:n+1])
    f_values_Herding =  my_mean_element(Herding_seq[0:n+1])
    alpha_Herding =  1/(n+1)*np.ones(n+1)   
    Herding_error_array[n] = evaluate_quadrature_on_unit_norm_function(alpha_Herding,Herding_seq[0:n+1],f_values_Herding,my_kernel_0)


### Get Bayesian Quadrature sequence
BQ_seq = BQ_get_sequence(N_nodes_max,my_kernel_0,my_mean_element,500)
BQ_N_nodes_list = list(range(N_nodes_max))
BQ_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    BQ_kernel_matrix = my_kernel_0.get_kernel_matrix(BQ_seq[0:n+1])
    f_values_BQ =  my_mean_element(BQ_seq[0:n+1])
    alpha_BQ =  Calculate_alpha_subspace_kernel_regression(BQ_kernel_matrix,f_values_BQ)   
    BQ_error_array[n] = evaluate_quadrature_on_unit_norm_function(alpha_BQ,BQ_seq[0:n+1],f_values_BQ,my_kernel_0)


### Plot number 4
plt.plot(N_nodes_list,np.log(HE_mean_error_array)/np.log(10), label= "DPPKQ",linestyle='-', marker='.',color='#003399')
plt.plot(N_nodes_list,np.log(DPP_loose_th_upper_bound_list)/np.log(10), label= "DPPKQ (UB)",linestyle='-', marker='.',color='#0099cc')
plt.plot(Herding_N_nodes_list,np.log(Herding_error_array)/np.log(10), label= "Herding",linestyle='-', marker='.',color = '#cc3300')
plt.plot(BQ_N_nodes_list,np.log(BQ_error_array)/np.log(10), label= "BQ",linestyle='-', marker='.',color='#993333')
plt.plot(MC_N_nodes_list,np.log(MC_mean_error_array)/np.log(10), label= "MC",linestyle='-', marker='.',color='red')
###
plt.legend( loc='lower left', bbox_to_anchor=(0,0), fontsize=11)
plt.xlabel(r'$\mathrm{N}$', fontsize=18)
plt.xticks(fontsize=16)
plt.ylabel(r'$\log_{10}$'+"(Squared error)", fontsize=18)
plt.yticks(fontsize=16)
if SAVE_PDF_FILES_FLAG ==1:
    plt.savefig('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+"_a_"+str(a)+'_fig_4.pdf')
plt.show()


### Plot number 6
plt.plot(N_nodes_list,np.log(HE_mean_error_array)/np.log(10), label= "DPPKQ",linestyle='-', marker='.',color='#003399')
plt.plot(N_nodes_list,np.log(DPP_loose_th_upper_bound_list)/np.log(10), label= "DPPKQ (UB)",linestyle='-', marker='.',color='#0099cc')
plt.plot(Herding_N_nodes_list,np.log(Herding_error_array)/np.log(10), label= "Herding",linestyle='-', marker='.',color = '#cc3300')
plt.plot(BQ_N_nodes_list,np.log(BQ_error_array)/np.log(10), label= "BQ",linestyle='-', marker='.',color='#993333')
plt.plot(MC_N_nodes_list,np.log(MC_mean_error_array)/np.log(10), label= "MC",linestyle='-', marker='.',color='red')
###
plt.legend( loc='lower left', bbox_to_anchor=(0,0), fontsize=13)
plt.xlabel(r'$\mathrm{N}$', fontsize=18)
plt.xticks(fontsize=16)
plt.ylabel(r'$\log_{10}$'+"(Squared error)", fontsize=18)
plt.yticks(fontsize=16)
if SAVE_PDF_FILES_FLAG ==1:
    plt.savefig('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+"_a_"+str(a)+'_fig_6.pdf')
plt.show()


### Save files
if SAVE_TXT_FILES_FLAG ==1:
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_DPPKQ.npy', HE_mean_error_array)    
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_DPPKQUB.npy', DPP_loose_th_upper_bound_list)  
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_N_List.npy', N_nodes_list)    
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_Herding.npy', Herding_error_array)    
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_BQ.npy', BQ_error_array)    
    np.save('Gaussian/Gaussian_kernel_scale_'+str(kernel_scale)+'_MC.npy', MC_error_array)
