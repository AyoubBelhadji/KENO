################################################
################################################

## Reader of files for the Gaussian kernel simulations results

################################################
################################################


import sys
sys.path.insert(0, '..')
## Import useful classic functions
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np


### Activate a flag for saving the results files
SAVE_PDF_FILES_FLAG = 1 # 1 is yes 0 is no


### Load files
kernel_scale = 0.5
Gaussian_folder = 'Gaussian/paper_figures/Gaussian_scale_0'+str(int(10*kernel_scale))+'/'
a = 0.25
b = 1/(2*np.power(kernel_scale,2))
c = np.sqrt(np.power(a,2)+2*a*b)
###
HE_mean_error_array = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_DPPKQ.npy')
DPP_loose_th_upper_bound_list = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_DPPKQUB.npy')
DPP_th_upper_bound_list = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_DPPKQCUB.npy')
N_nodes_list = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_N_List.npy')
Herding_error_array = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_Herding.npy')
BQ_error_array = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_BQ.npy')
MC_error_array = np.load(Gaussian_folder+'Gaussian_kernel_scale_'+str(kernel_scale)+'_MC.npy')
###
[N_nodes_max,_] = np.shape(MC_error_array)
MC_mean_error_array = np.zeros(N_nodes_max)
###
for n in list(range(N_nodes_max)):
    MC_mean_error_array[n] = np.mean(MC_error_array[:,n], axis = 0)
###
MC_N_nodes_list = list(range(N_nodes_max))
Herding_N_nodes_list = list(range(N_nodes_max))
BQ_N_nodes_list = list(range(N_nodes_max))


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
    plt.savefig('Gaussian/reader_figures/Gaussian_kernel_scale_'+str(kernel_scale)+"_a_"+str(a)+'_fig_4.pdf')
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
    plt.savefig('Gaussian/reader_figures/Gaussian_kernel_scale_'+str(kernel_scale)+"_a_"+str(a)+'_fig_6.pdf')
plt.show()