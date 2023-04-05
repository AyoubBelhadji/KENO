################################################
################################################

## Reader of files for the Sobolev RKHS simulations results

################################################
################################################


import sys
sys.path.insert(0, '..')
## Import useful classic functions
import matplotlib.pyplot as plt
import numpy as np
## Import the evaluations functions
from env.evaluations import *

### Activate a flag for saving the results files
SAVE_PDF_FILES_FLAG = 1 # 1 is yes 0 is no


### Load files
kernel_order = 1
Sobolev_folder = 'Sobolev/paper_figures/Sobolev_s'+str(kernel_order)+'/'


CUE_mean_error_array = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_DPPKQ.npy')
CUE_unbiased_mean_error_array = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_DPPUQ.npy')
iid_lvs_mean_error_array = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ0.npy')
reg_iid_lvs_mean_error_array_1 = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ1.npy')
reg_iid_lvs_mean_error_array_2 = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_LVSQ2.npy')
N_nodes_list = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_N_List.npy')
Herding_error_array = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_Herding.npy')
BQ_error_array = np.load(Sobolev_folder + 'Sobolev_kernel_order_'+str(kernel_order)+'_BQ.npy')




N_exp = 50

N_nodes_max = max(N_nodes_list)
N_nodes_list_shifted = [i-1 for i in N_nodes_list]
N_nodes_list_len = len(N_nodes_list)




Herding_N_nodes_list = list(range(N_nodes_max))
del Herding_N_nodes_list[0]
Herding_N_nodes_list.insert(N_nodes_max-1,N_nodes_max)

BQ_N_nodes_list = list(range(N_nodes_max))
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
    plt.savefig('Sobolev/reader_figures/Sobolev_kernel_order_'+str(kernel_order)+'_fig_1.pdf')
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
    plt.savefig('Sobolev/reader_figures/Sobolev_kernel_order_'+str(kernel_order)+'_fig_2.pdf')
plt.show()


