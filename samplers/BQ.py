import numpy as np
from copy import deepcopy




def BQ_sequential_loss(old_seq,my_kernel,new_x,mu):
    local_kernel = deepcopy(my_kernel)
    [N] = old_seq.shape
    local_kernel.set_node(new_x)
    new_seq = np.zeros(N+1)
    new_seq[0:N] = old_seq
    new_seq[N] = new_x

    if N ==0:
        return -mu(new_x)
    else:
        kernel_matrix = local_kernel.get_kernel_matrix(new_seq)
        k_new_seq = np.zeros(N+1)
        for _ in list(range(N+1)):
            k_new_seq[_] = mu(new_seq[_])
        if np.linalg.matrix_rank(kernel_matrix)<N+1:
            return np.inf
        else:
            return  - np.dot(np.dot(k_new_seq,np.linalg.inv(kernel_matrix)),k_new_seq)



def BQ_get_sequence(N_,my_kernel,mu,queries_num):
    output_array = np.zeros(N_)
    seq_list = []
    for n in list(range(N_)):
        print("Bayesian quadrature " + str(n) + "-th node")
        pool_of_samples = np.random.normal(0, 1, queries_num)
        tmp_loss = BQ_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[0],mu)
        tmp_node = pool_of_samples[0]
        counter = 0
        
        for q in list(range(queries_num-1)):
            if BQ_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu) <tmp_loss:
                counter = counter +1
                tmp_loss = BQ_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu)
                tmp_node = pool_of_samples[q+1]
        seq_list.append(tmp_node)
        output_array[n] = tmp_node
    return output_array




def BQ_sequential_loss_Sobolev(old_seq,my_kernel,new_x,mu):
    local_kernel = deepcopy(my_kernel)
    [N] = old_seq.shape
    local_kernel.set_node(new_x)
    new_seq = np.zeros(N+1)
    new_seq[0:N] = old_seq
    new_seq[N] = new_x
    if N ==0:
        return 1
    else:
        kernel_matrix = local_kernel.get_kernel_matrix(new_seq)
        k_new_seq = np.ones(N+1)
        return  - np.dot(np.dot(k_new_seq,np.linalg.inv(kernel_matrix)),k_new_seq)




def BQ_get_sequence_Sobolev(N_,my_kernel,mu,queries_num):
    output_array = np.zeros(N_)
    seq_list = []
    for n in list(range(N_)):
        print("Bayesian quadrature " + str(n) + "-th node")
        pool_of_samples = np.random.uniform(0,1,queries_num)
        tmp_loss = BQ_sequential_loss_Sobolev(np.asarray(seq_list),my_kernel,pool_of_samples[0],mu)
        tmp_node = pool_of_samples[0]
        counter = 0
        for q in list(range(queries_num-1)):
            if BQ_sequential_loss_Sobolev(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu) <tmp_loss:
                counter = counter +1
                tmp_loss = BQ_sequential_loss_Sobolev(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu)
                tmp_node = pool_of_samples[q+1]  
        seq_list.append(tmp_node)
        output_array[n] = tmp_node
    return output_array



