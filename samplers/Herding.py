import numpy as np
from copy import deepcopy


def herding_sequential_loss(old_seq,my_kernel,new_x,mu):
    local_kernel = deepcopy(my_kernel)
    [N] = old_seq.shape
    local_kernel.set_node(new_x)
    k_new_x_seq = local_kernel.evaluate_at_I(old_seq)
    return -2*(N+1)*mu(new_x) + np.sum(k_new_x_seq)


def herding_get_sequence(N_,my_kernel,mu,queries_num):
    output_array = np.zeros(N_)
    seq_list = []
    for n in list(range(N_)):
        print("Herding " + str(n) + "-th node")
        pool_of_samples = np.random.normal(0, 1, queries_num)
        tmp_loss = herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[0],mu)
        tmp_node = pool_of_samples[0]
        for q in list(range(queries_num-1)):
            if herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu) <tmp_loss:
                tmp_loss = herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu)
                tmp_node = pool_of_samples[q+1]
        seq_list.append(tmp_node)
        output_array[n] = tmp_node
    return output_array

def herding_get_sequence_Sobolev(N_,my_kernel,mu,queries_num):
    output_array = np.zeros(N_)
    seq_list = []
    for n in list(range(N_)):
        print("Herding " + str(n) + "-th node")
        pool_of_samples = np.random.uniform(0,1,queries_num)
        tmp_loss = herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[0],mu)
        tmp_node = pool_of_samples[0]
        for q in list(range(queries_num-1)):
            if herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu) <tmp_loss:
                tmp_loss = herding_sequential_loss(np.asarray(seq_list),my_kernel,pool_of_samples[q+1],mu)
                tmp_node = pool_of_samples[q+1]
        seq_list.append(tmp_node)
        output_array[n] = tmp_node
    return output_array
