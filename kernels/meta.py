import numpy as np



def Calculate_alpha_subspace_kernel_regression(K,f_values):
    # K is NxN kernel matrix evaluated at the nodes
    # f_values is the vector of the evaluation of f at the N nodes
    # N is the number of nodes
    [N,_] = K.shape
    K_inv = np.linalg.pinv(K)
    alpha = np.dot(K_inv,f_values)
    return alpha
    

def Calculate_alpha_ridge_kernel_regression(K,f_values,lambda_reg):
    # K is NxN kernel matrix evaluated at the nodes
    # f_values is the vector of the evaluation of f at the N nodes
    # N is the number of nodes
    [N,_] = K.shape
    K_reg = K + lambda_reg*np.eye(N)
    K_reg_inv = np.linalg.inv(K_reg)
    alpha = np.dot(K_reg_inv,f_values)
    return alpha



