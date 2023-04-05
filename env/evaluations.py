import numpy as np

def evaluate_function_at_interval(f,I):
    N = len(I)
    f_list = [0]*N
    for i in list(range(N)):
        f_list[i] = f(I[i])
    return f_list



def evaluate_quadrature_on_unit_norm_function_2(weights,nodes,my_function,my_kernel):
    # This function evaluate the RKHS norm defined by a kernel of a quadrature  
    # weights is the array of weights
    # nodes is the array of nodes
    # my_function is the function
    # my_kernel is the kernel
    [N] = nodes.shape
    function_evals = my_function.evaluate_at_I(nodes)
    kernel_matrix = my_kernel.get_kernel_matrix(nodes)
    output_var = np.dot(np.dot(weights.T,kernel_matrix),weights) - 2*np.dot(function_evals,weights)
    return output_var




def evaluate_quadrature_on_unit_norm_function(weights,nodes,function_evals,my_kernel):
    # This function evaluate the RKHS norm defined by a kernel of a quadrature  
    # weights is the array of weights
    # nodes is the array of nodes
    # my_function is the function
    # my_kernel is the kernel
    [N] = nodes.shape

    kernel_matrix = my_kernel.get_kernel_matrix(nodes)

    output_var = max(1 - 2*np.dot(weights,function_evals) + np.dot(np.dot(weights.T,kernel_matrix),weights),0)
    return output_var




def evaluate_regression_function_at_interval(I,alpha,my_kernel,seq):
    [N] = seq.shape
    [M] = I.shape
    
    evaluation_matrix = np.zeros((N,M))
    for n in list(range(N)):
        for m in list(range(M)):
            evaluation_matrix[n,m] = my_kernel(seq[n],I[m])
    
    evaluation_vector = np.dot(alpha,evaluation_matrix)
    return evaluation_vector





def get_correlation(vector_1,vector_2):
    [N] = vector_1.shape
    matrix_1 = np.ones((N,2))
    matrix_2 = np.ones((N,2))
    matrix_1[:,1] = vector_1
    matrix_2[:,1] = vector_2
    correlation_vector = np.dot(np.linalg.pinv(matrix_1),matrix_2)
    #correlation_coefficient = np.power(1/np.linalg.norm(vector_1),2)*np.dot(vector_1,vector_2)
    #return correlation_coefficient
    return correlation_vector[1,1]

