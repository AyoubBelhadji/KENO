
import numpy as np


    
class Gaussian_Kernel_Function():
    def __init__(self,scale,node):
        self.scale = scale
        self.node = node
        self.name = 'Gaussian_kernel'
        self.kernel = Gaussian_kernel(scale)
    def set_node(self,x):
        self.node = x
    def evaluate_at_x(self,x):
        return self.kernel(self.node,x)
    def evaluate_at_I(self,I):
        [N] = I.shape
        evaluation_vector= np.zeros(N)
        for n in list(range(N)):
            evaluation_vector[n] = self.evaluate_at_x(I[n])
        return np.asarray(evaluation_vector)
    def get_kernel_matrix(self,nodes):
        [N] = nodes.shape
        kernel_matrix = np.zeros((N,N))
        for n1 in list(range(N)):
            for n2 in list(range(N)):
                kernel_matrix[n1,n2] = self.kernel(nodes[n1],nodes[n2])
        return kernel_matrix
    

def Gaussian_kernel(scale):
    def Gaussian_kernel_aux(x,y):
        scaled_diff_x_y = np.power(x-y,2)/(2*np.power(scale,2))
        return np.exp(-scaled_diff_x_y)
    return Gaussian_kernel_aux

def get_kernel_matrix_Gaussian_kernel(scale,seq):
    [N] = seq.shape
    my_Gaussian_kernel = Gaussian_kernel(scale)
    kernel_matrix = np.zeros((N,N))
    for n1 in list(range(N)):
        for n2 in list(range(N)):
            kernel_matrix[n1,n2] = my_Gaussian_kernel(seq[n1],seq[n2])
    return kernel_matrix



def mean_element(a,scale):
    def mean_element_aux(x):
        b = 1/(2*np.power(scale,2))
        c = np.sqrt(np.power(a,2)+2*a*b)
        A = a+b+c
        alpha = np.sqrt(np.sqrt(c/a))*np.sqrt(np.sqrt(2*a/A))
        #print(mean_norm)
        return alpha*np.exp(-np.power(x,2)*(c-a))
    return mean_element_aux


def DPP_loose_th_upper_bound(I,a,scale):
    N = len(I)
    output_var = [0]*N
    ## eigenvalue of the operator
    #a = 1/2
    b = 1/(2*np.power(scale,2))
    #b= 1/(2*scale)
    c = np.sqrt(np.power(a,2)+2*a*b)
    A = a+b+c
    B = b/A
    alpha_factor = np.sqrt(2*a/A)
    for n in list(range(N)):
        output_var[n] = alpha_factor*np.power(B,I[n])*I[n]/(1-B)
    return output_var  