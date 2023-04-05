import sys
#sys.path.append('../')
sys.path.insert(0, '..')
##

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np
import math
#from opt import *
#from opt.OptEntropicDescent import *




    
class Bernoulli_Kernel_Function():
    def __init__(self,order,node):
        self.order = order
        self.node = node
        self.name = 'Bernoulli_kernel'
        self.kernel = Bernoulli_kernel(order)
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
    

class Bernoulli_Polynomial():
    def __init__(self,order):
        self.order = order
        self.coeffs = self.calculate_coeffs()
    def calculate_coeffs(self):
        if self.order == 0:
            return [1.0]
        #elif self.order ==1:
        #    return [1.0,-0.5]    
        elif self.order >0:
            tmp_polynomial_n_1 = Bernoulli_Polynomial(int(self.order-1))
            polynomial_n_1_coeffs = tmp_polynomial_n_1.coeffs
            tmp_array_poly_n_1 = np.asarray(polynomial_n_1_coeffs)
            reccurence_derivative_coeffs_for_high_coefficients = self.order*np.reciprocal(np.linspace(self.order,1,self.order))
            tmp_array_poly_n = np.multiply(reccurence_derivative_coeffs_for_high_coefficients,tmp_array_poly_n_1)
            
            reccurence_derivative_coeffs_for_coefficient_0 = np.reciprocal(np.linspace(self.order+1,2,self.order))
            coefficient_0 = -np.sum(np.multiply(reccurence_derivative_coeffs_for_coefficient_0,tmp_array_poly_n))
            polynomial_n_coeffs = list(tmp_array_poly_n)
            polynomial_n_coeffs.append(coefficient_0)
            #self.coeffs = list(tmp_array_sum)
            return polynomial_n_coeffs
    def evaluate_at_X(self,X):
        list_of_powers_of_X = [1]*(self.order+1)
        for o in list(range(self.order)):
            list_of_powers_of_X[o+1] = X*list_of_powers_of_X[o]
        list_of_powers_of_X = list(reversed(list_of_powers_of_X))
        tmp_mult_list = [a*b for a,b in zip(list_of_powers_of_X,self.coeffs)]
        return sum(tmp_mult_list)
def Bernoulli_kernel(order):
    def Bernoulli_kernel_aux(x,y):
        Bernoulli_polynomial_2_order = Bernoulli_Polynomial(2*order)
        frac_diff_x_y = (x-y)%1
        t = 1+ (np.power((-1),order-1)*np.power(2*math.pi,2*order)/(math.factorial(2*order)))*Bernoulli_polynomial_2_order.evaluate_at_X(frac_diff_x_y)
        return t
    return Bernoulli_kernel_aux

def get_kernel_matrix_Bernoulli_kernel(order,seq):
    [N] = seq.shape
    my_Bernoulli_kernel = Bernoulli_kernel(order)
    kernel_matrix = np.zeros((N,N))
    for n1 in list(range(N)):
        for n2 in list(range(N)):
            kernel_matrix[n1,n2] = my_Bernoulli_kernel(seq[n1],seq[n2])
    return kernel_matrix



def my_test_2():
    return 0
