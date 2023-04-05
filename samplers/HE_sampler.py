import numpy as np



def HE_seq_generator(N_,scale):
    x1 = np.random.randn(N_, N_)
    x2 = np.random.randn(N_, N_)
    x = x1+x2*1j
    y = (x + np.conjugate(np.transpose(x)))/2
    s,_ = np.linalg.eigh(y)
    return s/np.sqrt(scale)






