import numpy as np
import math


def CUE_seq_generator(N_):
    x1 = np.random.randn(N_, N_)
    x2 = np.random.randn(N_, N_)
    x = (x1+x2*1j)/np.sqrt(2)
    q,r = np.linalg.qr(x)
    r = np.diag(np.divide(np.diag(r),np.abs(np.diag(r))))
    u = np.dot(q,r)
    s,_ = np.linalg.eig(u)
    s_angle = (np.angle(s,deg=False)/math.pi +np.ones(N_))/2
    return s_angle

