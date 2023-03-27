
import numpy as np
#%%
def find_gauge(n):
    return (1/(np.sqrt(n*(1-n))) - 1 )

def fermi_dist(energy, beta):
    """ Fermi Dirac distribution"""
    exponent = np.asarray(beta*energy).clip(-600, 600)
    return 1./(np.exp(exponent) + 1)

def schwinger_dressing(Sz_avg, sign =0, delta = 0.000001):
    P_avg1 = 1/np.sqrt(0.5 + delta + (-1)**(sign)*Sz_avg)  
    return P_avg1

def avg_val(Op,eigvals, eigvecs, beta):
    eigvals = eigvals-min(eigvals)
    boltz = np.diag([np.exp(-beta*eig) for eig in eigvals])
    O_eig_basis = np.matmul(np.transpose(eigvecs), np.matmul(Op, eigvecs))
    z_partition =  np.trace(boltz)
    O_avg = 1/z_partition*np.trace(O_eig_basis.dot(boltz))
    return O_avg