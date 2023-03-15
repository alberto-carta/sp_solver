#%%
import numpy as np

#%%
def avg_val(Op,eigvals, eigvecs, beta):
    # boltz = np.tile([np.exp(-beta*eig) for eig in eigvals],(dim,1))
    boltz = np.diag([np.exp(-beta*eig) for eig in eigvals])
    O_eig_basis = np.matmul(np.transpose(eigvecs), np.matmul(Op, eigvecs))
    z_partition =  np.trace(boltz)
    O_avg = 1/z_partition*np.trace(O_eig_basis.dot(boltz))
    return O_avg