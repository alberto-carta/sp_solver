#%%
import numpy as np

from triqs.operators.util.hamiltonians import h_int_kanamori, h_int_density,h_int_slater, diagonal_part
from triqs.operators.util import U_matrix_kanamori
from  sp_solver.slave_spin.ss_solver import SlaveSpinSolver

# %%
n_orb=2
gf_struct =  [('down',n_orb), ('up',n_orb)] 
spin_names = ["down","up"]
orb_names = [i for i in range(n_orb)]
U = 4.0
J = 0.0

#%% generate density density matrix from triqs
Umat=U_matrix_kanamori(n_orb, U, J)
h_int = h_int_density(spin_names, orb_names,off_diag=True, U=Umat[0], Uprime=Umat[1])

ss_problem = SlaveSpinSolver(h_int=h_int, gf_struct=gf_struct)
#%%Debugging
print(f"{ss_problem.block_names=}")
print(f"{ss_problem.block_mult=}")
print(f"{ss_problem.idx_list=}")
print(f"{ss_problem.idx_dict=}")
print(f"{ss_problem.n_states=}")
print(f"{ss_problem.dims=}")
# %%
ss_problem.create_h_int_sspin()
# print(np.diag(ss_problem.h_int_sspin.toarray()))

# %%

Z_list = [1.0 for i in range(ss_problem.n_states)] 
lambdas_list = [1.0*0 for i in range(ss_problem.n_states)] 
occup_list = [0.5 for i in range(ss_problem.n_states)] 
avg_en_list = [-0.21221 for i in range(ss_problem.n_states)]



ss_problem.create_h_kin_sspin(Z_list=Z_list,
                              avg_en_list=avg_en_list,
                              lambdas_list=lambdas_list,
                              occup_list=occup_list)