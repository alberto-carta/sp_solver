#%%
import numpy as np
from triqs.operators.util.hamiltonians import h_int_kanamori, h_int_density,h_int_slater, diagonal_part
from triqs.operators.util import U_matrix_kanamori
from triqs.operators.operators import Operator
from triqs.operators.util.observables import S_op, c_dag, n
import triqs.operators.util.extractors as ex

# %%
n_orb=2
# gf_struct =  [('down',range(n_orb)), ('up',range(n_orb))] 
gf_struct =  [('down',n_orb), ('up',n_orb)] 
spin_names = ["up","down"]

isospin_names = ["on","off"]
orb_names = [i for i in range(n_orb)]
U = 4.0
J = 0.0

#%%
Umat=U_matrix_kanamori(n_orb, U, J)
# Umat=U_matrix_kanamori(n_orb, U, J)
h_int = h_int_density(spin_names, orb_names,off_diag=True, U=Umat[0], Uprime=Umat[1])
# h_int = h_int_kanamori(spin_names, orb_names,off_diag=True, U=Umat[0], Uprime=Umat[1])


        
# %%
idx_list = [ (ispin, iorb)  for ispin in spin_names for iorb in range(n_orb)] 

# invert relationship between spin and state index
idx_dict = { member[0]:{} for member in idx_list }
for idx, idx_tuple in enumerate(idx_list):
    idx_dict[idx_tuple[0]][idx_tuple[1]]=idx


def bit_test_spin(state, index):
    """
    Beautiful solution from Oscar Najera,
    the state number (orbital+spin) in binary representation
    has 1 whenever the state is occupied

    The bit test evaluates whether there is a 1 in the index position
    Shifts 'state' in binary by 'index' positions and looks if the last
    bit is a 1
    
    Example:
    The state 15 -> 1111 has all states occupied: 
    The state  5 -> 0101 has both spin up states occupied
    The state  2 -> 0010 has only the first spin down state occupied
    """
    return (state >> index) & 1

def gen_Z_op(c_gauge, n_states, idx):
    if idx >= n_states:
        raise ValueError("The provided index does not match with the number of states")
    
    dims =  2**n_states
    # The effect of O_op is to flip the spin in a particular state
    # multiply by 1 if you go from 0 --> 1 and 
    # multiply by the gauge if you go from 1 --> 0
    # so the states connected by O_op are related by a bitflip
    # on the idx-th position 
    
    bitflip_pos = (1 << idx)
    bitflip = lambda x: x ^ bitflip_pos


    occupied_states = list(map(lambda x:bit_test_spin(x, idx), range(dims)))
    #this determines when you have a one in the idx position
    connected_states =enumerate(map(bitflip, range(dims)))
    
    O_op = np.zeros((dims, dims))
    for i, j in connected_states:
        if occupied_states[i]:
            O_op[j, i] = 1
        else:
            O_op[j, i] = c_gauge
    return O_op
#%%
from scipy import sparse
n_states = 2 * n_orb
dim = 2**n_states

#   sparse
Z_dag_list = [sparse.csr_matrix(gen_Z_op(0, n_states= n_states, idx=i)) for i in range(n_states)]
Z_op_list = [Z_op.transpose(copy=True) for Z_op in Z_dag_list]
N_list = [Z_op_list[iz].dot(Z_dag_list[iz]) for iz, _ in enumerate(Z_dag_list)]

#   non sparse
# Z_dag_list = [gen_Z_op(0, n_states= dim, idx=i) for i in range(dim)]
# Z_op_list = [Z_op.T for Z_op in Z_dag_list]
# %% interaction hamiltonian
# h_int_sspin = sparse.csc_matrix((dim,dim), dtype=complex) 
h_int_sspin = sparse.csc_matrix((dim,dim), dtype=float) 

for term, coeff in h_int:
    terms_dict = {0:{}, 1:{},2:{},3:{}}

    for ic, c_op in enumerate(term):
        is_dagger = c_op[0]
        spin_idx = c_op[1][0]
        orb_idx = c_op[1][1]
        abs_idx = idx_dict[spin_idx][orb_idx]
        
        if is_dagger:
            op_matrix = Z_dag_list[abs_idx]
        else:
            op_matrix = Z_op_list[abs_idx]

        terms_dict[ic]['spin_idx']=spin_idx
        terms_dict[ic]['orb_idx']=orb_idx
        terms_dict[ic]['abs_idx']=abs_idx
        terms_dict[ic]['op_matrix']= op_matrix

    for ic in range(4):
        if ic == 0:
            h_int_term = terms_dict[ic]['op_matrix']
        else:
            h_int_term = terms_dict[ic]['op_matrix'].dot(h_int_term)
    
    h_int_sspin += coeff*h_int_term

print(np.diag(h_int_sspin.toarray()))

    
# %% kill sparsity here
from slavesolver.functions import find_gauge, schwinger_dressing, get_h

Z_list = [1.0 for i in range(n_states)] 
lambdas_list = [1.0*0 for i in range(n_states)] 
avg_en_list = [-0.21221 for i in range(n_states)]

n_occup =0.5
def create_single_part_term(Z_list, lambdas_list, avg_en_list):

    h_kin_sspin = sparse.csc_matrix((dim,dim), dtype=float)
    
    P_plus_up = schwinger_dressing(n_occup-0.5)
    P_minus_up = schwinger_dressing(n_occup-0.5, sign=1)
    dress_up = P_minus_up*P_plus_up

    # add kinetic term
    for iflavor in range(n_states):
        kin_term = dress_up*Z_list[iflavor]* avg_en_list[iflavor]*Z_dag_list[iflavor]
        kin_term += dress_up*np.conjugate(Z_list[iflavor])*avg_en_list[iflavor]*Z_op_list[iflavor]

        h_kin_sspin += kin_term

    # add lambdas
    for iflavor in range(n_states):
        lambda_term = lambdas_list[iflavor]*N_list[iflavor]

        h_kin_sspin += lambda_term
    return h_kin_sspin

h_kin_sspin = create_single_part_term(Z_list, lambdas_list, avg_en_list) 
# %%
from scipy.optimize import root

def avg_val(Op,eigvals, eigvecs, beta):
    # boltz = np.tile([np.exp(-beta*eig) for eig in eigvals],(dim,1))
    boltz = np.diag([np.exp(-beta*eig) for eig in eigvals])
    O_eig_basis = np.matmul(np.transpose(eigvecs), np.matmul(Op, eigvecs))
    z_partition =  np.trace(boltz)
    O_avg = 1/z_partition*np.trace(O_eig_basis.dot(boltz))
    return O_avg

def diagonalize_h(lambdas_list):
    H_free = create_single_part_term(Z_list, lambdas_list, avg_en_list) 
    H_int = h_int_sspin
    H_tot =  (H_free+H_int).toarray()
    
    eigvals, eigvecs = np.linalg.eigh(H_tot)
    eigvals = eigvals-min(eigvals) #for boltzmann normalization

    return H_tot, eigvals, eigvecs

def constrain_n(lambdas_list, n_fermions):
    """
    Evaluates the difference between the average number of
    bosons and fermions to enforce the constrain equation
    """
    _, eigvals, eigvecs = diagonalize_h(lambdas_list)
    n_op_list = N_list

    # n_bosons = np.array([avg_grnd(n_op, eigvals, eigvecs, parameters) for n_op in n_op_list]) 
    n_bosons = np.array([avg_val(n_op.toarray(), eigvals, eigvecs, beta=400) for n_op in n_op_list]) 
    n_fermions = np.array(n_fermions)

    return n_bosons-n_fermions

def find_lambdas(lambdas_guess, n_fermions):
    """
    Partial evaluation of the costrain n function to accept the state parameters 
    and finds for which lambdas the costrain is satisfied

    look if worth putting method
    """
    
    solution = root(lambda x: constrain_n(x, n_fermions), lambdas_guess, method='krylov', tol=10e-5)
    return solution.x, solution.success, solution.message
# %%
guess = [-1]*n_states
target_pop = [0.5]*n_states
x = find_lambdas(lambdas_guess=guess, n_fermions=target_pop)


final_lambdas = x[0]
print(f"{final_lambdas=}")
h_final, eigvals, eigvecs = diagonalize_h(final_lambdas)


P_plus_up = schwinger_dressing(n_occup-0.5)
P_minus_up = schwinger_dressing(n_occup-0.5, sign=1)
dress_up = P_minus_up*P_plus_up

Z_avg = avg_val(Z_op_list[0].toarray(), eigvals, eigvecs, 400)*dress_up

print(Z_avg)

# %%
