from sp_solver.solver import AbstractSubsidiaryProblem
from sp_solver.slave_spin.ss_structures import bit_test_spin, gen_Z_op
from sp_solver.slave_spin.ss_utils import schwinger_dressing, fermi_dist, avg_val
from scipy import sparse 
from scipy.optimize import root 

import numpy as np

class SlaveSpinSolver(AbstractSubsidiaryProblem):

    def __init__(self, h_int, gf_struct, sigma_guess=None):
        super().__init__(h_int, gf_struct, sigma_guess)
        
        """
        :todo: think whether to bring this part of the initialization
        to the abstract class
        """
        self.idx_list = [ (bl_name, iflavor)  for ibl, bl_name in enumerate(self.block_names) for iflavor in range(self.block_mult[ibl])] 

        # invert relationship between spin and state index
        self.idx_dict = { member[0]:{} for member in self.idx_list }
        for idx, idx_tuple in enumerate(self.idx_list):
            self.idx_dict[idx_tuple[0]][idx_tuple[1]]=idx
        
        self.n_states = len(self.idx_list) 
        self.dims = 2**self.n_states
        
        # For the moment no gauge is implemented, schwinger dressing
        # formulation is used
        self.Z_dag_list = [sparse.csr_matrix(gen_Z_op(0, n_states= self.n_states, idx=i)) for i in range(self.n_states)]
        self.Z_op_list = [Z_op.transpose(copy=True) for Z_op in self.Z_dag_list]
        self.N_list = [self.Z_op_list[iz].dot(self.Z_dag_list[iz]) for iz, _ in enumerate(self.Z_dag_list)]
    



    def _create_h_int_sspin(self, in_place=False):
        """
        Creates a slave spin hamiltonian starting from the triqs interaction hamiltonian
        """
        h_int_sspin = sparse.csc_matrix((self.dims,self.dims), dtype=float) 
        for term, coeff in self.h_int:
            terms_dict = {0:{}, 1:{},2:{},3:{}}

            for ic, c_op in enumerate(term):
                is_dagger = c_op[0]
                spin_idx = c_op[1][0]
                orb_idx = c_op[1][1]
                abs_idx = self.idx_dict[spin_idx][orb_idx]
                
                if is_dagger:
                    op_matrix = self.Z_dag_list[abs_idx]
                else:
                    op_matrix = self.Z_op_list[abs_idx]

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
            if in_place:
                self.h_kin_sspin = h_int_sspin
        return h_int_sspin
    
    def _create_h_kin_sspin(self, avg_en_list, Z_list, lambdas_list, occup_list, in_place=False):

        h_kin_sspin = sparse.csc_matrix((self.dims,self.dims), dtype=float)
        

        # add kinetic term
        for iflavor in range(self.n_states):
            n_occup = occup_list[iflavor]
            P_plus_up = schwinger_dressing(n_occup-0.5)
            P_minus_up = schwinger_dressing(n_occup-0.5, sign=1)
            dress_up = P_minus_up*P_plus_up

            kin_term = dress_up*Z_list[iflavor]* avg_en_list[iflavor]*self.Z_dag_list[iflavor]
            kin_term += dress_up*np.conjugate(Z_list[iflavor])*avg_en_list[iflavor]*self.Z_op_list[iflavor]

            h_kin_sspin += kin_term

        # add lambdas
        for iflavor in range(self.n_states):
            lambda_term = lambdas_list[iflavor]*self.N_list[iflavor]

            h_kin_sspin += lambda_term

        if in_place:
            self.h_kin_sspin = h_kin_sspin
        return h_kin_sspin

    # def _create_h_sspin(self):

    def diagonalize_h(self, lambdas_list):
        H_free = create_single_part_term(Z_list, lambdas_list, avg_en_list) 
        H_int = h_int_sspin
        H_tot =  (H_free+H_int).toarray()
        
        eigvals, eigvecs = np.linalg.eigh(H_tot)
        eigvals = eigvals-min(eigvals) #for boltzmann normalization

        return H_tot, eigvals, eigvecs

    def evaluate_constraint(self, lambdas_list, n_fermions):
        """
        Evaluates the difference between the average number of
        bosons and fermions to enforce the constraint equation
        """
        _, eigvals, eigvecs = diagonalize_h(lambdas_list)
        n_op_list = N_list

        # n_bosons = np.array([avg_grnd(n_op, eigvals, eigvecs, parameters) for n_op in n_op_list]) 
        n_bosons = np.array([avg_val(n_op.toarray(), eigvals, eigvecs, beta=400) for n_op in n_op_list]) 
        n_fermions = np.array(n_fermions)

        return n_bosons-n_fermions

    def find_lambdas(self, lambdas_guess, n_fermions):
        """
        Partial evaluation of the costrain n function to accept the state parameters 
        and finds for which lambdas the costrain is satisfied

        look if worth putting method
        """
        
        solution = root(lambda x: self.evaluate_constraint(x, n_fermions), lambdas_guess, method='krylov', tol=10e-5)
        return solution.x, solution.success, solution.message
    def solve(self):
        return 1
