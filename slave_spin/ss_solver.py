from sp_solver.solver import AbstractSubsidiaryProblem
from sp_solver.slave_spin.ss_structures import bit_test_spin, gen_Z_op
from sp_solver.slave_spin.ss_utils import schwinger_dressing, fermi_dist, avg_val
from scipy import sparse 
from copy import deepcopy
from scipy.optimize import root 


import numpy as np

class SlaveSpinSolver(AbstractSubsidiaryProblem):

    def __init__(self, h_int, gf_struct,beta=400, sigma_guess=None):
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

        ####parameters initialized to None
        ########################################
        # average energy of the fermions
        self.avg_en_list = None  
        # guess variables that get updated at every iteration
        self.lambdas_guess_list = None 
        self.Z_guess_list = None 
        # results of the solver
        self.lambdas_list = None 
        self.lambdas_kinetic_list = None 
        self.occup_list = None 
        self.Z_list = None
        ######################################### 

        #other
        self.beta=beta
        self.diag_method = 'numpy'
        self.num_eigvals = 10
    
    def update_guesses(self, **kwargs):
        allowed_args = {'avg_en_list', 'lambdas_guess_list', 'occup_list', 'Z_guess_list'}
        for key in kwargs:
            value = kwargs[key]
            if key in allowed_args:
                self.__setattr__(key, value)
            else:
                raise ValueError(f"SP_SOLVER-Slavespin: Trying to set forbidden key '{key}' in slave problem, please select one of the following: {allowed_args}") 



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
                self.h_int_sspin = h_int_sspin
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

    def _create_h_sspin(self,avg_en_list, Z_list, lambdas_list, occup_list,  in_place=False ):
        h_kin_sspin = self._create_h_kin_sspin(avg_en_list, Z_list, lambdas_list, occup_list, in_place=in_place)
        h_int_sspin = self._create_h_int_sspin(in_place=in_place)
        h_sspin = h_kin_sspin + h_int_sspin
        if in_place:
            self.h_sspin = h_sspin
        return h_sspin
    
    


    def diagonalize_h(self, method = "scipy", n_eigs=10, in_place = False):
        
        if method == 'numpy':
            H_tot =  self.h_sspin.toarray()
            eigvals, eigvecs = np.linalg.eigh(H_tot)
        elif method == 'scipy':
            H_tot =  self.h_sspin
            eigvals, eigvecs = sparse.linalg.eigsh(self.h_sspin, k=n_eigs, which = 'SA')

        if in_place:
            self.eigvals = eigvals
            self.eigvecs = eigvecs
        return eigvals, eigvecs

    def _evaluate_constraint(self, eigvals, eigvecs):
        """
        Evaluates the difference between the average number of
        bosons and fermions to enforce the constraint equation
        """
        n_op_list = self.N_list

        # n_bosons = np.array([avg_grnd(n_op, eigvals, eigvecs, parameters) for n_op in n_op_list]) 
        n_bosons = np.array([avg_val(n_op.toarray(), eigvals, eigvecs, beta=self.beta) for n_op in n_op_list]) 
        n_fermions = np.array(self.occup_list)

        return n_bosons-n_fermions
    


    def find_lambdas(self, Z_list=None, update_guess=True):
        """
        Partial evaluation of the costrain n function to accept the state parameters 
        and finds for which lambdas the costrain is satisfied

        look if worth putting method
        """
    # def _create_h_sspin(self,avg_en_list, Z_list, lambdas_list, occup_list,  in_place=False ):
        if Z_list is None:
            print("Setting guess to Z_list in slave_spin solver")
            Z_list = deepcopy(self.Z_guess_list)
        def constraint_equation(lambdas_guess):
            self._create_h_sspin(
                avg_en_list=self.avg_en_list,
                Z_list= Z_list,
                occup_list=self.occup_list,
                lambdas_list = lambdas_guess,
                in_place=True)
            self.diagonalize_h(in_place=True, method=self.diag_method)
            return self._evaluate_constraint(self.eigvals, self.eigvecs)


        solution = root(constraint_equation, self.lambdas_guess_list, method='krylov', tol=10e-5)

        self.lambdas_list = solution.x
        if update_guess:
            self.lambdas_guess_list = deepcopy(solution.x)
        return solution.x, solution.success, solution.message
    
    def iterate_solver(self, Z_guess_list):
        
        def dressing(iflavor):
            n_occup = self.occup_list[iflavor]
            P_plus_up = schwinger_dressing(n_occup-0.5)
            P_minus_up = schwinger_dressing(n_occup-0.5, sign=1)
            dress_up = P_minus_up*P_plus_up
            return dress_up
        
        lambdas, _, _ = self.find_lambdas(Z_guess_list)
        Z_new_list = np.array([dressing(iflavor)*avg_val(z_op.toarray(), self.eigvals, self.eigvecs, beta=self.beta) for iflavor, z_op in enumerate(self.Z_op_list)])

        

        
        return Z_new_list

        
    def solve(self):
        return 1
