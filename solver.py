from abc import ABC, abstractmethod
class GeneralSolver(object):
#not now
    def __init__(self, h_int):
        pass

class AbstractSubsidiaryProblem(ABC):
    """
    Class that gets all the general stuff needed for all
    mean field subsidiary problems 
    """

    def __init__(self, h_int, gf_struct, sigma_guess=None):
        self.h_int = h_int
        self.gf_struct = gf_struct
        self.sigma_guess = sigma_guess
        self.block_names = [bl[0] for bl in self.gf_struct]
        self.block_mult = [bl[1] for bl in self.gf_struct]



    @abstractmethod
    def solve(self):
        pass