
from scipy import sparse
import numpy as np
#%%
def bit_test_spin(state, index):
    """
    Checks whether the index-th position is spin up.

    Beautiful solution from Oscar Najera,
    the state number (orbital+spin) in binary representation
    has 1 whenever the state is occupied

    The bit test evaluates whether there is a 1 in the index position
    Shifts 'state' in binary by 'index' positions and looks if the last
    bit is a 1
    
    Example:
    * The state 15 -> 1111 has all states occupied;
                    bit_test_spin(15, 0)=True 
    * The state  5 -> 0101 has both spin up states occupied;
                    bit_test_spin(5, 0)=True
    * The state  2 -> 0010 has only the first spin down state occupied;
                    bit_test_spin(2, 0)=False
    """
    return (state >> index) & 1

def gen_Z_op(c_gauge, n_states, idx):
    """
    Creates a general z operator as described in
    c_gauge: float, gauge degree of freedom. 0.0 gives spin_down
    n_states: integer, number of total states in the system
    idx: par
    """
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