from nash_equilibrium_pkg import nash
import numpy as np

if __name__ == "__main__":
    a = np.array([
        [4,0,6,2,2,1],
        [3,8,4,10,4,4],
        [1,2,6,5,0,0],
        [6,6,4,4,10,3],
        [10,4,6,4,0,9],
        [10,7,0,7,9,8]])

    nash.nash_equilibrium(a)
