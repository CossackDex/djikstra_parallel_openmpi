import scipy
import numpy as np
from mpi4py import MPI


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # A B C D E
        graph = np.array([[np.inf, 2, np.inf, 4, np.inf],
                          [np.inf, 3, 3, np.inf, np.inf],
                          [np.inf, np.inf, np.inf, np.inf, 2],
                          [np.inf, 2, np.inf, 4, np.inf],
                          [np.inf, 2, np.inf, 4, np.inf]])
