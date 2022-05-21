import NNcpp

# Perform the MPI initialization
import mpi4py
from mpi4py import MPI

def am_i_the_master():
    rank, size = NNcpp.MPI_get_rank_size()
    if rank == 0:
        return True
    return False

def pprint(*args, **kwargs):
    if am_i_the_master():
        print(*args, **kwargs)
    