import NNcpp
import minnie, minnie.Parallel
from minnie.Parallel import pprint as print

import mpi4py
from mpi4py import MPI
print("MPI.Is_initialized()", MPI.Is_initialized())
print("MPI.Get_library_version()", MPI.Get_library_version())
#print("cpp.mpi_size(), cpp.mpi_rank()", cpp.mpi_size(), cpp.mpi_rank()) # our pybind mpi wrapped functions
def test_mpi():

    print("BEFORE INIT")
    NNcpp.MPI_init()
    print("AFTER INIT")

    rank, size = NNcpp.MPI_get_rank_size()
    print("I'm rank {} size {}".format(rank, size))

if __name__ == "__main__":
    test_mpi()

        
