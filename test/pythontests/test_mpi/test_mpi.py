import NNcpp
import minnie, minnie.Parallel

def test_mpi():

    print("BEFORE INIT")
    NNcpp.MPI_init()
    print("AFTER INIT")

    rank, size = NNcpp.MPI_get_rank_size()
    print("I'm rank {} size {}".format(rank, size))

if __name__ == "__main__":
    test_mpi()

        
