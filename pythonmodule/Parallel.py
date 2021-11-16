import NNcpp


# Initialize MPI
NNcpp.MPI_init()


def am_i_the_master():
    rank, size = NNcpp.MPI_get_rank_size()
    if rank == 0:
        return True
    return False

def pprint(*args, **kwargs):
    if am_i_the_master():
        print(*args, **kwargs)
    