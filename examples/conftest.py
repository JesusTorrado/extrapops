import os
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
IS_MAIN = not bool(RANK)

_output_folder = "output"


def ensure_output_folder():
    """
    Creates output folder if it does not exist.
    """
    if IS_MAIN:
        if not os.path.isdir(_output_folder):
            os.makedirs(_output_folder, exist_ok=True)
