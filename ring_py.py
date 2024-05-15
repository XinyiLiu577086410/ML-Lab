from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

next_proc = (rank + 1) % size
prev_proc = (rank + size - 1) % size

tag = 2
message = 10

if 0 == rank:
    comm.send(message, dest=next_proc, tag=tag)

while(1):
    message = comm.recv(source=prev_proc, tag=tag)  
    comm.Recv_init

    if 0 == rank:
        message = message - 1
        print ("Process %d decremented value: %d\n"%(rank, message))

    comm.send(message, dest=next_proc, tag=tag)

    if 0 == message:
        print ("Process %d exiting\n" %(rank))
        break

if 0 == rank:
    message = comm.recv(source=prev_proc, tag=tag)