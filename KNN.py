import numpy as np
import pandas as pd # type: ignore
from mpi4py import MPI # type: ignore
from tqdm import tqdm # type: ignore

def Minkovskiy(x, y, p = 2):
	return np.sum(abs(x - y) ** p) ** (1 / p)

# parameter:
K = 5 #  how many nearest neighbors
P = 2 #  parameter of Minkovskiy's distance
VecLen = 784 #  length of feature vector

if __name__ == "__main__":
	# MPI setup
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()


	train_pd = pd.read_csv("train.csv")
	train_np = train_pd.values 	# train_np[0]: label + train_np[1:]: feature

	total = 0
	_total = 0


	# MPI distribute workloads
	if rank == 0:
		test_pd = pd.read_csv("test.csv")
		test_np_send = np.array(test_pd.values.copy(), dtype='i')	# test_np[:]: feature
		_total = len(test_np_send)
		if total % size != 0:
			print("Fatal Error: total % size != 0")
			comm.Abort()
		# padding 0 after test dataset
		total = (_total + size - 1) // size * size  
		test_np_send.resize([size, total // size, VecLen])
	else:
		test_np_send = None

	total = comm.bcast(total, root=0)

	test_np_recv = np.empty([total // size, VecLen], dtype='i')
	print("size of test_np_recv:" ,test_np_recv.shape)
	comm.Scatter(test_np_send, test_np_recv, root = 0)
	res_send = []

	with tqdm(total=len(test_np_recv)) as bar:
		for point in test_np_recv:
			distances = np.array([[Minkovskiy(point, x[1:]), x[0]] for x in train_np])
			distances = distances[distances[:, 0].argsort()][:K]
			# 统计 label 0～9 在邻近点出现了几次
			tmp = np.array([[i, np.sum(distances[..., 1] == i)] for i in range(0, 10)]) 
			max_likelihood = tmp[:, 1].argmax() # 找出出现最多的 label
			res_send.append(max_likelihood)
			bar.update(1)
	if rank == 0:
		res_recv = np.empty([total], dtype='i')
	else:
		res_recv = None

	comm.Gather(np.array(res_send, dtype='i'), res_recv, root = 0)
	
	if rank == 0:
		data = np.array([[i + 1 for i in range(_total)], res_recv[:_total]])
		tmp = pd.DataFrame(data=data.T, columns=["ImageId", "Label"])
		tmp.to_csv("knn-ans.csv", index=False)