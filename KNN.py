import numpy as np
import pandas as pd 
from mpi4py import MPI 
from tqdm import tqdm

def Minkovskiy(x, y, p = 2):
	return np.sum(abs(x - y) ** p) ** (1 / p)

# 参数:
K = 5 #  邻居个数
P = 2 #  Minkovskiy 距离的 p 值
VecLen = 784 #  特征向量长度

if __name__ == "__main__":
	# MPI 初始化
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()


	train_pd = pd.read_csv("train.csv")
	# train_np[0]: 标签， train_np[1:]: 特征向量
	train_np = train_pd.values

	total = 0
	_total = 0


	# 在 MPI 进程间分配数据 
	if rank == 0:
		test_pd = pd.read_csv("test.csv")
		test_np_send = np.array(test_pd.values.copy(), dtype='i')	
		_total = len(test_np_send)
		if total % size != 0:
			print("Fatal Error: total % size != 0")
			comm.Abort()
		# 在最后一行补 0， 使得数据个数能够被 size 整除 
		total = (_total + size - 1) // size * size  
		test_np_send.resize([size, total // size, VecLen])
	else:
		test_np_send = None

	total = comm.bcast(total, root=0)

	test_np_recv = np.empty([total // size, VecLen], dtype='i')
	comm.Scatter(test_np_send, test_np_recv, root = 0)
	
	# 类型: python array
	res_send = []

	for point in test_np_recv:
		distances = np.array([[Minkovskiy(point, x[1:]), x[0]] for x in train_np])
		distances = distances[distances[:, 0].argsort()][:K]
		# 统计 label 0～9 在邻近点出现了几次
		tmp = np.array([[i, np.sum(distances[..., 1] == i)] for i in range(0, 10)]) 
		# 找出出现最多的 label
		max_likelihood = tmp[:, 1].argmax() 
		res_send.append(max_likelihood)


	if rank == 0:
		res_recv = np.empty([total], dtype='i')
	else:
		res_recv = None

	# 将结果收集到 rank 0 进程
	comm.Gather(np.array(res_send, dtype='i'), res_recv, root = 0)
	
	if rank == 0:
		data = np.array([[i + 1 for i in range(_total)], res_recv[:_total]])
		tmp = pd.DataFrame(data=data.T, columns=["ImageId", "Label"])
		tmp.to_csv("knn-ans.csv", index=False)