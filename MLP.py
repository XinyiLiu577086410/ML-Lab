import torch 
from torch import nn	
from torch.utils.data import TensorDataset, DataLoader
import tqdm as tqdm
import pandas as pd
       

def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
		nn.init.zeros_(tensor=m.bias)


def accuracy(y_hat, y):
	"""计算预测正确的数量"""
	y_hat = y_hat.argmax(axis=1)
	y = y.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())

class Accumulator:
	"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n

	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]

	def reset(self):
		self.data = [0.0] * len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, lr, loss):
	updater = torch.optim.SGD(params=net.parameters(), lr=lr)
	metric = Accumulator(3)  # 记录总损失，总TP，数据数
	for epoch in range(num_epochs):
		# for X, y in tqdm.tqdm(train_iter):
		for X, y in train_iter:
			X = X.to(device)
			y = y.to(device)
			y_hat = net(X)
			batch_loss = loss(y_hat, y)
			updater.zero_grad()
			batch_loss.mean().backward()
			updater.step()
			metric.add(float(batch_loss.sum()), accuracy(y_hat, y), y.numel()/10)
		aver_loss = metric[0] / metric[2]
		aver_acc = metric[1] / metric[2]
		test_acc = aver_acc
		print("Epoch: %d, Loss: %.4f, Acc: %.4f"%(epoch, aver_loss, aver_acc))
		# print("%d & %.4f & %.4f \\\\"%(epoch, aver_loss, aver_acc))
		metric.reset()


def ans(net, eval_iter):
	with open("mlp-ans.csv", "w") as f:	
		X = test_input.to(device)
		y_hat = net(X)
		y_hat = y_hat.argmax(axis=1)
		df = pd.DataFrame(columns=['ImageId', 'Label'])
		df['ImageId'] = range(1, len(y_hat)+1)
		df['Label'] = y_hat.cpu().numpy()
		df.to_csv(f, header=True, index=False)


if __name__ == "__main__":
	# hyperparameters
	batch_size = 256
	num_epochs = 10
	learning_rate = 0.05

	# load data
	train_data = pd.read_csv("train.csv")
	test_data = pd.read_csv("test.csv")
	train_data_value = train_data.values
	test_data_value = test_data.values
	train_input = torch.tensor(data=train_data_value[:,1:], dtype=torch.float32)
	test_input = torch.tensor(data=test_data_value, dtype=torch.float32)
	train_label = torch.tensor(data=train_data_value[:,0:1], dtype=torch.int32)
	train_onehot = torch.zeros((train_label.shape[0],10), dtype=torch.int32)
	for i in range(len(train_label)): train_onehot[i][train_label[i][0]] = 1
	train_input /= 255.0
	test_input /= 255.0
	train_dataset = TensorDataset(train_input, train_onehot.float())
	test_dataset = TensorDataset(test_input)
	train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_iter = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

	# build model
	loss = nn.CrossEntropyLoss(reduction='none')
	
	# net = nn.Sequential(nn.Linear(784, 10))
	net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
	
	net.apply(fn=init_weights)
	device = torch.device("cuda" if torch.cuda.is_available() else \
						"mps" if torch.backends.mps.is_available() else "cpu") 
	net.to(device)

	# train
	net.train()
	train(net=net, train_iter=train_iter, lr = learning_rate, loss=loss)
	# eval
	net.eval()
	ans(net=net, eval_iter=test_iter)	
