import tqdm as tqdm
import pandas as pd
import numpy as np  

# Utils
# 参考《动手学深度学习》
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


# 参数
input_size = 784
hidden_size = 128
output_size = 10
number_hidden_layers = 5

# Math functions
def init_weights(m):
	pass

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def grad_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def grad_softmax(x):
	return (1 - softmax(x)) * softmax(x)

def linear(X, w, b):
	return np.matmul(X, w) + b

w_input = np.zero([input_size, hidden_size], dtype='f')
w_hidden_arr = np.zero([number_hidden_layers-1, hidden_size, hidden_size], dtype='f')
w_output = np.zero([hidden_size, output_size], dtype='f')

b_input = np.zero([hidden_size], dtype='f')
b_hidden_arr = np.zero([number_hidden_layers, hidden_size], dtype='f')
b_output = np.zero([output_size], dtype='f')

w_input_grad = np.zero([784, 128], dtype='f')
w_hidden_grad_arr = np.zero([4, 128], dtype='f')
w_output_grad = np.zero([128, 10], dtype='f')

b_input_grad = np.zero([128], dtype='f')
b_hidden_grad_arr = np.zero([4, 128], dtype='f')
b_output_grad = np.zero([10], dtype='f')

input_layer = np.zero([784], dtype='f')
hidden_layer = np.array([5, 128], dtype='f') 
output_layer = np.zero([10], dtype='f')

def feedforward(X):
	# input layer
	hidden_layer[0] = sigmoid(linear(input_layer, w_input, b))
	for i in range(4):
		hidden_layer[i+1] = sigmoid(linear(hidden[i], w_hidden_arr[i], b_hidden_arr[i]))
	output_layer = softmax(linear(hidden_layer[4], w_output, b_output))
	return None

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
	net = nn.Sequential(nn.Linear(784,10))
	
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
