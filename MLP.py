import pandas as pd
import numpy as np

# 计算准确率
def accuracy(y_hat, y):
	"""计算预测正确的数量"""
def accuracy(y_hat, y):  
	"""计算预测正确的数量"""
	y_hat = y_hat[:,].argmax(axis=1)
	y = y[:,].argmax(axis=1)
	cmp = np.sum(y_hat[:] == y[:])
	return cmp / y.shape[0]

# 激活函数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 激活函数的导数
def grad_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def LeakyReLU(x):
	return np.where(x > 0, x, 0.00 * x)

def grad_LeakyReLU(x):
	return np.where(x > 0, 1, 0.00)

# softmax函数
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# softmax函数的导数
def grad_softmax(x):
	return (1 - softmax(x)) * softmax(x)

# 线性函数
def linear(X, w, b):
	return np.matmul(X, w) + b

# 激活函数
activation = sigmoid
grad_activation = grad_sigmoid
# activation = LeakyReLU
# grad_activation = grad_LeakyReLU

# MLP模型的参数
input_size = 784
hidden_size = 320
output_size = 10
number_hidden_layers = 3
numclass = 10

# 训练的超参数
batch_size = 200
num_epochs = 50
learning_rate = 0.02

# 规定参数（ndarray）的形状
# 可训练
w_input = np.zeros((input_size, hidden_size), dtype="float32")
w_hidden_arr = np.zeros([number_hidden_layers-1, hidden_size, hidden_size], dtype="float32")
w_output = np.zeros([hidden_size, output_size], dtype="float32")
b_input = np.zeros([hidden_size], dtype="float32")
b_hidden_arr = np.zeros([number_hidden_layers, hidden_size], dtype="float32")
b_output = np.zeros([output_size], dtype="float32")

# 参数的梯度
w_input_grad = np.zeros([batch_size, input_size, hidden_size], dtype="float32")
w_hidden_grad_arr = np.zeros([number_hidden_layers-1, batch_size, hidden_size, hidden_size], dtype="float32")
w_output_grad = np.zeros([batch_size, hidden_size, output_size], dtype="float32")
b_input_grad = np.zeros([batch_size, hidden_size], dtype="float32")
b_hidden_grad_arr = np.zeros([number_hidden_layers, batch_size, hidden_size], dtype="float32")
b_output_grad = np.zeros([batch_size, output_size], dtype="float32")

# 输入输出的形状、梯度
input_layer = np.zeros([batch_size, input_size], dtype="float32")
hidden_layers = np.zeros([number_hidden_layers, batch_size, hidden_size], dtype="float32") 
output_layer = np.zeros([batch_size, output_size], dtype="float32")
softmax_output = np.zeros([batch_size, output_size], dtype="float32")
input_layer_grad = np.zeros([input_size], dtype="float32")
hidden_layers_grad = np.zeros([number_hidden_layers, batch_size, hidden_size], dtype="float32")
output_layer_grad = np.zeros([output_size], dtype="float32")
train_label = np.zeros([batch_size, output_size], dtype="float32")

# 损失（一个batch的）
loss = np.zeros([batch_size], dtype="float32")

# 初始化参数

def init_weights() -> None:
	w_input[...] = np.random.normal(size=(input_size, hidden_size))
	w_hidden_arr[...] = np.random.normal(size=(number_hidden_layers-1, hidden_size, hidden_size))
	w_output[...] = np.random.normal(size=(hidden_size, output_size))
	return None

# 前向传播
def feed_forward() -> None:
	hidden_layers[0, ...] = linear(input_layer, w_input, b_input)
	for i in range(0, number_hidden_layers-1, 1):
		hidden_layers[i+1, ...] = activation(linear(hidden_layers[i], w_hidden_arr[i], b_hidden_arr[i]))
	output_layer[...] = linear(hidden_layers[number_hidden_layers-1], w_output, b_output)
	softmax_output[...] = softmax(output_layer)
	return None

# 计算损失（交叉熵）
def calculate_loss() -> None:
	loss[...] = np.array([-np.inner(np.log(softmax_output[i]), softmax_output[i]) for i in range(batch_size)])

# 反向传播
def back_propagation() -> None:

	# 输出层
	output_layer_grad = (softmax_output - train_label)
	w_output_grad = np.matmul(hidden_layers[number_hidden_layers-1].T, output_layer_grad)
	hidden_layers_grad[number_hidden_layers-1] = np.matmul(output_layer_grad, w_output.T)
	b_output_grad = output_layer_grad

	# 隐层
	for i in range(number_hidden_layers-1, 0, -1):
		w_hidden_grad_arr[i-1] = np.matmul(hidden_layers[i-1].T, hidden_layers_grad[i] * grad_activation(\
								linear(hidden_layers[i-1], w_hidden_arr[i-1], b_hidden_arr[i-1])))
		b_hidden_grad_arr[i-1] = hidden_layers_grad[i] * grad_activation(\
								linear(hidden_layers[i-1], w_hidden_arr[i-1], b_hidden_arr[i-1]))
		hidden_layers_grad[i-1] = np.matmul(hidden_layers_grad[i] * grad_activation(\
							linear(hidden_layers[i-1], w_hidden_arr[i-1], b_hidden_arr[i-1])), w_hidden_arr[i-1].T)

	# 输入层
	w_input_grad = np.matmul(input_layer.T, hidden_layers_grad[0] )
	b_input_grad = hidden_layers_grad[0]
	
	return None

# 更新参数
def update_weights() -> None:
	# 梯度下降
	w_input[...] -= learning_rate * w_input_grad.sum(axis=0) / batch_size
	b_input[...] -= learning_rate * b_input_grad.sum(axis=0) / batch_size
	for i in range(0, number_hidden_layers-1, 1):
		w_hidden_arr[i, ...] -= learning_rate * w_hidden_grad_arr[i].sum(axis=0) / batch_size
		b_hidden_arr[i, ...] -= learning_rate * b_hidden_grad_arr[i].sum(axis=0) / batch_size
	w_output[...] -= learning_rate * w_output_grad.sum(axis=0) / batch_size
	b_output[...] -= learning_rate * b_output_grad.sum(axis=0) / batch_size
	
	return None


if __name__ == "__main__":

	# 读取数据
	train_pd = pd.read_csv("train.csv")
	# 归一化特征
	train_feature = train_pd.values.copy() [:, 1:] / 255.0
	# one-hot编码
	train_labels = np.eye(numclass)[ train_pd.values.copy() [:, 0] ].reshape(\
									len(train_pd.values) // batch_size, batch_size, numclass)

	test_pd = pd.read_csv("test.csv")
	test_feature = test_pd.values.copy() / 255.0

	# 训练参数的初始化
	init_weights()

	for epoch in range(num_epochs):
		train_loss = 0
		train_acc = 0
		for i in range(0, train_feature.shape[0], batch_size):
			input_layer[...] = train_feature[i:i+batch_size]
			train_label[...] = train_labels[i // batch_size]
			feed_forward()
			calculate_loss()
			train_loss += loss.sum() / batch_size
			back_propagation()
			update_weights()
		print(f"Epoch {epoch}, Loss: {train_loss}, Accuracy: {accuracy(softmax_output, train_label)}")

	ans = []

	for epoch in range(1):
		test_acc = 0
		test_loss = 0
		for i in range(0, test_feature.shape[0], batch_size):
			input_layer[...] = test_feature[i:i+batch_size]
			feed_forward()
			calculate_loss()
			test_loss += loss.sum() / batch_size
			# back_propagation()
			# update_weights()
			ans.extend(softmax_output[:,].argmax(axis=1))
		print(f"Epoch {epoch}, Loss: {test_loss}")

	pd.DataFrame(np.array([[i+1 for i in range(len(ans))], ans]).T, \
				columns=["ImageId", "Label"]).to_csv("pred.csv", index=False, header=True )

	# 保存模型
	np.save("w_input.npy", w_input)
	np.save("w_hidden_arr.npy", w_hidden_arr)
	np.save("w_output.npy", w_output)
	np.save("b_input.npy", b_input)
	np.save("b_hidden_arr.npy", b_hidden_arr)
	np.save("b_output.npy", b_output)





