## 生成训练集和测试集
import pandas
import random
import numpy as np
train_orig = pandas.read_csv("train_orig.csv")
random.seed(0)
random.shuffle(train_orig.values)
train, test = train_orig.values[:37800], train_orig.values[37800:42000]
pandas.DataFrame(train, columns=train_orig.columns).to_csv("train.csv", index=False)
pandas.DataFrame(np.array([[i+1 for i in range(len(test))], test[:, 0]]).T, columns=["ImageId", "Label"]).to_csv("std-ans.csv", index=0)
pandas.DataFrame(test[:, 1:], columns=train_orig.columns[1:]).to_csv("test.csv", index=False)