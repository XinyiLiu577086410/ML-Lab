import numpy as np
import pandas as pd

ans = pd.read_csv("std-ans.csv")
pred = pd.read_csv("pred.csv")

print("Accuracy: ", np.sum(ans["Label"].values == pred["Label"].values) / len(ans))