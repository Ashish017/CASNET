import pandas as pd
import matplotlib.pyplot as plt
from parameters import hyperparameters as p

for variance in p.variance_limits:
	for j in range(1,4):
		f = pd.read_csv("Generated_data/retraining_data_{}_{}.csv".format(variance, j))

		cols = [col for col in f.columns if col[0]=="M"]

		data = []
		for i in range(len(f[cols[0]])):
			data_point = 0.0
			for col in cols:
				data_point += f[col][i]
			data.append(data_point/len(cols))

		x = [i for i in range(len(f[cols[0]]))]

		plt.plot(x, data)

		f = pd.read_csv("Generated_data/retraining_test_data_{}_{}.csv".format(variance, j))

		cols = [col for col in f.columns if col[0]=="M"]

		data = []
		for i in range(len(f[cols[0]])):
			data_point = 0.0
			for col in cols:
				data_point += f[col][i]
			data.append(data_point/len(cols))

		x = [i for i in range(len(f[cols[0]]))]

		plt.plot(x, data, color="red")
	plt.ylim(0.0, 1.0)
	plt.show()