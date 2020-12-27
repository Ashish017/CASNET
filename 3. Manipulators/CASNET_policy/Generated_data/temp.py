import pandas as pd

seeds = [1,2,3]
variances = [0.3,0.4,0.5,0.6]

all_data = {}

for var in variances:
	for seed in seeds:
		file = "retraining_test_data_{}_{}.csv".format(var, seed)
		file = pd.read_csv(file)
		cols = file.columns
		cols = [col for col in cols if col[0]=="M"]

		for col in cols:
			all_data[col + "_{}".format(seed)] = file[col]

frame = pd.DataFrame(all_data)
frame.to_csv("Casnet_retraining_data.csv")