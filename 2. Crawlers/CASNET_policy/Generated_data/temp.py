import pandas as pd
import matplotlib.pyplot as plt

t1 = pd.read_csv("test_1.csv")
t2 = pd.read_csv("test_2.csv")
t3 = pd.read_csv("test_3.csv")

files = [t1,t2,t3]

di = {}

for i, file in enumerate(files):
	for col in file.columns:
		if col[0] != "U":
			name = col + "_seed_{}".format(i+1)
			di[name] = file[col]

frame = pd.DataFrame(di)
frame.to_csv("Crawler_CASNET_test.csv")