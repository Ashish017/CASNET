import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

files = os.listdir()
files = [file for file in files if file[0]=="M"]

di = {}

for file in files:
	frame = pd.read_csv(file)
	di[file[:-4]] = frame['success_rate']

frame = pd.DataFrame(di)
frame.to_csv("Manipulator_expert_policy.csv")