from parameters import hyperparameters as p
from parameters import settings
import datetime, os, time
import pandas as pd
import time

class Logger:
	def __init__(self):
		os.system("clear")
		try:
			os.mkdir("Generated_data")
		except FileExistsError:
			pass
		self.di = {}
		self.t = time.time()
		self.t_start = eta = datetime.datetime.now()

		for env_name in settings.env_names:
			self.di[env_name] = []

	def add_rewards(self, names, rewards, steps):
		for i in range(names):
			self.di[settings.env_names[i]].append(rewards[i].item())

	def save(self):
		if settings.mode == 'test':
			frame = pd.DataFrame(self.di)
			frame.to_csv("Generated_data/{}_{}.csv".format(settings.mode, settings.seed))

	def show_update(self, step):
		os.system("clear")
		terminal_width = os.get_terminal_size().columns
		bar_width = terminal_width - 20
		percent = (step/p.max_numsteps)*100
		iteration_speed = round(10/(time.time() - self.t),1)
		eta = (100-percent) * ((datetime.datetime.now()-self.t_start)/(percent+1e-6))
		percent = round(percent, 1)
		print(" Mode: " + settings.mode + " \tSeed: {}".format(settings.seed) + " \tSpeed: {}".format(iteration_speed) + " Steps/second" + " \tETA: {}".format(str(eta).split(".")[0]), end="\n\n")
		done_bar = 	"\033[94m{}\033[00m".format(chr(9606)*int(bar_width*percent/100))	
		remaining = "\033[97m{}\033[00m".format(chr(9606)*int(bar_width*(100-percent)/100))
		print(" "+done_bar+remaining, end=" ")
		print("{} %".format(percent), end="\n\n")
		self.t = time.time()