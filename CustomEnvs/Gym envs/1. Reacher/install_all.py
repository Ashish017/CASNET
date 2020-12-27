import os

folders = os.listdir()

for folder in folders:
	if folder.startswith("R"):
		os.system("pip3 install -e " + folder)