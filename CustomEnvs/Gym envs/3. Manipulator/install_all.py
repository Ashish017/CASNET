import os

folders = os.listdir()

for folder in folders:
	if not folder.startswith("i"):
		os.system("pip3 install -e " + folder)