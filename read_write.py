import numpy as np
from pandas import read_csv


def read_X100(path):
	# path resembling: "data/Xte0_mat100.csv"
	return np.loadtxt(path)


def read_labels(path):
	# need to ignore the first col
	return 2 * np.loadtxt(path, delimiter=",", skiprows=1)[:, 1, None] - 1


def read(path):
	with open(path) as csvfile:
		df = read_csv(csvfile)
		for i, char in enumerate(["A", "C", "G", "T"]):
			df['seq'] = df['seq'].str.replace(char, str(i))
		df = df['seq'].str.split('', expand=True)
		array = df.to_numpy()[:, 1:-1].astype(np.int)
	# print(array)
	# print(array.shape)
	return array


def write(Y, path, offset, append=True):
	"""
	:param Y: of shape (n,1), with values ±1
	:param path: relative path of the file
	:param offset:
	:param append: boolean to add to the end of the file or rewrite the all file
	:return:
	prints into the file "path" in the following format

	# Id, Bound
	# 0, 0
	# 1, 0
	# 2, 0
	# ....
	# 2998, 0
	# 2999, 0
	"""
	n, _ = Y.shape
	write_array = np.block([
		[np.arange(n) + offset],
		[(1 + Y.squeeze()) / 2]
	]).astype(np.int32).T

	with open(path, "a" if append else "w") as f:
		np.savetxt(f,
				   write_array,
				   fmt="%.0f",
				   delimiter=",",
				   header="Id,Bound" if not append else "",
				   comments="")

# write(np.random.choice([-1,1],5).reshape(5,1), "data/test2.csv",0)
