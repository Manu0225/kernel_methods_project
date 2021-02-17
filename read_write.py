import numpy as np


def load_X100(path):
	# path resembling: "data/Xte0_mat100.csv"
	return np.loadtxt(path)


def load_labels(path):
	# need to ignore the first col
	return 2*np.loadtxt(path, delimiter=",", skiprows=1)[:, 1]-1


def write(Y, path, offset):
	"""
	:param Y: of shape (n,1), with values ±1
	:param path: relative path of the file
	:param offset:
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
		[(1+Y.squeeze())/2]
	]).astype(np.int32).T

	np.savetxt(path, write_array, fmt="%.0f", delimiter=",", header="Id,Bound", comments="")

# write(np.random.choice([-1,1],5).reshape(5,1), "data/test2.csv",0)