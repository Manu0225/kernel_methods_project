from math import floor
import numpy as np
import read_write
import kernels
import methods  # .methods as methods
from multiprocessing import Pool


def main():
	for i in range(3):
		X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		y = read_write.read_labels(f"data/Ytr{i}.csv")
		X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		X_train, y_train, X_val, y_val = split(X, y, proportion_train=.8)
		kernel_class = kernels.Polynomial
		method_class = methods.SVM
		params_kernel = [(i,) for i in range(1, 2, 3)]
		reg_vals = 10. ** np.arange(-2, 3, 1)
		res = validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals)
		print(res)


# linear_kernel = kernels.Linear()
# # gaussian_kernel = kernels.Gaussian()
# svm = methods.SVM(linear_kernel, reg_val=1e-1)
# # = methods.KernelLogisticRegression(linear_kernel, reg_val=1e-3)
# svm.learn(X_train, y_train)
# y_pred = svm.predict(X_train)
# print(methods.accuracy(y_train, y_pred))
# y_test = svm.predict(X_test)
# # print(y_test)
# read_write.write(y_test,
# 				 "predictions/Yte.csv",
# 				 offset=i * 1000,
# 				 append=(i != 0))


def split(X, y, proportion_train):
	N = len(X)
	assert (len(y) == N)
	sigma = np.random.permutation(N)
	n_train = floor(N * proportion_train)
	X_train, y_train = X[sigma][:n_train], y[sigma][:n_train]
	X_val, y_val = X[sigma][n_train:], y[sigma][n_train:]
	return X_train, y_train, X_val, y_val


def eval(X_train, y_train, X_val, y_val, param, reg_val, kernel_class, method_class):
	kernel = kernel_class(*param)
	method = method_class(kernel, reg_val=reg_val)
	method.learn(X_train, y_train)
	y_pred = method.predict(X_val)
	acc = methods.accuracy(y_val, y_pred)
	print(acc, param, reg_val)
	return acc, param, reg_val




def validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals):
	best_acc = 0
	best_param = None
	best_reg_val = None
	# ls_args = [[X_train, y_train, X_val, y_val, param, reg_val, kernel_class, method_class]
	# 		   for param in params_kernel
	# 		   for reg_val in reg_vals]
	# with Pool(12) as p:
	# 	ls_ret = p.starmap(eval, ls_args)
	# 	# np.argmax(np.array(ls_ret)[:, 0])
	# ls_ret = sorted(ls_ret, key=lambda x: x[0])
	# return ls_ret[-1]
	for param in params_kernel:
		for reg_val in reg_vals:
			kernel = kernel_class(*param)
			method = method_class(kernel, reg_val=reg_val)
			method.learn(X_train, y_train)
			y_pred = method.predict(X_val)
			acc = methods.accuracy(y_val, y_pred)
			print(acc, param, reg_val)
			if acc > best_acc:
				best_acc = acc
				best_param = param
				best_reg_val = reg_val
	return best_acc, best_param, best_reg_val


if __name__ == '__main__':
	main()
