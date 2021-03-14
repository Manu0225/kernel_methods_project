from math import floor
import numpy as np
import read_write
import kernels
import methods  # .methods as methods
from multiprocessing import Pool


def main_val():
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		X_train, y_train, X_val, y_val = split(X, y, proportion_train=.8)
		kernel_class = kernels.Gaussian
		method_class = methods.KernelLogisticRegression
		params_kernel = [(10. ** i,) for i in range(1, 4, 1)]
		reg_vals = 10. ** np.arange(-2, 3, 1)
		res = validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals)
	# print(res)


def main():
	kernel_0 = kernels.SpectrumKernel(15)
	kernel_1 = kernels.SpectrumKernel(6)
	kernel_2 = kernels.SpectrumKernel(6)
	ls_methods = [
		methods.SVM(kernel_0, reg_val=.1),
		methods.SVM(kernel_1, reg_val=.1),
		methods.SVM(kernel_2, reg_val=.01),
	]
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		ls_methods[i].learn(X, y)
		y_pred = ls_methods[i].predict(X)
		print(methods.accuracy(y,y_pred))
		y_test = ls_methods[i].predict(X_test)

		read_write.write(y_test,
						 "predictions/Yte.csv",
						 offset=i*1000,
						 append=(i != 0)
						 )



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


# def eval(X_train, y_train, X_val, y_val, param, reg_val, kernel_class, method_class):
# 	kernel = kernel_class(*param)
# 	method = method_class(kernel, reg_val=reg_val)
# 	method.learn(X_train, y_train)
# 	y_pred = method.predict(X_val)
# 	acc = methods.accuracy(y_val, y_pred)
# 	print(acc, param, reg_val)
# 	return acc, param, reg_val


def validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals):
	best_acc = 0
	best_param = None
	best_reg_val = None

	for param in params_kernel:
		for reg_val in reg_vals:
			kernel = kernel_class(*param)
			method = method_class(kernel, reg_val=reg_val)
			method.learn(X_train, y_train)
			y_pred = method.predict(X_val)
			acc = methods.accuracy(y_val, y_pred)
			print(f"{acc}, {param[0]}, {reg_val}")
			if acc > best_acc:
				best_acc = acc
				best_param = param
				best_reg_val = reg_val
	return best_acc, best_param, best_reg_val


if __name__ == '__main__':
	main()
