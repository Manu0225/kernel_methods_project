from math import floor
import numpy as np
import read_write
import kernels
import methods  # .methods as methods
from multiprocessing import Pool


def generate_kernel_matrices(X_train, method_class, kernel_class, ls_params):
	ls_kernels = [kernel_class(*param) for param in ls_params]

	ls_K = []
	ls_phi = []

	for i in range(len(ls_params)):
		method = method_class(ls_kernels[i])
		K, phi = method.kernel.fit(X_train)
		ls_K.append(K)
		ls_phi.append(phi)

	ls_K = np.array(ls_K)
	ls_phi = np.array(ls_phi)
	return ls_K, ls_phi


def main_sum():
	ls_kernel = [
		kernels.SpectrumKernel(15),
		kernels.SpectrumKernel(6),
		kernels.SpectrumKernel(6)
	]
	ls_kernel_prime = [
		kernels.Gaussian(1),
		kernels.Gaussian(1),
		kernels.Gaussian(.01)
	]
	# ls_methods = [
	# methods.SVM(kernel_0, reg_val=.1),
	# methods.SVM(kernel_1, reg_val=.1),
	# methods.SVM(kernel_2, reg_val=.01),
	# ]
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")
		l = X.shape[1]

		X_cat = np.concatenate((X, X), axis=-1)

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		# X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		X_train, y_train, X_val, y_val = split(X_cat, y, proportion_train=.8)
		# k,m,A

		kernel_class = kernels.SumKernel
		method_class = methods.SVM
		params_kernel = [(ls_kernel[i], ls_kernel_prime[i], l, l)]
		# [(k, max(floor(k / 10), 1), 4) for k in range(6, 18, 3)]  # 10. **
		reg_vals = 10. ** np.arange(-2, 3, 1 / 2)
		res = validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals)


def main_poly():
	ls_kernel = [
		kernels.SpectrumKernel(15),
		kernels.SpectrumKernel(6),
		kernels.SpectrumKernel(6)
	]
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		# X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		X_train, y_train, X_val, y_val = split(X, y, proportion_train=.8)
		# k,m,A

		kernel_class = kernels.PolyKernel
		method_class = methods.KernelLogisticRegression
		print(kernel_class, method_class)
		params_kernel = [(ls_kernel[i], deg) for deg in range(2, 5)]
		# [(k, max(floor(k / 10), 1), 4) for k in range(6, 18, 3)]  # 10. **
		reg_vals = 10. ** np.arange(-3, 4, 1)

		res = validation(X_train, y_train, X_val, y_val, kernel_class, method_class, params_kernel, reg_vals)


def main_val():
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		# X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		X_train, y_train, X_val, y_val = split(X, y, proportion_train=.8)
		# k,m,A
		kernel_class = kernels.MismatchKernel
		method_class = methods.SVM
		params_kernel = [(i, max(floor(i / 10), 1), 4) for i in range(3, 16, 2)]  # Mismatch
		# params_kernel = [(10.**i,) for i in range(-2, 2, 1)] # gaussian
		reg_vals = 10. ** np.arange(-2, 3, 1.5)

		ls_K, ls_phi = generate_kernel_matrices(X_train, method_class, kernel_class, params_kernel)

		res = validation(X_train, y_train, X_val, y_val,
						 kernel_class, ls_K, ls_phi, method_class, params_kernel, reg_vals)


# print(res)


def main_rendu():
	ls_kernel = [
		kernels.MismatchKernel(12, 1, 4),
		kernels.MismatchKernel(12, 1, 4),
		kernels.MismatchKernel(9, 1, 4)
	]
	ls_kernel_prime = [
		kernels.Gaussian(1),
		kernels.Gaussian(1),
		kernels.Gaussian(.01)
	]
	ls_reg_val = [1, 10, 1]
	ls_methods = [
		methods.SVM(ls_kernel[i],
					reg_val=ls_reg_val[i])
		for i in range(3)
	]
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")
		# X_cat = np.concatenate((X, X), axis=-1)
		# X_test_cat = np.concatenate((X_test, X_test), axis=-1)
		ls_methods[i].learn(X, y, normalize=False)
		y_pred = ls_methods[i].predict(X)
		print(methods.accuracy(y, y_pred))
		y_test = ls_methods[i].predict(X_test)

		read_write.write(y_test,
						 "predictions/Yte.csv",
						 offset=i * 1000,
						 append=(i != 0)
						 )


def main():
	main_val()
	return


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


def validation(X_train, y_train, X_val, y_val, kernel_class, ls_K, ls_phi,
			   method_class, params_kernel, reg_vals):
	best_acc = 0
	best_param = None
	best_reg_val = None

	for i, param in enumerate(params_kernel):
		for j, reg_val in enumerate(reg_vals):
			kernel = kernel_class(*param)
			method = method_class(kernel, reg_val=reg_val)
			method.learn(X_train, y_train, K=ls_K[i], phi=ls_phi[i], normalize=True)
			y_pred = method.predict(X_val)
			acc = methods.accuracy(y_val, y_pred)
			print(f"{acc}, {param}, {reg_val}")
			if acc > best_acc:
				best_acc = acc
				best_param = param
				best_reg_val = reg_val
	return best_acc, best_param, best_reg_val


if __name__ == '__main__':
	main()
