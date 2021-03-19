from math import floor
import numpy as np
import read_write
import kernels
import methods  # .methods as methods


# from multiprocessing import Pool


def generate_kernel_matrices(X_train, kernel_class, ls_params):
	ls_kernels = [kernel_class(*param) for param in ls_params]

	for kernel in ls_kernels:
		K, phi = kernel.fit(X_train)
		yield K, phi


def main_sum():
	ls_kernel = [
		kernels.SpectrumKernel(8, True),
		kernels.SpectrumKernel(6, True),
		kernels.SpectrumKernel(6, True)
	]
	ls_kernel_prime = [
		kernels.Gaussian(.1, True),
		kernels.Gaussian(.1, True),
		kernels.Gaussian(.1, True)
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
		length = X.shape[1]

		X_cat = np.concatenate((X, X), axis=-1)

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		# X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		# k,m,A

		kernel_class = kernels.SumKernel
		method_class = methods.KernelRidgeRegression
		params_kernel = [(ls_kernel[i], ls_kernel_prime[i], length, length, False)]
		# [(k, max(floor(k / 10), 1), 4) for k in range(6, 18, 3)]  # 10. **
		reg_vals = 10. ** np.arange(-2, 3, 1 / 4)
		validation(X_cat, y, kernel_class, method_class, params_kernel, reg_vals)


def main_poly():
	ls_kernel = [
		kernels.SpectrumKernel(15),
		kernels.SpectrumKernel(6),
		kernels.SpectrumKernel(6)
	]
	for i in range(3):
		print("##################",
			  f"i={i}")
		X = read_write.read(f"data/Xtr{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		kernel_class = kernels.PolyKernel
		method_class = methods.KernelLogisticRegression
		print(kernel_class, method_class)
		params_kernel = [(ls_kernel[i], deg, True) for deg in range(2, 6)]
		# [(k, max(floor(k / 10), 1), 4) for k in range(6, 18, 3)]  # 10. **
		reg_vals = 10. ** np.arange(-3, 4, 1)

		validation(X, y, kernel_class, method_class, params_kernel, reg_vals)


def main_val():
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		# X_test = read_write.read(f"data/Xte{i}.csv")

		y = read_write.read_labels(f"data/Ytr{i}.csv")

		# k,m,A
		kernel_class = kernels.Gaussian

		method_class = methods.KernelRidgeRegression
		# params_kernel = [(i, True) for i in range(4, 16, 2)]  # Spectrum
		# params_kernel = [(i, 0, 4, True) for i in range(4, 16, 2)]  # Mismatch #max(floor(i / 5)
		params_kernel = [(10. ** i, True) for i in range(-3, 3, 1)]  # gaussian
		reg_vals = 10. ** np.arange(-2, 3, 1)

		validation(X, y,
				   kernel_class, method_class, params_kernel, reg_vals)


# print(res)


def main_rendu():
	ls_kernel = [
		kernels.SpectrumKernel(8, True),
		kernels.SpectrumKernel(6, True),
		kernels.SpectrumKernel(6, True)
	]
	ls_kernel_prime = [
		kernels.Gaussian(.1, True),
		kernels.Gaussian(.1, True),
		kernels.Gaussian(.1, True)
	]
	# ls_kernel = [#manque des true
	# 	kernels.MismatchKernel(10, 1, 4),
	# 	kernels.MismatchKernel(9, 1, 4),
	# 	kernels.MismatchKernel(8, 1, 4)
	# ]
	ls_reg_val = [0.03162277660168379, 0.03162277660168379, 0.01]
	ls_methods = [
		methods.KernelRidgeRegression(
			kernels.SumKernel(ls_kernel[i], ls_kernel_prime[i], 101, 101, False),
			reg_val=ls_reg_val[i])
		for i in range(3)
		# methods.SVM(ls_kernel[1],
		# 			reg_val=ls_reg_val[1]),
		# methods.KernelRidgeRegression(ls_kernel[2],
		# 							  reg_val=ls_reg_val[2]),
	]
	for i in range(3):
		print("##################",
			  f"i={i}")
		# X = read_write.read_X100(f"data/Xtr{i}_mat100.csv")
		X = read_write.read(f"data/Xtr{i}.csv")
		X_cat = np.concatenate((X, X), axis=-1)

		# X_test = read_write.read_X100(f"data/Xte{i}_mat100.csv")
		X_test = read_write.read(f"data/Xte{i}.csv")
		X_test_cat = np.concatenate((X_test, X_test), axis=-1)

		y = read_write.read_labels(f"data/Ytr{i}.csv")
		# X_cat = np.concatenate((X, X), axis=-1)
		# X_test_cat = np.concatenate((X_test, X_test), axis=-1)
		ls_methods[i].learn(X_cat, y)
		y_pred = ls_methods[i].predict(X_cat)
		print(methods.accuracy(y, y_pred))
		y_test = ls_methods[i].predict(X_test_cat)

		read_write.write(y_test,
						 "predictions/Yte.csv",
						 offset=i * 1000,
						 append=(i != 0)
						 )


def main():
	main_rendu()
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
	return X_train, y_train, X_val, y_val, sigma


def extract_kernel(K, feats, sigma, n_train):
	K_train = K[sigma, :][:, sigma][:n_train, :n_train]
	K_val = K[sigma, :][:, sigma][n_train:, :n_train]
	if feats is None:
		feats_train, feats_val = None, None
	elif isinstance(feats, np.ndarray):
		feats_train = feats[sigma][:n_train]
		feats_val = feats[sigma][n_train:]
	else:
		# feats is assumed to be an iterable of ndarrays
		feats_train = [f[sigma][:n_train] if f is not None else None for f in feats]
		feats_val = [f[sigma][n_train:] if f is not None else None for f in feats]
	return K_train, feats_train, K_val, feats_val


# def eval(X_train, y_train, X_val, y_val, param, reg_val, kernel_class, method_class):
# 	kernel = kernel_class(*param)
# 	method = method_class(kernel, reg_val=reg_val)
# 	method.learn(X_train, y_train)
# 	y_pred = method.predict(X_val)
# 	acc = methods.accuracy(y_val, y_pred)
# 	print(acc, param, reg_val)
# 	return acc, param, reg_val


def validation(X, y, kernel_class, method_class, params_kernel, reg_vals, n_iters=5):
	# best_acc = 0
	# best_param = None
	# best_reg_val = None
	generator_kernel_matrices = \
		generate_kernel_matrices(X, kernel_class, ls_params=params_kernel)
	for param, (K, phi) in zip(params_kernel, generator_kernel_matrices):
		try:
			ls_acc = np.zeros((n_iters, len(reg_vals)))
			for it in range(n_iters):
				X_train, y_train, X_val, y_val, sigma = split(X, y, proportion_train=.8)
				n_train = len(X_train)
				n_val = len(X_val)
				K_train, feats_train, K_val, feats_val = extract_kernel(K, phi, sigma, n_train)
				assert K_train.shape == (n_train, n_train)
				assert K_val.shape == (n_val, n_train)

				for j, reg_val in enumerate(reg_vals):
					kernel = kernel_class(*param)
					method = method_class(kernel, reg_val=reg_val)
					method.learn(X_train, y_train, K=K_train, phi=feats_train)
					y_pred = method.predict(X_val, K_prime=K_val, phi_prime=feats_val)
					acc = methods.accuracy(y_val, y_pred)
					ls_acc[it, j] = acc
			avg_acc = np.mean(ls_acc, axis=0)
			std_acc = np.std(ls_acc, axis=0)
			for acc, reg_val, std in zip(avg_acc, reg_vals, std_acc):
				print(f"{acc:.4f}, {param}, {reg_val}, {std:.3f}")
		except Exception as e:
			raise (e)
			print("petit message pour nous")
			print(type(e))
			print(e)


# if acc > best_acc:
# 	best_acc = acc
# 	best_param = param
# 	best_reg_val = reg_val


# return best_acc, best_param, best_reg_val


if __name__ == '__main__':
	main()
