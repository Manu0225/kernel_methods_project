import read_write
import kernels
import methods


def main():
	X_train = read_write.load_X100("data/Xtr0_mat100.csv")
	y_train = read_write.load_labels("data/Ytr0.csv")
	X_test = read_write.load_X100("data/Xte0_mat100.csv")
	linear_kernel = kernels.Linear()
	klr = methods.KernelLogisticRegression(linear_kernel, reg_val=1e-1)
	klr.learn(X_train, y_train)
	y_pred = klr.predict(X_train)
	print(methods.accuracy(y_train, y_pred))
	y_test = klr.predict(X_test)
	# print(y_test)
	read_write.write(y_test, "predictions/Yte0.csv", 0)


if __name__ == '__main__':
	main()
