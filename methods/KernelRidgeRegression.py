from methods import Method
import numpy as np


class KernelRidgeRegression(Method):
	def __init__(self, kernel, reg_val=0.1):
		super(KernelRidgeRegression, self).__init__()
		self.X = None
		self.kernel = kernel
		self.reg_val = reg_val
		self.alpha = None

	def learn(self, X, Y):
		self.X = X
		K = self.kernel.fit(X)
		n, _ = K.shape
		assert Y.shape == (n, 1)

		self.alpha = np.linalg.solve(K + self.reg_val * n * np.identity(n), Y)
		assert self.alpha.shape == (n, 1)
