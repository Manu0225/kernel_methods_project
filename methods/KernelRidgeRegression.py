from methods import Method
import numpy as np


class KernelRidgeRegression(Method):
	def __init__(self, kernel, reg_val=0.1):
		super(KernelRidgeRegression, self).__init__()
		self.kernel = kernel
		self.reg_val = reg_val

	def _kernel_learn(self, K, Y):
		n, _ = K.shape
		assert Y.shape == (n, 1)

		alpha = np.linalg.solve(K + self.reg_val * n * np.identity(n), Y)
		assert alpha.shape == (n, 1)
		return alpha
