from methods import Method
import cvxpy as cp
import numpy as np


class SVM(Method):
	def __init__(self, kernel, reg_val=0.1):
		super(SVM, self).__init__()
		self.kernel = kernel
		self.reg_val = reg_val
		self.alpha = None

	def _kernel_learn(self, K, Y, correction=1e-7):
		K += correction * np.eye(K.shape[0])
		n, _ = K.shape
		alpha = cp.Variable((n, 1))
		problem = cp.Problem(cp.Minimize(- 2 * alpha.T @ Y + cp.quad_form(alpha, K)),
							 [
								 2 * self.reg_val * n * cp.multiply(Y, alpha) <= 1,
								 cp.multiply(Y, alpha) >= 0
							 ]
							 )
		problem.solve()
		return alpha.value
