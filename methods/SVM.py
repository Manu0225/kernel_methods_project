from methods import Method
import cvxpy as cp


class SVM(Method):
	def __init__(self, kernel, reg_val=0.1):
		self.kernel = kernel
		self.reg_val = reg_val
		self.alpha = None

	def learn(self, X, Y):
		K = self.kernel.fit(X)
		n, _ = K.shape
		alpha = cp.Variable((n, 1))
		problem = cp.Problem(cp.Minimize(- 2 * alpha.T @ Y + cp.quad_form(alpha, K)),
							 [
								 2 * self.reg_val * n * cp.multiply(Y, alpha) <= 1,
							  cp.multiply(Y, alpha) >= 0
							 ]
							 )
		problem.solve()
		self.alpha = alpha.value
