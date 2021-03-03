from methods import Method
import numpy as np


def line_search(f, f_x, nt_decrement, x, delta_x, alpha, beta):
	t = 1
	scalar_product = alpha * (-nt_decrement)
	assert (scalar_product < 0)
	in_dom, val = f(x + t * delta_x)
	while (not in_dom) or val > f_x + t * scalar_product:
		t = beta * t
		in_dom, val = f(x + t * delta_x)
	return t


def newton_step(grad_f_x, hessian_f_x):
	nt_step = -np.linalg.lstsq(hessian_f_x, grad_f_x, rcond=None)[0]
	nt_decrement = - np.dot(grad_f_x.T, nt_step)
	return nt_step, nt_decrement


def newton_method(f, f_deriv, x_0, eps=1e-7, alpha=0.1, beta=0.5):
	x = x_0
	n = 0
	while True:
		f_x, grad_f_x, hessian_f_x = f_deriv(x)
		nt_step, nt_decrement = newton_step(grad_f_x, hessian_f_x)
		if nt_decrement / 2 <= eps:
			break
		t = line_search(f, f_x, nt_decrement, x, nt_step, alpha, beta)
		x = x + t * nt_step
		n += 1
	return x


def logistic(u):
	""" logistic function """
	exp_u_pos_part = np.exp(-np.clip(u, 0, None))
	exp_u_neg_part = np.exp(np.clip(u, None, 0))
	return - np.clip(u, None, 0) + np.log1p(exp_u_neg_part + exp_u_pos_part - 1)


def sigmoid(u):
	""" sigmoid function """
	exp_u_pos_part = np.exp(-np.clip(u, 0, None))
	exp_u_neg_part = np.exp(np.clip(u, None, 0))
	return exp_u_neg_part / (exp_u_neg_part + exp_u_pos_part)


def make_kernel_logistic_regression_funcs(K, Y, reg_val):
	n, _ = Y.shape
	assert Y.shape == (n, 1)
	assert K.shape == (n, n)

	def loss(alpha):
		assert alpha.shape == (n, 1)
		K_alpha = K @ alpha
		return True, np.sum(logistic(Y * K_alpha)) / n + (reg_val / 2) * np.dot(alpha.T, K_alpha)

	def oracle(alpha):
		assert alpha.shape == (n, 1)
		K_alpha = K @ alpha
		value = np.sum(logistic(Y * K_alpha)) / n + (reg_val / 2) * np.dot(alpha.T, K_alpha)

		P = -sigmoid(- Y * K_alpha)
		grad = (1 / n) * K @ (P * Y) + reg_val * K_alpha

		W = sigmoid(Y * K_alpha) * sigmoid(- Y * K_alpha)
		hessian = (1 / n) * K @ (W * K) + reg_val * K
		return value, grad, hessian

	return loss, oracle


class KernelLogisticRegression(Method):
	def __init__(self, kernel, reg_val=0.1):
		self.kernel = kernel
		self.reg_val = reg_val

	def learn(self, X, Y, tol=1e-6):
		self.X = X
		K = self.kernel.fit(X)
		n, _ = K.shape
		assert Y.shape == (n, 1)

		loss, oracle = make_kernel_logistic_regression_funcs(K, Y, self.reg_val)
		alpha_0 = np.ones((n, 1))
		alpha = newton_method(loss, oracle, alpha_0, eps=tol)
		self.alpha = alpha
