from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist

# X and X_prime are (n, d) and (m, d) numpy vectors resp.
# Y is a (n,1) numpy vectors


class Kernel(ABC):
	@abstractmethod
	def K(self, X, X_prime):
		pass

	def fit(self, X):
		self.X = X
		return self.K(X, X)

	def make_rkhs_func(self, alpha):
		return lambda Xprime : self.K(Xprime, self.X) @ alpha

class Linear(Kernel):
	def __init__(self):
		super(Linear, self).__init__()

	def K(self, X, X_prime):
		return X @ X_prime.T

class Gaussian(Kernel):
	def __init__(self, alpha=0.1):
		super(Gaussian, self).__init__()
		self.alpha = alpha

	def K(self, X, X_prime):
		C = cdist(X, X_prime, 'euclidean') ** 2
		return np.exp(- self.alpha / 2 * C)
