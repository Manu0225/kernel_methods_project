from abc import ABC, abstractmethod
import numpy as np


class Method(ABC):
	def __init__(self):
		self.kernel = None
		self.alpha = None

	@abstractmethod
	def _kernel_learn(self, K, Y):
		pass

	def learn(self, X, Y, normalize=False):
		K = self.kernel.fit(X)
		norm = np.amax(np.abs(K)) if normalize else 1
		K = K / norm
		self.alpha = self._kernel_learn(K, Y)
		self.alpha /= norm

	def predict(self, X_prime):
		rkhs_func = self.kernel.make_rkhs_func(self.alpha)
		f = rkhs_func(X_prime)

		h = np.sign(f)
		h[h == 0] = 1

		return h


def accuracy(y_true, y_pred):
	return np.sum(y_true == y_pred) / len(y_true)
