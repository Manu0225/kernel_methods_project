from abc import ABC, abstractmethod
import numpy as np


class Method(ABC):
	def __init__(self):
		self.kernel = None
		self.X = None
		self.alpha = None

	@abstractmethod
	def learn(self, X, Y):
		pass

	def predict(self, X_prime):
		rkhs_func = self.kernel.make_rkhs_func(self.alpha)
		f = rkhs_func(X_prime)

		h = np.sign(f)
		h[h == 0] = 1

		return h


def accuracy(y_true, y_pred):
	return np.sum(y_true == y_pred) / len(y_true)
