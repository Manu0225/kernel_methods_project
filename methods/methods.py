from abc import ABC, abstractmethod
import numpy as np


class Method(ABC):
	def __init__(self):
		self.kernel = None
		self.alpha = None

	@abstractmethod
	def _kernel_learn(self, K, Y):
		pass

	def learn(self, X, Y, K=None, phi=None, normalize=False):
		K, _ = self.kernel.fit(X, K=K, phi=phi)
		norm_matrix = np.diag(K).reshape(-1, 1) @ np.diag(K).reshape(1,-1) if normalize else 1  # np.amax(np.abs(K))
		K = K / norm_matrix
		self.alpha = self._kernel_learn(K, Y)
		# self.alpha /= norm_matrix
		# QUE METTRE ICI ? np.diag(K) ?

	# return K, phi, self.alpha

	def predict(self, X_prime):
		rkhs_func = self.kernel.make_rkhs_func(self.alpha)
		f = rkhs_func(X_prime)

		h = np.sign(f)
		h[h == 0] = 1

		return h


def accuracy(y_true, y_pred):
	return np.sum(y_true == y_pred) / len(y_true)
