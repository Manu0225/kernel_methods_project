from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import scipy.sparse


# X and X_prime are (n, d) and (m, d) numpy vectors resp.
# Y is a (n,1) numpy vectors


class Kernel(ABC):
	@abstractmethod
	def fit(self, X):
		pass

	@abstractmethod
	def make_rkhs_func(self, alpha):
		pass


class NaiveKernel(ABC):
	@abstractmethod
	def __init__(self):
		self.X = None

	def K(self, X, X_prime):
		pass

	def fit(self, X):
		self.X = X
		return self.K(X, X)

	def make_rkhs_func(self, alpha):
		return lambda Xprime: self.K(Xprime, self.X) @ alpha


class Linear(NaiveKernel):
	def __init__(self):
		super(Linear, self).__init__()

	def K(self, X, X_prime):
		return X @ X_prime.T


class Gaussian(NaiveKernel):
	def __init__(self, alpha=0.1):
		super(Gaussian, self).__init__()
		self.alpha = alpha

	def K(self, X, X_prime):
		C = cdist(X, X_prime, 'euclidean') ** 2
		return np.exp(- self.alpha / 2 * C)


class FeaturesKernel(Kernel):
	def __init__(self):
		self.feats = None

	@abstractmethod
	def features(self, X):
		pass

	def fit(self, X):
		self.feats = self.features(X)
		return (self.feats.dot(self.feats.T)).todense().A

	def make_rkhs_func(self, alpha):
		w = self.feats.T.dot(alpha)
		return lambda Xprime: self.features(Xprime).dot(w)


class SpectrumKernel(FeaturesKernel):
	def __init__(self, k):
		super(SpectrumKernel, self).__init__()
		self.k = k

	def from_decomposition(self, decomp, basis):
		n, _ = decomp.shape
		assert decomp.shape == (n, self.k)

		return decomp.dot(basis ** np.arange(self.k))

	def features(self, X):
		a = np.max(X) + 1
		n, length = X.shape

		phi = scipy.sparse.lil_matrix((n, a ** self.k), dtype=np.uint)
		for i in range(self.k - 1, length):
			to_mod_indices = self.from_decomposition(X[:, i - (self.k - 1):i + 1], a)
			for j, index in enumerate(to_mod_indices):
				phi[j, index] += 1
		return phi.tocsr()


class Polynomial(NaiveKernel):
	def __init__(self, degree):
		super(Polynomial, self).__init__()
		assert (isinstance(degree, int))
		self.degree = degree

	def K(self, X, X_prime):
		return (X @ X_prime.T) ** self.degree
