from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.special
from tqdm import tqdm


class Kernel(ABC):
	"""
	generic class for kernels.

	each class of kernels will be a subclass of this one, either directly or inderectly through FeatureKernel
	"""

	def __init__(self, normalize=False):
		self.normalize = normalize
		self.X = None

	@abstractmethod
	def _K(self, X, X_prime):
		"""
		Abstract method to be implemented by kernel subclasses,
		Return the matrix k(X[i], X_prime[j])
		"""
		pass

	def K(self, X, X_prime):
		"""
		Return the matrix k(X[i], X_prime[j]) potentially with normalization
		"""
		K = self._K(X, X_prime)
		if self.normalize:
			diag_K = np.diag(self._K(X, X))
			diag_K_prime = np.diag(self._K(X_prime, X_prime))
			K = diag_K[:, None] * (K * diag_K_prime[None, :])
		return K

	def fit(self, X, K=None, phi=None):
		"""
		Return the matrix K(X, X) and can be used by subclasses to store info
		arguments K and phi are used to pass already computed kernel matrix and features
		"""
		self.X = X
		K = self.K(X, X) if K is None else K
		return K, None

	def apply_rkhs_func(self, alpha, X_prime, K_prime=None, phi_prime=None):
		"""
		Given the points x_1, ..., x_n previously used in fit, apply the function sum_i alpha_i K_{x_i}
		to Xprime
		"""
		return (self.K(X_prime, self.X) @ alpha if K_prime is None
				else K_prime @ alpha)


class Linear(Kernel):
	def __init__(self, *args):
		super(Linear, self).__init__(*args)

	def _K(self, X, X_prime):
		return X @ X_prime.T


class Gaussian(Kernel):
	def __init__(self, alpha=0.1, *args):
		super(Gaussian, self).__init__(*args)
		self.alpha = alpha

	def _K(self, X, X_prime):
		C = cdist(X, X_prime, 'euclidean') ** 2
		return np.exp(- self.alpha / 2 * C)


class Polynomial(Kernel):
	def __init__(self, degree, *args):
		super(Polynomial, self).__init__(*args)
		assert (isinstance(degree, int))
		self.degree = degree

	def _K(self, X, X_prime):
		return (X @ X_prime.T) ** self.degree


class FeaturesKernel(Kernel):
	"""
	Main abstract class for kernels implemented using features
	"""

	def __init__(self, *args):
		super(FeaturesKernel, self).__init__(*args)
		self.feats = None

	@abstractmethod
	def _features(self, X):
		"""
		Abstract method to be implemented by feature kernel subclasses
		returns the features for X
		"""
		pass

	def _K(self, X, X_prime):
		feats = self.features(X)
		feats_prime = self.features(X_prime)
		return (feats.dot(feats_prime.T)).todense().A

	def features(self, X):
		"""
		returns the features for X, potentially with normalization
		"""
		feats = self._features(X)
		if self.normalize:
			norm = scipy.sparse.linalg.norm(feats, axis=-1)
			assert norm.shape == (feats.shape[0],)
			diag = scipy.sparse.diags(norm ** (-1 / 2), format="csr")
			feats = diag @ feats
		assert scipy.sparse.issparse(feats)
		return feats

	def fit(self, X, K=None, phi=None):
		self.feats = self.features(X) if phi is None else phi
		K = (self.feats.dot(self.feats.T)).todense().A if K is None else K
		return K, self.feats

	def apply_rkhs_func(self, alpha, Xprime, K_prime=None, phi_prime=None):
		w = self.feats.T.dot(alpha)
		return (self.features(Xprime).dot(w) if phi_prime is None else
				phi_prime.dot(w))


class SpectrumKernel(FeaturesKernel):
	def __init__(self, k, *args):
		super(SpectrumKernel, self).__init__(*args)
		self.k = k

	def from_decomposition(self, decomp, basis):
		n, _ = decomp.shape
		assert decomp.shape == (n, self.k)

		return decomp.dot(basis ** np.arange(self.k))

	def _features(self, X):
		a = np.max(X) + 1
		n, length = X.shape

		phi = scipy.sparse.lil_matrix((n, a ** self.k), dtype=np.uint)
		for i in range(self.k - 1, length):
			to_mod_indices = self.from_decomposition(X[:, i - (self.k - 1):i + 1], a)
			for j, index in enumerate(to_mod_indices):
				phi[j, index] += 1
		return phi.tocsr()


class MismatchKernel(FeaturesKernel):
	def __init__(self, k, m, A, *args):
		super(FeaturesKernel, self).__init__(*args)
		self.k = k
		self.m = m
		assert k > m

		self.A = A
		self.max_n_matches_per_sample = sum([scipy.special.comb(k, i, exact=True) * (A - 1) ** i for i in range(m + 1)])

	def generate_matches(self, x, buff):
		"""
		given x a sample of size k, return naively all the sequences of size k
		which match with x up to m mismatches
		"""
		stack = [([], 0)]
		buff_index = 0
		while len(stack) > 0:
			length, curr_mismatches = stack.pop()
			if len(length) == self.k:
				buff[buff_index] = np.array(length)
				buff_index += 1
			elif curr_mismatches == self.m:
				buff[buff_index] = np.concatenate((np.array(length), x[len(length):]))
				buff_index += 1
			else:
				for a in range(self.A):
					stack.append((length + [a], curr_mismatches + int(a != x[len(length)])))
		assert buff_index == self.max_n_matches_per_sample

	def from_decomposition(self, decomp):

		assert decomp.shape == (self.max_n_matches_per_sample, self.k)

		return decomp.dot(self.A ** np.arange(self.k))

	def _features(self, X):
		assert (X <= self.A - 1).all()
		n, length = X.shape

		phi = scipy.sparse.lil_matrix((n, self.A ** self.k), dtype=np.uint)
		buff = np.empty((self.max_n_matches_per_sample, self.k), dtype=np.int)
		for j, row in enumerate(tqdm(X)):
			for i in range(self.k - 1, length):
				x = row[i - (self.k - 1):i + 1]
				self.generate_matches(x, buff)
				to_mod_indices = self.from_decomposition(buff)
				for index in to_mod_indices:
					phi[j, index] += 1
		return phi.tocsr()


class SumKernel(Kernel):
	def __init__(self, kernel_1, kernel_2, d1, d2, *args):
		super(SumKernel, self).__init__(*args)
		self.kernel_1 = kernel_1
		self.kernel_2 = kernel_2
		self.d1 = d1
		self.d2 = d2

	def _K(self, X, X_prime):
		assert X.shape[1] == self.d1 + self.d2
		assert X_prime.shape[1] == self.d1 + self.d2
		X_1, X_2 = X[:, :self.d1], X[:, self.d1:]
		X_1_prime, X_2_prime = X_prime[:, :self.d1], X_prime[:, self.d1:]
		return self.kernel_1.K(X_1, X_1_prime) + self.kernel_2.K(X_2, X_2_prime)

	def fit(self, X, K=None, phi=None):
		assert X.shape[1] == self.d1 + self.d2
		if K is None:
			assert phi is None
			X_1, X_2 = X[:, :self.d1], X[:, self.d1:]
			K1, phi_1 = self.kernel_1.fit(X_1)
			K2, phi_2 = self.kernel_2.fit(X_2)
			K = K1 + K2
			phi = (phi_1, phi_2)
		else:
			assert phi is not None
			phi_1, phi_2 = phi
			self.kernel_1.feats = phi_1
			self.kernel_2.feats = phi_2
		return K, phi

	def apply_rkhs_func(self, alpha, X_prime, K_prime=None, phi_prime=None):
		phi_1_prime, phi_2_prime = phi_prime if phi_prime is not None else (None, None)
		X_prime_1, X_prime_2 = X_prime[:, :self.d1], X_prime[:, self.d1:]
		res1 = self.kernel_1.apply_rkhs_func(alpha, X_prime_1, K_prime=K_prime, phi_prime=phi_1_prime)
		res2 = self.kernel_2.apply_rkhs_func(alpha, X_prime_2, K_prime=K_prime, phi_prime=phi_2_prime)

		return res1 + res2


class FeaturesPolyKernel(FeaturesKernel):
	def __init__(self, feature_kernel, degree, *args):
		super(FeaturesPolyKernel, self).__init__(*args)
		self.feature_kernel = feature_kernel
		self.degree = degree

	@staticmethod
	def features_tensor_product(phi, A):
		n, d = phi.shape
		_, p = A.shape
		assert A.shape == (n, p)

		res = scipy.sparse.vstack(
			[scipy.sparse.kron(A.getrow(i), phi.getrow(i), format='csr')
			 for i in range(n)],
			format='csr')
		assert res.shape == (n, d * p)

		return res

	def _features(self, X, phi=None):
		phi = self.feature_kernel.features(X) if phi is None else phi

		res = phi
		for _ in range(self.degree - 1):
			res = self.features_tensor_product(phi, res)
		return res


class PolyKernel(Kernel):
	def __init__(self, kernel, degree, *args):
		super(PolyKernel, self).__init__(*args)
		self.kernel = kernel
		self.degree = degree

	def _K(self, X, X_prime):
		K = self.kernel.K(X, X_prime)
		K_tiled = np.tile(K[None, :, :], (self.degree, 1, 1))
		assert K_tiled.shape == (self.degree,) + K.shape

		res = np.sum(np.cumprod(K_tiled, axis=0), axis=0)
		assert res.shape == K.shape

		return res
