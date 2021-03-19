from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import scipy.sparse
import scipy.special
from tqdm import tqdm


# X and X_prime are (n, d) and (m, d) numpy vectors resp.
# Y is a (n,1) numpy vectors


class Kernel(ABC):
	@abstractmethod
	def fit(self, X, K=None, phi=None):
		pass

	@abstractmethod
	def make_rkhs_func(self, alpha):
		pass

	@abstractmethod
	def K(self, X, X_prime):
		pass


class NaiveKernel(ABC):
	def __init__(self):
		self.X = None

	@abstractmethod
	def K(self, X, X_prime):
		pass

	def fit(self, X, K=None, phi=None):
		self.X = X
		K = self.K(X, X) if K is None else K
		return K, None

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

	def K(self, X, X_prime):
		print("K en cours")
		feats = self.features(X)
		feats_prime = self.features(X_prime)
		return (feats.dot(feats_prime.T)).todense().A

	def fit(self, X, K=None, phi=None):
		self.feats = self.features(X) if phi is None else phi
		K = (self.feats.dot(self.feats.T)).todense().A if K is None else K
		return K, self.feats

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


class MismatchKernel(FeaturesKernel):
	def __init__(self, k, m, A):
		super(FeaturesKernel, self).__init__()
		self.k = k
		self.m = m
		assert k > m

		self.A = A
		self.max_n_matches_per_sample = sum([scipy.special.comb(k, l, exact=True) * (A - 1) ** l for l in range(m + 1)])

	def generate_matches(self, x, buff):
		"""
		given x a sample of size k, return naively all the sequences of size k
		which match with x up to m mismatches
		"""
		stack = []
		stack.append(([], 0))
		buff_index = 0
		while len(stack) > 0:
			l, curr_mismatches = stack.pop()
			if len(l) == self.k:
				buff[buff_index] = np.array(l)
				buff_index += 1
			elif curr_mismatches == self.m:
				buff[buff_index] = np.concatenate((np.array(l), x[len(l):]))
				buff_index += 1
			else:
				for a in range(self.A):
					stack.append((l + [a], curr_mismatches + int(a != x[len(l)])))
		assert buff_index == self.max_n_matches_per_sample

	def from_decomposition(self, decomp):

		assert decomp.shape == (self.max_n_matches_per_sample, self.k)

		return decomp.dot(self.A ** np.arange(self.k))

	def features(self, X):
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


class Polynomial(NaiveKernel):
	def __init__(self, degree):
		super(Polynomial, self).__init__()
		assert (isinstance(degree, int))
		self.degree = degree

	def K(self, X, X_prime):
		return (X @ X_prime.T) ** self.degree


class SumKernel(NaiveKernel):
	def __init__(self, kernel_1, kernel_2, d1, d2):
		super(SumKernel, self).__init__()
		self.kernel_1 = kernel_1
		self.kernel_2 = kernel_2
		self.d1 = d1
		self.d2 = d2

	def K(self, X, X_prime):
		assert X.shape[1] == self.d1 + self.d2
		assert X_prime.shape[1] == self.d1 + self.d2
		X_1, X_2 = X[:, :self.d1], X[:, self.d1:]
		X_1_prime, X_2_prime = X_prime[:, :self.d1], X_prime[:, self.d1:]
		return self.kernel_1.K(X_1, X_1_prime) + self.kernel_2.K(X_2, X_2_prime)

	def make_rkhs_func(self, alpha):
		f1 = self.kernel_1.make_rkhs_func(alpha)
		f2 = self.kernel_2.make_rkhs_func(alpha)

		def rkhs_func(Xprime):
			Xprime_1, Xprime_2 = Xprime[:, :self.d1], Xprime[:, self.d1:]
			return f1(Xprime_1) + f2(Xprime_2)

		return rkhs_func


class FeaturesPolyKernel(FeaturesKernel):
	def __init__(self, feature_kernel, degree):
		super(FeaturesPolyKernel, self).__init__()
		self.feature_kernel = feature_kernel
		self.degree = degree

	@staticmethod
	def features_tensor_product(phi, A):
		n, d = phi.shape
		_, p = A.shape
		assert A.shape == (n, p)

		res_nnz = sum([A.getrow(i).nnz * phi.getrow(i).nnz for i in range(n)])
		print(f"A nnz = {A.nnz} phi nnz = {phi.nnz} result nnz = {res_nnz}")
		res = scipy.sparse.vstack([scipy.sparse.kron(A.getrow(i), phi.getrow(i),
													 format='csr')
								   for i in range(n)], format='csr')
		assert res.shape == (n, d * p)

		return res

	def features(self, X, phi=None):
		phi = self.feature_kernel.features(X) if phi is None else phi

		res = phi
		for _ in range(self.degree - 1):
			res = self.features_tensor_product(phi, res)
		return res


class PolyKernel(NaiveKernel):
	def __init__(self, kernel, degree):
		super(PolyKernel, self).__init__()
		self.kernel = kernel
		self.degree = degree

	def K(self, X, X_prime):
		K = self.kernel.K(X, X_prime)
		K_tiled = np.tile(K[None, :, :], (self.degree, 1, 1))
		assert K_tiled.shape == (self.degree,) + K.shape

		res = np.sum(np.cumprod(K_tiled, axis=0), axis=0)
		assert res.shape == K.shape

		return res
