import pytest
import numpy as np
import methods
import kernels

gen = np.random

n = 1000
d = 100
l = 106
A = 3


@pytest.fixture
def x():
	x = gen.random(size=(n, d)) + 1
	x[:n // 2, :] *= -1
	assert x.shape == (n, d)
	return x


@pytest.fixture
def y():
	y = np.ones((n, 1))
	y[:n // 2] = -1
	assert y.shape == (n, 1)
	return y


@pytest.fixture
def seq():
	seq = gen.choice([0, 1], size=(n, l))
	seq[n // 2:, :] += 1
	assert seq.shape == (n, l)
	return seq

@pytest.fixture
def simple_seq():
	seq = np.zeros((10, l))
	seq[10 // 2:, :] += 1
	return seq

x_train, x_test = x, x
y_train, y_test = y, y
seq_train, seq_test = seq, seq

TEST_METHODS = [
	methods.KernelRidgeRegression,
	methods.KernelLogisticRegression,
	methods.SVM
]

TEST_KERNELS = [
	(kernels.Linear(), .1),
	(kernels.Gaussian(.1), .1),
	(kernels.Polynomial(degree=3), .1)
]

TEST_SEQ_KERNELS = [
	#kernels.SpectrumKernel(1),
	#kernels.SpectrumKernel(2),
	kernels.MismatchKernel(3, 1, A)
]


@pytest.mark.parametrize('method', TEST_METHODS)
@pytest.mark.parametrize('kernel_with_reg', TEST_KERNELS)
def test_methods(method, kernel_with_reg, x_train, y_train, x_test, y_test):
	kernel, reg = kernel_with_reg
	meth = method(kernel, reg_val=reg)
	meth.learn(x_train, y_train)
	y_est = meth.predict(x_train)
	np.testing.assert_equal(y_est, y_train)
	y_est = meth.predict(x_test)
	np.testing.assert_equal(y_est, y_test)


@pytest.mark.parametrize('method', TEST_METHODS)
@pytest.mark.parametrize('kernel', TEST_SEQ_KERNELS)
def test_seq_methods(method, kernel, seq_train, y_train, seq_test, y_test):
	K = kernel.fit(seq_train)
	np.testing.assert_equal(K, K.T)
	assert (np.all(np.linalg.eigvalsh(K) > 0)),f"Sp(K)={np.sort(np.real(np.linalg.eigvalsh(K)))}"
	meth = method(kernel)
	meth.learn(seq_train, y_train)
	y_est = meth.predict(seq_train)
	np.testing.assert_equal(y_est, y_train)
	y_est = meth.predict(seq_test)
	np.testing.assert_equal(y_est, y_test)

def test_mismatch_kernel(simple_seq):
	n = len(simple_seq)
	kernel = kernels.MismatchKernel(3, 1, 2)
	K = kernel.fit(simple_seq)
	print(K)
	assert (K[:n//2, :n//2] >= 1).all()
	assert (K[n//2:, n//2:] >= 1).all()
	zeros = np.zeros((n//2, n//2))
	np.testing.assert_equal(K[:n//2, n//2:], zeros)
	np.testing.assert_equal(K[n//2:, :n//2], zeros)




