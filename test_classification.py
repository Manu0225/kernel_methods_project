import pytest
import numpy as np
gen = np.random

import methods
import kernels

n = 2000
d = 100

@pytest.fixture
def x():
    x = gen.random(size=(n, d)) + 1
    x[:n//2, :] *= -1
    assert x.shape == (n, d)
    return x

@pytest.fixture
def y():
    y = np.ones((n, 1))
    y[:n//2] = -1
    assert y.shape == (n, 1)
    return y

x_train, x_test = x, x
y_train, y_test = y, y

TEST_METHODS = [
            methods.KernelRidgeRegression,
            methods.KernelLogisticRegression,
        ]

TEST_KERNELS = [
            kernels.Linear,
            kernels.Gaussian,
        ]

@pytest.mark.parametrize('method', TEST_METHODS)
@pytest.mark.parametrize('kernel', TEST_KERNELS)
def test_methods(method, kernel, x_train, y_train, x_test, y_test):
    meth = method(kernel())
    meth.learn(x_train, y_train)
    y_est = meth.predict(x_train)
    np.testing.assert_equal(y_est, y_train)
    y_est = meth.predict(x_test)
    np.testing.assert_equal(y_est, y_test)
