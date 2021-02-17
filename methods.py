from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp

class Method(ABC):
    def __init__(self):
        self.kernel = None

    @abstractmethod
    def learn(self, X, Y):
        self.X = None
        self.alpha = None
        pass

    def predict(self, X_prime):
        K = self.kernel.K(X_prime, self.X)

        n, _ = self.alpha.shape
        m, _ = K.shape
        assert K.shape == (m, n)

        f = K @ self.alpha
        assert f.shape == (m, 1)

        h = np.sign(f)
        h[h==0] = 1

        return h

class KernelRidgeRegression(Method):
    def __init__(self, kernel, reg_val):
        self.kernel = kernel
        self.reg_val = reg_val

    def learn(self, X, Y):
        self.X = X
        K = self.kernel.K(X, X)
        n, d = K
        assert Y.shape == (n, 1)

        self.alpha = np.linalg.solve(K + self.reg_val * n * np.identity(n), Y)
        assert self.alpha.shape == (n, 1)

def KernelLogisticRegression(Method):
    def __init__(self, kernel, reg_val):
        self.kernel = kernel
        self.reg_val = reg_val

    def learn(self, X, Y):
        self.X = X
        K = self.kernel.K(X, X)
        n, d = K
        assert Y.shape == (n, 1)
        
        alpha = cp.Variable(n)
        objective = cp.Minimize((1/n) * cp.sum(cp.logistic(- cp.multiply(Y,  (K @ alpha)))) + (self.reg_val/2) * cp.quad_form(alpha, K))
        prob = cp.Problem(objective)
        prob.solve()

        self.alpha = alpha.value


