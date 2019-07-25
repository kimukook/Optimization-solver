'''
This is a python script that implements ADAptive Moment estimation algorithm to find the optimizer of a stochastic
function.

Author  :  Muhan Zhao
Date    :  Jul. 24, 2019
Location:  West Hill, LA, CA
'''
# reference link:
import numpy as np


class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        assert 0 < beta1 < 1 and 0 < beta2 < 1, 'beta1 and beta2 should be within 0 and 1'
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        self.m_hat = []
        self.v_hat = []
        self.x = []
        self.previous_x = []
        self.g = []
        self.f = []

    def adam(self, func, jac, x):
        self.x = x
        while 1:
            self.t += 1
            self.g = jac(self.x)
            self.m = self.beta1 * self.m + (1 - self.beta1) * self.g
            self.v = self.beta2 * self.v + (1 - self.beta2) * np.multiply(self.g, self.g)
            self.m_hat = self.m / (1 - self.beta1 ** self.t)
            self.v_hat = self.v / (1 - self.beta2 ** self.t)
            self.previous_x = self.x
            self.x = self.x - self.alpha * self.m_hat / (np.sqrt(self.v_hat) + self.epsilon)
            if np.linalg.norm(self.x - self.previous_x) < self.epsilon:
                break
        return self.x


if __name__ == '__main__':

    def func(x):
        b = 2 * np.ones((2, 1))
        return np.dot(x.T, x) - 2 * np.dot(x.T, b) + np.dot(b.T, b)

    def grad_func(x):  # calculates the gradient
        return 2 * x - 2 * 2 * np.ones((2, 1))

    opt = Adam()
    x = np.ones((2, 1))
    sol = opt.adam(func, grad_func, x)
    print(sol)
