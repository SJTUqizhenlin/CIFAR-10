import numpy as np

class opt_add(object):
    def __init__(self):
        pass

    def fw(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        y = np.add(x1, x2, out=None)
        return y

    def bw(self, base_grad):
        self.grad_1 = np.copy(base_grad)
        self.grad_2 = np.copy(base_grad)

class opt_mul(object):
    def __init__(self):
        pass

    def fw(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        y = np.matmul(x1, x2, out=None)
        return y

    def bw(self, base_grad):
        tr_1 = np.transpose(self.x1, axes=None)
        tr_2 = np.transpose(self.x2, axes=None)
        self.grad_1 = np.matmul(base_grad, tr_2, out=None)
        self.grad_2 = np.matmul(tr_1, base_grad, out=None)

class opt_tanh(object):
    def __init__(self):
        self.b = 2 / 3
        e = np.e
        self.a = np.divide(np.power(e, self.b) + np.power(e, -self.b), 
            np.power(e, self.b) - np.power(e, -self.b))

    def fw(self, x):
        self.x = x
        y = np.multiply(np.tanh(np.multiply(x, self.b)), self.a)
        return y

    def bw(self, base_grad):
        local = np.ones(self.x.shape, dtype=np.float64)
        divi = np.square(np.cosh(np.multiply(self.x, self.b)))
        np.divide(local, divi, out=local)
        np.multiply(local, self.a * self.b, out=local)
        self.grad = np.multiply(local, base_grad, out=None)

class opt_squareLoss(object):
    def __init__(self):
        pass

    def fw(self, y, y_right):
        self.y = y
        self.y_right = y_right
        right_v = np.negative(np.ones(y.shape, dtype=np.float64))
        right_v[y_right,:] = np.negative(right_v[y_right,:])
        loss = np.sum(np.square(np.subtract(y, right_v)))
        return loss * 0.5

    def bw(self):
        tmp = np.copy(self.y)
        indicate = np.ones(self.y.shape, dtype=np.float64)
        indicate[self.y_right,:] = np.negative(indicate[self.y_right,:])
        self.grad = np.add(tmp, indicate, out=None)
