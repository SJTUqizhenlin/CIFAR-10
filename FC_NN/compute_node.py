import numpy as np

class opt_add(object):
    def __init__(self):
        pass

    def fw(self, x1, x2):
        y = np.add(x1, x2)
        return y

    def bw(self, base_grad):
        self.grad1 = np.copy(base_grad)
        self.grad2 = np.copy(base_grad)

class opt_matmul(object):
    def __init__(self):
        pass

    def fw(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        y = np.matmul(x1, x2)
        return y

    def bw(self, base_grad):
        tr1 = np.transpose(self.x1)
        tr2 = np.transpose(self.x2)
        self.grad1 = np.matmul(base_grad, tr2)
        self.grad2 = np.matmul(tr1, base_grad)

class opt_relu(object):
    def __init__(self):
        pass

    def fw(self, x):
        self.x = x
        self.sig = np.divide(np.add(np.sign(x), 1), 2)
        y = np.multiply(self.x, self.sig)
        return y

    def bw(self, base_grad):
        self.grad = np.multiply(base_grad, self.sig)

class opt_softmaxLoss(object):
    def __init__(self):
        pass

    def fw(self, y, y_true):
        self.y_true = y_true
        if (np.max(y) > 40.):
            print("Error: Exp Overflow.")
        mid = np.exp(y)
        midsum = np.sum(mid)
        self.p = np.divide(mid, midsum)
        loss = np.negative(np.log(self.p[y_true, 0]))
        return loss

    def bw(self):
        mid = np.copy(self.p)
        np.subtract(mid[self.y_true], 1, out=mid[self.y_true])
        self.grad = mid
