import numpy as np 
from func_cython.func_cython import conv_func
from func_cython.func_cython import pool_func

class node_add(object):
    def __init__(self):
        pass
    
    def fw(self, x1, x2):
        y = np.add(x1, x2)
        return y
    
    def bw(self, bk_grad):
        return bk_grad, bk_grad
    
class node_matmul(object):
    def __init__(self):
        pass
    
    def fw(self, x1, x2):
        y = np.matmul(x1, x2)
        self.x1 = x1 
        self.x2 = x2 
        return y 

    def bw(self, bk_grad):
        trx1 = np.transpose(self.x1)
        trx2 = np.transpose(self.x2)
        grad_1 = np.matmul(bk_grad, trx2)
        grad_2 = np.matmul(trx1, bk_grad)
        return grad_1, grad_2 

class node_relu(object):
    def __init__(self):
        pass

    def fw(self, x):
        self.x = x
        self.sig = (np.sign(x) + 1.) / 2
        y = np.multiply(self.x, self.sig)
        return y

    def bw(self, bk_grad):
        grad = np.multiply(bk_grad, self.sig)
        return grad 

class node_softmaxLoss(object):
    def __init__(self):
        pass

    def fw(self, y, y_true):
        self.y_true = y_true
        if (np.max(y) > 700):
            print("Warning: Exp Overflow.")
        mid = np.exp(y)
        midsum = np.sum(mid)
        self.p = np.divide(mid, midsum)
        loss = np.negative(np.log(self.p[y_true, 0]))
        return loss

    def bw(self):
        mid = np.copy(self.p)
        np.subtract(mid[self.y_true], 1, out=mid[self.y_true])
        return mid 

class node_conv(object):
    def __init__(self):
        pass
    
    def fw(self, x, k, b):
        self.x = x
        self.k = k 
        y = conv_func.convfw_cython(x, k)
        y = np.add(y, b)
        return y 
    
    def bw(self, bk_grad):
        grad_1, grad_2 = conv_func.convbw_cython(bk_grad, self.x, self.k)
        grad_3 = np.average(bk_grad)
        return grad_1, grad_2, grad_3 

class node_maxpool(object):
    def __init__(self):
        pass

    def fw(self, x):
        self.x = x 
        y = pool_func.poolfw_cython(x)
        return y 

    def bw(self, bk_grad):
        grad = pool_func.poolbw_cython(bk_grad, self.x)
        return grad 

class node_flat(object):
    def __init__(self):
        pass 

    def fw(self, x):
        self.xshape = x.shape 
        xsize = x.shape[0] * x.shape[1] * x.shape[2]
        y = np.reshape(x, (xsize, 1))
        return y 

    def bw(self, bk_grad):
        grad = np.reshape(bk_grad, self.xshape)
        return grad 