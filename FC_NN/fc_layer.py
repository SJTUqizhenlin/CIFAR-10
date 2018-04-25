import numpy as np
from compute_node import *

class layer(object):
    def __init__(self, w, b):
        self.matmul_node = opt_matmul()
        self.add_node = opt_add()
        self.relu_node = opt_relu()
        self.w = w
        self.b = b
        self.times = 0
        self.dw = np.zeros(w.shape)
        self.db = np.zeros(b.shape)
        self.vw = np.zeros(w.shape)
        self.vb = np.zeros(b.shape)

    def fw(self, x):
        y1 = self.matmul_node.fw(self.w, x)
        y2 = self.add_node.fw(y1, self.b)
        y3 = self.relu_node.fw(y2)
        return y3

    def bw(self, base_grad):
        self.times += 1
        self.relu_node.bw(base_grad)
        self.add_node.bw(self.relu_node.grad)
        np.add(self.db, self.add_node.grad2, out=self.db)
        self.matmul_node.bw(self.add_node.grad1)
        np.add(self.dw, self.matmul_node.grad1, out=self.dw)
        self.grad = self.matmul_node.grad2

    def update(self, rate1, rate2):
        np.divide(self.dw, self.times, out=self.dw)
        np.divide(self.db, self.times, out=self.db)
        self.vw = np.add(np.multiply(self.vw, 0.9), self.dw)
        self.vb = np.add(np.multiply(self.vb, 0.9), self.db)
        np.subtract(self.w, np.multiply(self.vw, rate1), out=self.w)
        np.subtract(self.b, np.multiply(self.vb, rate2), out=self.b)
        self.times = 0
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)