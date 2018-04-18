import numpy as np
from compute_node import *

class layer(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.mul_nd = opt_mul()
        self.add_nd = opt_add()
        self.tanh_nd = opt_tanh()
        self.times = 0
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.pre_delw = np.zeros(self.w.shape)
        self.pre_delb = np.zeros(self.b.shape)

    def fw(self, x):
        y1 = self.mul_nd.fw(self.w, x)
        y2 = self.add_nd.fw(y1, self.b)
        y3 = self.tanh_nd.fw(y2)
        return y3

    def bw(self, base_grad):
        self.tanh_nd.bw(base_grad)
        self.add_nd.bw(self.tanh_nd.grad)
        np.add(self.db, self.add_nd.grad_2, out=self.db)
        self.mul_nd.bw(self.add_nd.grad_1)
        np.add(self.dw, self.mul_nd.grad_1, out=self.dw)
        self.grad = self.mul_nd.grad_2
        self.times += 1

    def update(self, rate1, rate2):
        delta_w = np.add(np.multiply(self.dw, rate1), self.pre_delw)
        delta_b = np.add(np.multiply(self.db, rate2), self.pre_delb)
        np.subtract(self.w, np.divide(delta_w, self.times), out=self.w)
        np.subtract(self.b, np.divide(delta_b, self.times), out=self.b)
        self.pre_delw = np.multiply(delta_w, 0.63)
        self.pre_delb = np.multiply(delta_b, 0.63)
        self.times = 0
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
