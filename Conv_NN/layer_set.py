import numpy as np
import numpy.matlib 
from node_set import node_add, node_matmul, node_relu 
from node_set import node_conv, node_maxpool

class layer_fc(object):
    def __init__(self, wshape, wscale, bshape):
        self.wscale = wscale
        self.node1 = node_matmul()
        self.node2 = node_add()
        self.node3 = node_relu()
        w = np.array(numpy.matlib.randn(wshape) * wscale) 
        b = np.zeros(bshape)
        self.w = w
        self.b = b
        self.batch_num = 0
        self.accu_dw = np.zeros(wshape)
        self.accu_db = np.zeros(bshape)
        self.vw = np.zeros(wshape) 
        self.vb = np.zeros(bshape)

    def fw(self, x):
        y1 = self.node1.fw(self.w, x)
        y2 = self.node2.fw(y1, self.b)
        y3 = self.node3.fw(y2)
        return y3

    def bw(self, bk_grad):
        self.batch_num += 1
        g3 = self.node3.bw(bk_grad)
        g2_1, g2_2 = self.node2.bw(g3)
        np.add(self.accu_db, g2_2, out=self.accu_db)
        g1_1, g1_2 = self.node1.bw(g2_1)
        np.add(self.accu_dw, g1_1, out=self.accu_dw)
        return g1_2

    def update(self, rate1, rate2):
        dw = np.divide(self.accu_dw, self.batch_num)
        db = np.divide(self.accu_db, self.batch_num)
        self.vw = self.vw * 0.9 + dw
        self.vb = self.vb * 0.9 + db
        np.subtract(self.w, np.multiply(self.vw, rate1), out=self.w)
        np.subtract(self.b, np.multiply(self.vb, rate2), out=self.b)
        self.batch_num = 0
        self.accu_dw = self.accu_dw * 0.0
        self.accu_db = self.accu_db * 0.0

class layer_conv(object):
    def randn3D(self, kshape):
        a = np.matlib.randn((kshape[0]*kshape[1], kshape[2]))
        return np.reshape(np.array(a), kshape)

    def __init__(self, kshape, kscale, knum):
        self.knum = knum 
        self.kshape, self.kscale = kshape, kscale 
        self.conv_node_list = []
        self.node2 = node_relu()
        self.k_list = []
        self.b_list = []
        self.accu_dk_list = []
        self.accu_db_list = [0] * knum 
        self.vk_list = []
        self.vb_list = [0] * knum 
        for i in range(knum):
            self.conv_node_list.append(node_conv())
            self.k_list.append(self.randn3D(kshape) * kscale)
            self.b_list.append(0)
            self.accu_dk_list.append(np.zeros(kshape))
            self.vk_list.append(np.zeros(kshape))
        self.batch_num = 0

    def fw(self, x):
        self.xshape = x.shape
        yshape1 = x.shape[1] - self.kshape[1] + 1
        yshape2 = x.shape[2] - self.kshape[2] + 1
        y1 = np.zeros((self.knum, yshape1, yshape2))
        for i in range(self.knum):
            y1[i] = self.conv_node_list[i].fw(x, self.k_list[i], self.b_list[i])
        y2 = self.node2.fw(y1)
        return y2 

    def bw(self, bk_grad):
        self.batch_num += 1
        grad = np.zeros(self.xshape)
        relu_grad = self.node2.bw(bk_grad)
        for i in range(self.knum):
            g1, g2, g3 = self.conv_node_list[i].bw(relu_grad[i])
            np.add(grad, g1, out=grad)
            np.add(self.accu_dk_list[i], g2, out=self.accu_dk_list[i])
            self.accu_db_list[i] += g3
        return grad 

    def update(self, rate1, rate2):
        for e in self.accu_dk_list:
            np.divide(e, self.batch_num, out=e)
        for e in self.b_list:
            e /= self.batch_num
        for i in range(self.knum):
            np.multiply(self.vk_list[i], 0.9, out=self.vk_list[i])
            np.add(self.vk_list[i], self.accu_dk_list[i], out=self.vk_list[i])
            self.vb_list[i] = self.vb_list[i] * 0.9 + self.accu_db_list[i]
        for i in range(self.knum):
            np.subtract(self.k_list[i], self.vk_list[i] * rate1, 
                out=self.k_list[i])
            self.b_list[i] -= self.vb_list[i] * rate2 
        self.batch_num = 0
        for e in self.accu_dk_list:
            np.multiply(e, 0.0)
        self.accu_db_list = [0] * self.knum 
