import numpy as np 
import numpy.matlib 
from layer_set import layer_fc, layer_conv
from node_set import node_softmaxLoss, node_maxpool, node_flat
from datetime import datetime

def layer_fc_test():
    ly = layer_fc((10, 25), 0.2, (10,1), 1)
    print(ly.w)
    x = np.ones((25, 1))
    print(ly.fw(x))
    bk_grad = np.ones((10, 1))
    print(ly.bw(bk_grad))
    print(ly.accu_dw)

def layer_conv_test():
    ly = layer_conv((3, 5, 5), 1, 6, 2)
    x = np.ones((3, 32, 32))
    b = np.ones((6, 28, 28))
    beg = datetime.now()
    y = ly.fw(x)
    g = ly.bw(b)
    end = datetime.now()
    print(y)
    print(g)
    print(end - beg)

def some_node_test():
    node1 = node_softmaxLoss()
    y = np.matlib.randn((10, 1))
    print(y)
    loss = node1.fw(y, 5)
    print(loss)
    y = node1.bw()
    print(y)
    node2 = node_maxpool()
    x = np.reshape(np.array(np.matlib.randn((12, 6))), (2,6,6))
    print(x)
    y = node2.fw(x)
    print(y)
    g = node2.bw(np.ones((2, 3, 3)))
    print(g)
    node3 = node_flat()
    x = np.reshape(np.array(np.matlib.randn((12, 6))), (2,6,6))
    print(x)
    y = node3.fw(x)
    print(y)
    g = node3.bw(np.ones((72, 1)))
    print(g)

def main():
    some_node_test()
    return 0

if __name__=="__main__":
    main()