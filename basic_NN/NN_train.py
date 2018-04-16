import random
import numpy as np
import pickle as pk
from read_and_deal import *
from compute_node import *
from fc_layer import *

def train_over(list1, layer1, layer2, loss, loop):
    loss_sum = 0
    for i in range(45000):
        y1 = layer1.fw(list1[i][0])
        y2 = layer2.fw(y1)
        lo = loss.fw(y2, list1[i][1])
        loss_sum += lo
        loss.bw()
        layer2.bw(loss.grad)
        layer1.bw(layer2.grad)
        if (i + 1) % 60 == 0:
            layer1.update(0.0009, 0.011)
            layer2.update(0.011, 0.08)
            if (i + 1) % 3000 == 0:
                print("current average loss: {0}".format(
                    loss_sum / 3000.))
                loss_sum = 0
    np.multiply(layer1.w, 0.996, out=layer1.w)
    np.multiply(layer1.b, 0.996, out=layer1.b)
    np.multiply(layer2.w, 0.996, out=layer2.w)
    np.multiply(layer2.b, 0.996, out=layer2.b)

def valid_over(list1, layer1, layer2):
    Count = 0
    for i in range(5000):
        y1 = layer1.fw(list1[i][0])
        y2 = layer2.fw(y1)
        y_hat = np.argmax(y2)
        if y_hat == list1[i][1]:
            Count += 1
    return (Count / 5000.)

def main():
    train_list = []
    fetch_train_list(train_list)
    w1 = np.random.normal(loc=0.0, scale=0.018, size=160*3072)
    w1 = np.reshape(w1, (160, 3072))
    w2 = np.random.normal(loc=0.0, scale=0.078, size=10*160)
    w2 = np.reshape(w2, (10, 160))
    b1 = np.zeros((160, 1), dtype=np.float64)
    b2 = np.zeros((10, 1), dtype=np.float64)
    layer1 = layer(w1, b1)
    layer2 = layer(w2, b2)
    loss = opt_squareLoss()
    loop = 0
    cur_acc = prev_acc = 0
    while True:
        loop += 1
        random.shuffle(train_list)
        train_over(train_list[0:45000], layer1, layer2, 
            loss, loop)
        acc = valid_over(train_list[45000:50000], 
            layer1, layer2)
        print("in loop {0}, accuracy is {1}".format(
            loop, acc))
        cur_acc += acc
        if loop % 10 == 0:
            cur_acc /= 10.
            if cur_acc - prev_acc < 0.0005:
                break
            prev_acc = cur_acc
            cur_acc = 0
    file1 = open("basicNN_res2.pkl", "wb")
    pk.dump({"w1": w1, "b1": b1, "w2":w2, "b2": b2 }, 
        file1)
    file1.close()

if __name__=="__main__":
    main()