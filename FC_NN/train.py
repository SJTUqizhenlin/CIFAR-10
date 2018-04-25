import random
import numpy as np
import pickle as pk
from compute_node import *
from fc_layer import *

def fetch_data(data_list):
    print("loading data...")
    for i in range(1, 6):
        file1 = open("../data/data_batch_{0}".format(i), "rb")
        dict1 = pk.load(file1, encoding="bytes")
        file1.close()
        imgs = dict1[b"data"]
        labels = dict1[b"labels"]
        for j in range(10000):
            x = np.reshape(imgs[j], (3072, 1))
            x = np.subtract(np.divide(x, 128.), 1.)
            y = labels[j]
            tp = (x, y)
            data_list.append(tp)
    print("finish loading.")

def valid_over(list1, layer1, layer2, layer3, lossnode):
    Count = 0
    Loss = 0
    for i in range(10000):
        y1 = layer1.fw(list1[i][0])
        y2 = layer2.fw(y1)
        y3 = layer3.fw(y2)
        Loss += lossnode.fw(y3, list1[i][1])
        y_hat = np.argmax(y3)
        if y_hat == list1[i][1]:
            Count += 1
    return (Count / 10000), (Loss / 10000)

def train_over(list1, layer1, layer2, layer3, lossnode):
    l = 0
    for i in range(40000):
        y1 = layer1.fw(list1[i][0])
        y2 = layer2.fw(y1)
        y3 = layer3.fw(y2)
        l += lossnode.fw(y3, list1[i][1])
        lossnode.bw()
        layer3.bw(lossnode.grad)
        layer2.bw(layer3.grad)
        layer1.bw(layer2.grad)
        if (i + 1) % 50 == 0:
            layer1.update(0.0002, 0.012)
            layer2.update(0.0008, 0.012)
            layer3.update(0.0023, 0.012)
            if (i + 1) % 1000 == 0:
                print("batch loss:", l / 1000)
                l = 0
    np.multiply(layer1.w, 0.994, out=layer1.w)
    np.multiply(layer1.b, 0.994, out=layer1.b)
    np.multiply(layer2.w, 0.994, out=layer2.w)
    np.multiply(layer2.b, 0.994, out=layer2.b)
    np.multiply(layer3.w, 0.994, out=layer3.w)
    np.multiply(layer3.b, 0.994, out=layer3.b)

def main():
    data_list = []
    fetch_data(data_list)
    w1 = np.random.normal(loc=0.0, scale=0.017, size=800*3072)
    w1 = np.reshape(w1, (800, 3072))
    b1 = np.zeros((800, 1))
    layer1 = layer(w1, b1)
    w2 = np.random.normal(loc=0.0, scale=0.064, size=90*800)
    w2 = np.reshape(w2, (90, 800))
    b2 = np.zeros((90, 1))
    layer2 = layer(w2, b2)
    w3 = np.random.normal(loc=0.0, scale=0.19, size=10*90)
    w3 = np.reshape(w3, (10, 90))
    b3 = np.zeros((10, 1))
    layer3 = layer(w3, b3)
    lossnode = opt_softmaxLoss()
    for loop in range(50):
        random.shuffle(data_list)
        ac, lo = valid_over(data_list[0:10000], 
            layer1, layer2, layer3, lossnode)
        print("In loop {0}, acc: {1}, loss: {2}".format(
            loop, ac, lo))
        train_over(data_list[10000:50000], 
            layer1, layer2, layer3, lossnode)
    print("dumping results...")
    file1 = open("FC_NN_res.pkl", "wb")
    pk.dump({"w1": w1, "b1": b1, "w2":w2, "b2": b2,
        "w3": w3, "b3": b3}, file1)
    file1.close()

if __name__=="__main__":
    main()