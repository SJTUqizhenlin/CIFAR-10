import numpy as np 
import matplotlib.pyplot as plt
import pickle as pk 
import random 
import numpy.random 
import numpy.matlib 
from layer_set import layer_conv, layer_fc 
from node_set import node_maxpool, node_flat, node_softmaxLoss
from data_augm import left_to_right, mult_by_scala

def fetch_data(data_list, n):
    print("loading data......")
    for i in range(1, 6):
        file1 = open("../data/data_batch_{0}".format(i), "rb")
        dict1 = pk.load(file1, encoding="bytes")
        file1.close()
        imgs = dict1[b"data"]
        labels = dict1[b"labels"]
        for j in range(n * 2000, n * 2000 + 2000):
            x = np.reshape(imgs[j], (3, 32, 32))
            x = np.subtract(np.divide(x, 64), 2)
            y = labels[j]
            tp = (x, y)
            data_list.append(tp)
            tp = (left_to_right(x), y)
            data_list.append(tp)
            tp = (mult_by_scala(x), y)
            data_list.append(tp)
    print("finished loading.")

def fetch_test(test_list):
    print("loading test......")
    file1 = open("../data/test_batch", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    imgs = dict1[b"data"]
    labels = dict1[b"labels"]
    for j in range(10000):
        x = np.reshape(imgs[j], (3, 32, 32))
        x = np.subtract(np.divide(x, 64), 2)
        y = labels[j]
        tp = (x, y)
        test_list.append(tp)
    print("finished loading.")

class net(object):
    def __init__(self):
        pass

    def build_net(self):
        self.ly1 = layer_conv((3,3,3), 0.18, 16)     # from 3,32,32 to 16,30,30
        self.ly2 = layer_conv((16,3,3), 0.08, 32)   # to 32,28,28
        self.nd1 = node_maxpool()                   # to 32,14,14
        self.ly3 = layer_conv((32,3,3), 0.058, 32)  # to 32,12,12
        self.ly4 = layer_conv((32,3,3), 0.058, 32)  # to 30,10,10
        self.nd2 = node_maxpool()                   # to 32,5,5
        self.ly5 = layer_conv((32,3,3), 0.058, 64)  # to 64,3,3
        self.ly6 = layer_conv((64,3,3), 0.042, 128)  # to 128,1,1
        self.nd3 = node_flat()                      # to 128,1
        self.ly7 = layer_fc((100, 128), 0.09, (100,1))     # to 100,1
        self.ly8 = layer_fc((10, 100), 0.1, (10,1))     # to 10,1
        self.nd4 = node_softmaxLoss()

def valid_over(list1, nn):
    reslist = [0] * 10
    Count_yes, loss = 0, 0
    list_len = len(list1)
    for i in range(list_len):
        x, y_true = list1[i][0], list1[i][1]
        y1 = nn.ly1.fw(x)
        y2 = nn.ly2.fw(y1)
        y3 = nn.nd1.fw(y2)
        y4 = nn.ly3.fw(y3)
        y5 = nn.ly4.fw(y4)
        y6 = nn.nd2.fw(y5)
        y7 = nn.ly5.fw(y6)
        y8 = nn.ly6.fw(y7)
        y9 = nn.nd3.fw(y8)
        y10 = nn.ly7.fw(y9)
        y11 = nn.ly8.fw(y10)
        y12 = np.argmax(y11)
        reslist[y12 - 1] += 1
        loss += nn.nd4.fw(y11, y_true)
        if y12 == y_true:
            Count_yes += 1
        if (i + 1) % 500 == 0:
            print("batch valid loss:", loss / (i + 1))
    print(reslist)
    return (Count_yes / list_len), (loss / list_len)

def train_over(list1, nn):
    loss, Count_yes = 0, 0
    list_len = len(list1)
    for i in range(list_len):
        x, y_true = list1[i][0], list1[i][1]
        y1 = nn.ly1.fw(x)
        y2 = nn.ly2.fw(y1)
        y3 = nn.nd1.fw(y2)
        y4 = nn.ly3.fw(y3)
        y5 = nn.ly4.fw(y4)
        y6 = nn.nd2.fw(y5)
        y7 = nn.ly5.fw(y6)
        y8 = nn.ly6.fw(y7)
        y9 = nn.nd3.fw(y8)
        y10 = nn.ly7.fw(y9)
        y11 = nn.ly8.fw(y10)
        y12 = np.argmax(y11)
        loss += nn.nd4.fw(y11, y_true)
        if y12 == y_true:
            Count_yes += 1
        g1 = nn.nd4.bw()
        g2 = nn.ly8.bw(g1)
        g3 = nn.ly7.bw(g2)
        g4 = nn.nd3.bw(g3)
        g5 = nn.ly6.bw(g4)
        g6 = nn.ly5.bw(g5)
        g7 = nn.nd2.bw(g6)
        g8 = nn.ly4.bw(g7)
        g9 = nn.ly3.bw(g8)
        g10 = nn.nd1.bw(g9)
        g11 = nn.ly2.bw(g10)
        g12 = nn.ly1.bw(g11)
        if (i + 1) % 50 == 0:
            nn.ly1.update(0.27, 0.45)
            nn.ly2.update(0.12, 0.45)
            nn.ly3.update(0.017, 0.09)
            nn.ly4.update(0.017, 0.09)
            nn.ly5.update(0.0017, 0.009)
            nn.ly6.update(0.0012, 0.009)
            nn.ly7.update(0.002, 0.006)
            nn.ly8.update(0.001, 0.003)
            if (i + 1) % 500 == 0:
                print("batch train loss:", loss / (i + 1))
    return (Count_yes / list_len), (loss / list_len)

def main():
    test_list = []
    fetch_test(test_list)
    nn = net()
    nn.build_net()
    loss_list_v = []
    loss_list_t = []
    acc_list_v = []
    acc_list_t = []
    for loop in range(500):
        data_list = []
        fetch_data(data_list, loop % 5)
        random.shuffle(data_list)
        acc, loss = valid_over(data_list[0:5000], nn)
        print("VALID loop {0}, acc is {1}, loss is {2}".format(loop, acc, loss))
        acc_list_v.append(acc)
        loss_list_v.append(loss)
        acc, loss = valid_over(test_list[0:10000], nn)
        print("TEST loop {0}, acc is {1}, loss is {2}".format(loop, acc, loss))
        acc_list_t.append(acc)
        loss_list_t.append(loss)
        acc, loss = train_over(data_list[5000:30000], nn)
        print("TRAIN loop {0}, acc is {1}, loss is {2}".format(loop, acc, loss))
        print()
    data_list = []
    fetch_data(data_list, loop % 5)
    random.shuffle(data_list)
    acc, loss = valid_over(data_list[0:5000], nn)
    print("VALID loop {0}, acc is {1}, loss is {2}".format(loop, acc, loss))
    acc_list_v.append(acc)
    loss_list_v.append(loss)
    acc, loss = valid_over(test_list[0:10000], nn)
    print("TEST loop {0}, acc is {1}, loss is {2}".format(loop, acc, loss))
    acc_list_t.append(acc)
    loss_list_t.append(loss)
    fig = plt.figure(num=17, figsize=(8, 8))
    fig1 = fig.add_subplot(2,1,1) #acc
    fig2 = fig.add_subplot(2,1,2) #loss
    fig1.set(ylabel="acc")
    fig2.set(ylabel="loss")
    fig1.plot(range(501), acc_list_v, label="valid", color="blue")
    fig1.plot(range(501), acc_list_t, label="test", color="red")
    fig2.plot(range(501), loss_list_v, label="valid", color="blue")
    fig2.plot(range(501), loss_list_t, label="test", color="red")
    plt.legend()
    plt.show()
    print("Memery recovering......")
    return 0

if __name__=="__main__":
    main()
    print("Memery recover finished.")


###
"""
GOT 69.8% TEST ACCURACY
"""
###