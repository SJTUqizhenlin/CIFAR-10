import pickle as pk
import numpy as np
from compute_node import *
from fc_layer import *

def fetch_data(test_list):
    print("loading test data...")
    file1 = open("../data/test_batch", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    for i in range(10000):
        x = np.reshape(dict1[b"data"][i], (3072, 1))
        x = np.subtract(np.divide(x, 128.), 1.)
        y = dict1[b"labels"][i]
        tp = (x, y)
        test_list.append(tp)
    print("finish loading")

def test_over(test_list, layer1, layer2, layer3):
    Count_acc = 0
    for i in range(10000):
        x = test_list[i][0]
        y_true = test_list[i][1]
        y1 = layer1.fw(x)
        y2 = layer2.fw(y1)
        y3 = layer3.fw(y2)
        y_hat = np.argmax(y3)
        if y_hat == y_true:
            Count_acc += 1
        if (i + 1) % 1000 == 0:
            print("in {0} tests, {1} are right".format(
                i + 1, Count_acc))
    return (Count_acc / 10000)

def main():
    test_list = []
    fetch_data(test_list)
    file2 = open("FC_NN_res.pkl", "rb")
    res_dict1 = pk.load(file2)
    file2.close()
    w1, b1 = res_dict1["w1"], res_dict1["b1"]
    w2, b2 = res_dict1["w2"], res_dict1["b2"]
    w3, b3 = res_dict1["w3"], res_dict1["b3"]
    layer1 = layer(w1, b1)
    layer2 = layer(w2, b2)
    layer3 = layer(w3, b3)
    acc = test_over(test_list, layer1, 
        layer2, layer3)
    print("on test batch, accuracy: {0}".format(acc))

if __name__=="__main__":
    main()