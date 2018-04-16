import numpy as np
import pickle as pk
from compute_node import *
from fc_layer import *

def regular_img(x):
    g = np.reshape(x, (3072, 1))
    g = np.divide(g, 255., out=None)
    g = np.subtract(g, 0.5, out=None)
    return g

def load_test_list(test_list):
    file1 = open("../data/test_batch", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    for i in range(10000):
        x = regular_img(dict1[b"data"][i])
        y = dict1[b"labels"][i]
        tp = (x, y)
        test_list.append(tp)

def test_over(lst, ly1, ly2):
    Count_yes = 0
    for i in range(10000):
        x = lst[i][0]
        y_right = lst[i][1]
        y1 = ly1.fw(x)
        y2 = ly2.fw(y1)
        y_hat = np.argmax(y2)
        if y_hat == y_right:
            Count_yes += 1
    return (Count_yes / 10000)

def main():
    test_list = []
    load_test_list(test_list)
    file2 = open("basicNN_res2.pkl", "rb")
    para_dict = pk.load(file2)
    file2.close()
    w1 = para_dict["w1"]
    w2 = para_dict["w2"]
    b1 = para_dict["b1"]
    b2 = para_dict["b2"]
    layer1 = layer(w1, b1)
    layer2 = layer(w2, b2)
    acc = test_over(test_list, layer1, layer2)
    print("on test batch, accuracy: {0}".format(
        acc))

if __name__=="__main__":
    main()