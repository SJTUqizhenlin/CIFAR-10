import numpy as np
import pickle as pk
from extract_HOG import *
from fc_layer import *

def test_over(datas, labels, layer1, layer2):
    Count_yes = 0
    for i in range(10000):
        x = get_HOGmat(datas[i])
        x = np.reshape(np.subtract(x, 0.281), (2187, 1))
        label = labels[i]
        y1 = layer1.fw(x)
        y2 = layer2.fw(y1)
        y_hat = np.argmax(y2)
        if y_hat == label:
            Count_yes += 1
        if (i + 1) % 1000 == 0:
            print("{0} are correct in {1} imgs".format(
                Count_yes, i + 1))
    print("on test batch, accuracy is {0}".format(
        Count_yes / 10000))

def main():
    file1 = open("../data/test_batch", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    file2 = open("hogNN_res.pkl", "rb")
    resdic = pk.load(file2)
    file2.close()
    w1, w2 = (resdic["w1"], resdic["w2"])
    b1, b2 = (resdic["b1"], resdic["b2"])
    layer1 = layer(w1, b1)
    layer2 = layer(w2, b2)
    datas = dict1[b"data"]
    labels = dict1[b"labels"]
    test_over(datas, labels, layer1, layer2)

if __name__=="__main__":
    main()