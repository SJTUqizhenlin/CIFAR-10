import numpy as np
import pickle as pk

def gray_scale(x):
    g = np.reshape(x, (3, 1024))
    scalar = np.mat([0.3, 0.59, 0.11])
    g = np.matmul(scalar, g, out=None)
    return np.reshape(g, (1024, 1))

def to_colum(x):
    g = np.reshape(x, (3072, 1))
    return g

def regu_img(x):
    tmp = np.divide(x, 255., out=None)
    tmp = np.subtract(tmp, 0.5, out=None)
    return tmp

def fetch_train_list(train_list):
    print("loading training data...")
    for i in range(1, 6):
        file1 = open("../data/data_batch_{0}".format(i), "rb")
        dict1 = pk.load(file1, encoding="bytes")
        file1.close()
        imgs = dict1[b"data"]
        labels = dict1[b"labels"]
        for j in range(10000):
            x = imgs[j]
            y = labels[j]
            x = to_colum(x)
            x = regu_img(x)
            tp = (x, y)
            train_list.append(tp)
    print("finished loading.")
