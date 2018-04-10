import numpy as np
import pickle as pk
import cv2


def load_train_set(name):
    file1 = open(name, "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    return dict1

def softmax_pre(w, x):
    y = np.matmul(w, x)
    y = np.exp(y)
    y_sum = np.sum(y)
    y = np.divide(y, y_sum)
    return y

def softmax_grad(x, y, y_right, dw):
    y = np.reshape(y, (10, 1))
    x = np.reshape(x, (1, 3072))
    np.subtract(dw, np.matmul(y, x), out=dw)
    np.add(dw[y_right,:], np.reshape(x, (3072,)), out=dw[y_right,:])

def update_w(w, image_batch, label_batch, learn_rate):
    for a in range(100):
        dw = np.zeros((10, 3072), dtype=np.float64)
        for b in range(100):
            i = a * 100 + b
            x = image_batch[i]
            y_right = label_batch[i]
            y = softmax_pre(w, x)
            softmax_grad(x, y, y_right, dw)
        np.divide(dw, 100., out=dw)
        np.add(np.multiply(w, 0.999998), np.multiply(dw, learn_rate), out=w)

def valid_w(w, image_batch, label_batch):
    Count_yes = 0
    for i in range(10000):
        x = image_batch[i]
        y_right = label_batch[i]
        y = softmax_pre(w, x)
        y_res = np.argmax(y)
        if y_res == y_right:
            Count_yes += 1
    return (Count_yes / 10000)

def regular_img_batch(img_bth):
    img_bth = np.divide(img_bth, 255.)
    return img_bth

def main():
    train_sets_dicts = []
    for i in range(1, 6):
        dict1 = load_train_set("../data/data_batch_{0}".format(i))
        train_sets_dicts.append(dict1)
    image_batches = []
    label_batches = []
    for dict1 in train_sets_dicts:
        image_batches.append(regular_img_batch(dict1[b"data"]))
        label_batches.append(dict1[b"labels"])
    w = np.zeros((10, 3072), dtype=np.float64)
    learn_rate = 0.0005
    i = 0
    prev_acc = curr_acc = 0
    while True:
        j = i % 5
        valid_image_batch = image_batches[j]
        valid_label_batch = label_batches[j]
        for k in range(5):
            if (k != j):
                update_w(w, image_batches[k], label_batches[k], learn_rate)
        print("After training loop {0}:".format(i + 1), end=" ")
        acc = valid_w(w, valid_image_batch, valid_label_batch)
        print("accuracy is {0}".format(acc))
        curr_acc += acc
        if i % 10 == 9:
            curr_acc /= 10.
            if curr_acc - prev_acc < 0.0003:
                break
            prev_acc = curr_acc
            curr_acc = 0
        i += 1
    print("dumping w")
    file1 = open("softmax_lin_res.pkl", "wb")
    pk.dump(w, file1)
    file1.close()


if __name__=="__main__":
    main()