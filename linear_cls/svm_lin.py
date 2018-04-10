import numpy as np
import pickle as pk
import cv2

def load_train_set(name):
    file1 = open(name, "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    return dict1

def regular_img_batch(img_bth):
    img_bth = np.divide(img_bth, 255.)
    return img_bth

def svm_pre(w, x):
    return np.matmul(w, x)

def svm_grad(x, y, y_right, dw):
    y = np.subtract(y, (y[y_right] - 1))
    sgn_y = np.divide(np.add(np.sign(y), 1), 2)
    num_grt = np.sum(sgn_y)
    np.subtract(dw, np.matmul(np.reshape(sgn_y, (10, 1)), np.reshape(x, (1, 3072))), out=dw)
    np.add(dw[y_right,:], np.multiply(x, num_grt), out=dw[y_right,:])

def update_w(w, img_bth, label_bth, learn_rate):
    for j in range(100):
        dw = np.zeros((10, 3072), dtype=np.float64)
        for k in range(100):
            i = j * 100 + k
            x = img_bth[i]
            y_right = label_bth[i]
            y = svm_pre(w, x)
            svm_grad(x, y, y_right, dw)
        np.divide(dw, 100., out=dw)
        np.add(np.multiply(w, 0.99998), np.multiply(dw, learn_rate), out=w)

def valid_w(w, img_batch, label_batch):
    Count_yes = 0
    for i in range(10000):
        x = img_batch[i]
        y_right = label_batch[i]
        y = np.matmul(w, x)
        y_pre = np.argmax(y)
        if y_pre == y_right:
            Count_yes += 1
    return (Count_yes / 10000)

def main():
    train_dicts = []
    for i in range(1, 6):
        dict1 = load_train_set("../data/data_batch_{0}".format(i))
        train_dicts.append(dict1)
    img_batch = []
    label_batch = []
    for dict1 in train_dicts:
        img_batch.append(regular_img_batch(dict1[b"data"]))
        label_batch.append(dict1[b"labels"])
    w = np.zeros((10, 3072), dtype=np.float64)
    learn_rate = 0.0005
    i = 0
    curr_acc = prev_acc = 0
    while True:
        j = i % 5
        valid_image_batch = img_batch[j]
        valid_label_batch = label_batch[j]
        for k in range(5):
            if k != j:
                update_w(w, img_batch[k], label_batch[k], learn_rate)
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
    file1 = open("svm_lin_res.pkl", "wb")
    pk.dump(w, file1)
    file1.close()


if __name__=="__main__":
    main()