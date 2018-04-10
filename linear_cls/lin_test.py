import numpy as np
import pickle as pk
import cv2
from softmax_lin import*


def show_by_cv2(img, name, enlarge=1):
    img1 = np.copy(img)
    img1 = change_to_2D(img1)
    img1[:,:,[0, 2]] = img1[:,:,[2, 0]]
    if not enlarge == 1:
        img1 = cv2.resize(img1, (0,0), 
            fx=enlarge, fy=enlarge, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(name, img1)

def change_to_2D(img1):
    img1 = np.reshape(img1, (3, 32, 32))
    img1 = img1.transpose(1, 2, 0)
    return img1

def test_all(imgs, labels, w):
    Count_yes = 0
    imgs = np.divide(imgs, 255.)
    for i in range(10000):
        y = np.argmax(softmax_pre(w, imgs[i]))
        if y == labels[i]:
            Count_yes += 1
    print("accuracy is: {0}".format(Count_yes / 10000))

def getMeta(img, w, metas):
    show_by_cv2(img, "pic", 2)
    cv2.waitKey(0)
    img = np.divide(img, 255.)
    y = np.argmax(softmax_pre(w, img))
    name = metas[y].decode("utf-8")
    print("this is: {0}".format(name))

def main():
    test_batch = load_train_set("../data/test_batch")
    imgs = test_batch[b"data"]
    labels = test_batch[b"labels"]
    metas = load_train_set("../data/batches.meta")[b"label_names"]
    mthd = input("which method?(softmax, svm): ")
    w_file = open("{0}_lin_res.pkl".format(mthd), "rb")
    w = pk.load(w_file)
    w_file.close()
    order = int(input("1.test accuracy 2.test one picture: "))
    if order == 1:
        test_all(imgs, labels, w)
    if order == 2:
        no = int(input("choose a pic(0--9999): "))
        getMeta(imgs[no], w, metas)


if __name__=="__main__":
    main()