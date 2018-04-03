import numpy as np
import pickle as pk
import cv2
from para_test import*

def test_accuracy(train_datas, train_labels, test_data, test_label):
    Count_yes = 0
    res_data = np.zeros((5,10000))
    vote = np.zeros((10), dtype=np.int8)
    for j in range(10000):
        predict = predict_img_with(test_data[j], res_data, vote, train_datas, train_labels)
        if predict == test_label[j]:
            Count_yes += 1
        if (j % 10 == 9):
            print("have finished {0}, accuracy is {1}"
                .format(j + 1, Count_yes / (j + 1)))

def test_one(test_data, train_datas, train_labels):
    n = int(input("Which pic? (0-9999):"))
    img1 = test_data[n]
    res_data = np.zeros((5,10000))
    vote = np.zeros((10), dtype=np.int8)
    predict = predict_img_with(img1, res_data, vote, train_datas, train_labels)
    file1 = open("../data/batches.meta", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    name_list = dict1[b"label_names"]
    name_res = name_list[predict].decode("utf-8")
    show_by_cv2(img1, "picture", 2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("This is {}".format(name_res))


def main():
    train_dicts = []
    for i in range(1, 6):
        train_dicts.append(load_train_dicts("../data/data_batch_" + str(i)))
    test_dict = load_train_dicts("../data/test_batch")
    train_datas = []
    train_labels = []
    for train_dict in train_dicts:
        train_datas.append(train_dict[b"data"])
        train_labels.append(train_dict[b"labels"])
    test_data = test_dict[b"data"]
    test_label = test_dict[b"labels"]
    order = input("Select func:\n0.test accuracy, 1.test one pic:")
    if (order == "0"):
        test_accuracy(train_datas, train_labels, test_data, test_label)
    else:
        test_one(test_data, train_datas, train_labels)



if __name__=="__main__":
    main()
