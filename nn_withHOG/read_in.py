import numpy as np
import pickle as pk

def read_and_deal(train_list):
    print("loading datas from HOG_data...")
    file1 = open("HOG_data.pkl", "rb")
    dict1 = pk.load(file1)
    file1.close()
    datas = dict1["data"]
    labels = dict1["labels"]
    for i in range(50000):
        img = np.reshape(np.subtract(datas[i], 0.281), (2187, 1))
        label = labels[i]
        tp = (img, label)
        train_list.append(tp)
    print("finished loading.")

def main():
    file1 = open("HOG_data.pkl", "rb")
    dict1 = pk.load(file1)
    file1.close()
    datas = dict1["data"]
    print(len(datas))
    print(datas[0].shape)
    mean_sum = 0
    maxall = 0
    for img in datas:
        maxi = np.max(img)
        if maxi > maxall:
            maxall = maxi
        mean_sum += np.divide(np.sum(img), 2187)
    print(mean_sum / 50000)
    print(maxall)

if __name__=="__main__":
    main()