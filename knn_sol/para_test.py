import numpy as np
import pickle as pk
import cv2

def load_train_dicts(name):
    file1 = open( name, "rb")
    dict1 = pk.load(file1, encoding="bytes")
    return dict1

def show_by_cv2(img1, name, enlarge=1):
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

def show_firstN_pics(num, batch_no, train_dicts):
    for i in range(num + 1):
        imgarr = train_dicts[batch_no][b"data"][i]
        show_by_cv2(imgarr, "pic"+str(i))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_img_with(img1, res_data, vote, model_datas, model_labels):
    for i in range(len(model_datas)):
        a = np.subtract(img1, model_datas[i])  #......L1 distance
        a = np.absolute(a)
        a = np.sum(a, axis=1)
        # a = np.subtract(img1, model_datas[i])  #......L2 distance
        # a = np.square(a)
        # a = np.sum(a, axis=1)
        # a = np.sqrt(a)
        res_data[i,:] = a
    k = 9                                #......(in "KNN")K = k
    vote[:] = 0
    sort_res_data = np.argsort(res_data, kind="heapsort")
    for i in range(k):
        ind = sort_res_data[0, i]
        n = model_labels[ind // 10000][ind % 10000]
        vote[n] += 1
    return np.argmax(vote)

def main():
    train_dicts = []
    for i in range(1, 6):
        train_dicts.append(load_train_dicts("../data/data_batch_" + str(i)))
    # show_firstN_pics(10, 3, train_dicts)
    train_datas = []
    for train_dict in train_dicts:
        train_datas.append(train_dict[b"data"])
    train_labels = []
    for train_dict in train_dicts:
        train_labels.append(train_dict[b"labels"])
    Count_all = 0
    Count_yes = 0
    for i in range(5):
        valid_datas = train_datas[i]
        valid_labels = train_labels[i]
        model_datas = []
        model_labels = []
        res_data = np.zeros((4,10000))
        vote = np.zeros((10), dtype=np.int8)
        for j in range(5):
            if not(j == i):
                model_datas.append(train_datas[j])
                model_labels.append(train_labels[j])
        # print(type(model_datas[0]))
        # print(type(model_labels[0]))
        # print(model_datas[0].shape)
        # print(len(model_labels[0]))
        for j in range(10000):
            Count_all += 1
            predict = predict_img_with(valid_datas[j], res_data, vote, model_datas, model_labels)
            if predict == valid_labels[j]:
                Count_yes += 1
            if (j % 10 == 9):
                print("have finished {0}, accuracy is {1}"
                    .format(i * 10000 + j + 1, Count_yes / Count_all))


if __name__=="__main__":
    main()
