import numpy as np
import pickle as pk

def hog2d(mtx):
    grad_opr = np.array([1, 0, -1])
    hog_row = np.zeros((30, 30))
    for i in range(1,31):
        hog_row[i-1,:] = np.convolve(mtx[i,:], grad_opr, "valid")
    mtxT = np.transpose(mtx)
    hog_col = np.zeros((30, 30))
    for i in range(1,31):
        hog_col[i-1,:] = np.convolve(mtxT[i,:], grad_opr, "valid")
    hog_col = np.transpose(hog_col)
    np.add(hog_row, 0.000001, out=hog_row)
    np.add(hog_col, 0.000001, out=hog_col)
    L = np.sqrt(np.add(np.square(hog_row), np.square(hog_col)))
    A = np.arctan(np.divide(hog_col, hog_row))
    return L, A

def add_f(L, A, f_list):
    for i in range(0, 27, 3):
        for j in range(0, 27, 3):
            listij = [0] * 9
            for x in range(6):
                for y in range(6):
                    pos = (A[i + x, j + y] * 9. / np.pi) + 4.5
                    pos = int(pos)
                    if pos < 0 or pos > 8:
                        print("Error: angle error")
                    listij[pos] += L[i + x, j + y]
            for i in range(9):
                listij[i] /= 722
            f_list.extend(listij)

def get_HOGmat(img):
    x = np.reshape(img, (3, 32, 32))
    r, g, b = (x[0], x[1], x[2])
    r_hogL, r_hogA = hog2d(r)
    g_hogL, g_hogA = hog2d(g)
    b_hogL, b_hogA = hog2d(b)
    f_list = []
    add_f(r_hogL, r_hogA, f_list)
    add_f(g_hogL, g_hogA, f_list)
    add_f(b_hogL, b_hogA, f_list)
    return np.array(f_list)

def main():
    data_list = []
    label_list = []
    for i in range(1, 6):
        file1 = open("../data/data_batch_{0}".format(i), "rb")
        dict1 = pk.load(file1, encoding="bytes")
        file1.close()
        datas = dict1[b"data"]
        labels = dict1[b"labels"]
        for j in range(10000):
            data_list.append(get_HOGmat(datas[j]))
            label_list.append(labels[j])
            if (j + 1) % 1000 == 0:
                print("now {0} imgs has been dealt".format((i - 1)*10000+j+1))
    file2 = open("HOG_data.pkl", "wb")
    pk.dump({"data": data_list, "labels": label_list}, file2)
    file2.close()

if __name__=="__main__":
    main()