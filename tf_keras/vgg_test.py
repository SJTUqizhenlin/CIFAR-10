import tensorflow as tf 
from tensorflow.python import keras
import numpy as np 
import pickle as pk 

def load_test_data():
    print("loading test data...")
    img_list = []
    label_list = []
    file1 = open("../data/test_batch", "rb")
    dict1 = pk.load(file1, encoding="bytes")
    file1.close()
    x = dict1[b"data"]
    x = np.subtract(np.divide(x, 64), 2)
    y = dict1[b"labels"]
    for j in range(10000):
        x1 = np.reshape(x[j], (3,32,32))
        img_list.append(x1)
        label_list.append(y[j])
    imgs = np.array(img_list)
    labels = np.array(label_list)
    return imgs, labels

imgs, labels = load_test_data()
labels = keras.utils.to_categorical(labels, num_classes=10)
path = "D:\\python_proj\\CIFAR10\\tf_keras\\vgg_res.krmd"
vgg_model = keras.models.load_model(filepath=path)

res =vgg_model.evaluate(x=imgs, y=labels, batch_size=32)
print()
print("accuracy:", res[1])

###
"""
This VGG-like model on cifar-10
gets 82.0% test accuracy.
"""
###