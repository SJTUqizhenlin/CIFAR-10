import tensorflow as tf 
from tensorflow.python import keras
import numpy as np 
import pickle as pk  
print("finish import.")

def load_train_data():
    print("loading training data...")
    img_list = []
    label_list = []
    for i in range(1, 6):
        file1 = open("../data/data_batch_{0}".format(i), "rb")
        dict1 = pk.load(file1, encoding="bytes")
        file1.close()
        x = dict1[b"data"]
        x = np.subtract(np.divide(x, 64), 2)
        y = dict1[b"labels"]
        for j in range(10000):
            x1 = np.reshape(x[j], (3,32,32))
            img_list.append(x1)
            x2 = np.copy(x1)
            x2 = x2[:,:,::-1]
            img_list.append(x2)
            label_list.append(y[j])
            label_list.append(y[j])
    imgs = np.array(img_list)
    labels = np.array(label_list)
    return imgs, labels

imgs, labels = load_train_data()
labels = keras.utils.to_categorical(labels, num_classes=10)

vgg_model = keras.models.Sequential()

vgg_model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu", input_shape=(3,32,32)))  # 32,30,30
vgg_model.add(keras.layers.BatchNormalization(axis=1))
vgg_model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu"))  # 64,28,28
vgg_model.add(keras.layers.MaxPool2D(data_format="channels_first"))
vgg_model.add(keras.layers.Dropout(0.25))

vgg_model.add(keras.layers.BatchNormalization(axis=1))
vgg_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu"))  # 64,12,12
vgg_model.add(keras.layers.BatchNormalization(axis=1))
vgg_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu"))  # 64,10,10
vgg_model.add(keras.layers.MaxPool2D(data_format="channels_first"))
vgg_model.add(keras.layers.Dropout(0.25))

vgg_model.add(keras.layers.BatchNormalization(axis=1))
vgg_model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu"))  # 128,3,3
vgg_model.add(keras.layers.BatchNormalization(axis=1))
vgg_model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="relu"))  # 256,1,1
vgg_model.add(keras.layers.Dropout(0.5))

vgg_model.add(keras.layers.Flatten()) # 256,
vgg_model.add(keras.layers.Dense(units=10, activation="softmax"))

adam = keras.optimizers.Adam(lr=0.0003)
vgg_model.compile(
    optimizer=adam,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

loss_ave, acc_ave = 0.0, 0.0
least_loss = 1e3
while True:
    for i in range(5):
        vgg_model.fit(x=imgs[0:80000], y=labels[0:80000], 
                        batch_size=64, epochs=1)
        res = vgg_model.evaluate(x=imgs[80000:100000], y=labels[80000:100000],
                                batch_size=64)
        loss_ave += res[0]
        acc_ave += res[1]
    loss_ave /= 5
    acc_ave /= 5
    print()
    print("loss:", loss_ave, "acc:", acc_ave)
    if loss_ave < least_loss:
        least_loss = loss_ave
    if loss_ave > least_loss * 1.1:
        break 
    loss_ave, acc_ave = 0.0, 0.0

path = "D:\\python_proj\\CIFAR10\\tf_keras\\vgg_res.krmd"
vgg_model.save(filepath=path, overwrite=True)