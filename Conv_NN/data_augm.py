import numpy as np 

def left_to_right(x):
    y = np.copy(x)
    y = y[:,:,::-1]
    return y 

def mult_by_scala(x):
    scala = np.random.random_sample() * 0.1
    y = np.copy(x)
    y = np.multiply(y, 0.95 + scala)
    return y
