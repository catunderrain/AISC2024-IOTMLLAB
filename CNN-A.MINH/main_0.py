import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return np.e**x
    return 1/(1+np.e**(-x))


def dis_softmax():
    xs = [i/10 for i in range(-160,160,1)]
    ys = [sigmoid(x) for x in xs]
    
    plt.plot(xs,ys)
    plt.scatter(5,sigmoid(5),c='r')
    plt.show()
    
    return None


if __name__ == '__main__':
    dis_softmax()
    
    pass