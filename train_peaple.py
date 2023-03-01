import json
import numpy as np
from PIL import Image
from numpyencoder import NumpyEncoder


class NeuralNetwork:
    def __init__(self, n, len, learning_rate=0.001):
        self.weights = np.random.randn(n, len) * np.sqrt(2 / n)
        self.bias = np.zeros((n, 1))
        self.n = n
        self.lr = learning_rate

    def train(self, inputs, target):
        inputs = np.array(inputs).reshape(-1, 1)
        target = np.array(target).reshape(-1, 1)

        predicted = self.predict(inputs)

        error = target - predicted

        self.weights += self.lr * error * inputs.T
        self.bias += self.lr * error

    def predict(self, inputs):
        inputs = np.array(inputs).reshape(-1, 1)

        output = np.dot(self.weights, inputs) + self.bias
        predicted = np.argmax(output)

        return predicted

    def get_weights(self):
        return self.weights

    def save_weights(self):
        with open("assets/data/w1i4.json","w") as fp:
            json.dump(list(self.get_weights()),fp,cls=NumpyEncoder)


def ImageToList(url):
    lst = []
    with Image.open(url) as image:
        image = image.resize((64, 64))
        array = list(image.getdata())
        for i in range(len(array)):
            for j in range(3):
                lst.append(array[i][j])
        return lst


def main():
    N = NeuralNetwork(34, 64 * 64 * 3)
    for i in range(10000):
        for j in range(34):
            print(int(i/100))
            N.train(ImageToList("./assets/peaple/" + str(j+1) + ".png"),j)

    N.save_weights()
    for i in range(34):
        print(i,":",N.predict(ImageToList('./assets/peaple/'+str(i+1)+'.png')))


main()

"""
def test():

    T = NeuralNetwork(3,3)

    for i in range(10000):
        T.train([1,1,0],1)
        T.train([1,0,0],1)
        T.train([0,0,0],0)
        T.train([0,1,0],0)
        T.train([1,1,1],2)
        T.train([1,0,1],2)

    print(T.predict([1,1,1]))
    print(T.predict([0,0,0]))


test()
"""
