import json
import math
import random

from PIL import Image
from tqdm import tqdm


class Train:
    def __init__(self, M):
        self.n = 64 * 64 * 3
        self.M = M
        self.retry_learn = 300
        self.weights = [self.initalization() for i in range(self.M-1)]
        print(self.weights[0] == self.weights[1])
        self.run()
        self.listInJson("assets/data/w5.json", self.weights)

    def initalization(self):
        w = []
        for i in range(self.n):
            w.append(random.random())
        return w

    def ImageToList(self, url):
        lst = []
        with Image.open(url) as image:
            image = image.resize((64, 64))
            array = list(image.getdata())
            for i in range(len(array)):
                for j in range(3):
                    lst.append(array[i][j])
            return lst

    def sigmoid(self, x):
        x /= 1000000  # for math overflow
        return 1 / (1 + math.exp(-x))

    def predict(self, input, w):
        output = 0
        for i in range(len(input)):
            output += w[i] * input[i]
        return self.sigmoid(output)

    def train(self, inputs, target, w):
        output = self.predict(inputs, w)
        error = target - output
        for i in range(len(inputs)):
            w[i] += error * inputs[i] * 0.1
        return w

    def listInJson(self, path, data):
        with open(path, "w") as fp:
            json.dump(data, fp)

    def run(self):
        lst_nb = [[False for j in range(self.M)]for i in range(self.M -1)]
        for h in tqdm(range(self.M-1)):    ###for all liste
            for i in range(self.retry_learn): ### for retry
                for j in range(h,self.M):  #for the current liste
                    nb = 0
                    if j > h:
                        nb = 1
                    lst_nb[h][j] = nb
                    self.weights[h] = self.train(self.ImageToList("assets/peaple/" + str(j + 1) + ".png"), nb,
                                                 self.weights[h])
        print(lst_nb)
        print("fin de l'entrainement")
        for j in tqdm(range(self.M)):
            for i in range(self.M-1):
                p = self.predict(self.ImageToList("assets/peaple/" + str(j + 1) + ".png"), self.weights[i])
                if p < 0.5:
                    print("l'èleve {} est le : {}".format(j, i))
                    break


if __name__ == '__main__':
    Train(34)
