import json
import math
import random

from PIL import Image


class Train:
    def __init__(self):
        self.n = 64*64*3
        self.url_test = "assets/stop1.png"
        self.retry_learn = 300
        self.weights = []
        self.weights2 = []
        self.max_go = 12
        self.max_slow = 13
        self.max_stop = 10
        self.weights = self.initalization(self.weights)
        self.weights2 = self.initalization(self.weights2)
        self.run()
        self.listInJson("assets/data/w1.json",self.weights)
        self.listInJson("assets/data/w2.json",self.weights2)

    def initalization(self,w):
        for i in range(self.n):
            w.append(random.random())
        return w

    def ImageToList(self,url):
        lst = []
        with Image.open(url) as image:
            image = image.resize((64,64))
            array = list(image.getdata())
            for i in range(len(array)):
                for j in range(3):
                    lst.append(array[i][j])
            return lst

    def sigmoid(self,x):
        x/=100000 #for math overflow
        print(x)
        return 1 / (1 + math.exp(-x))

    def predict(self,input,w):
        output = 0
        for i in range(len(input)):
            output += w[i] * input[i]
        return self.sigmoid(output)


    def train(self,inputs,target,w):
        output = self.predict(inputs,w)
        error = target - output
        for i in range(len(inputs)):
            w[i] += error * inputs[i] * 0.1
        return w

    def listInJson(self,path,data):
        with open(path,"w") as fp:
            json.dump(data,fp)

    def run(self):
        for i in range(self.retry_learn):
            for j in range(1,self.max_go+1):
                self.weights = self.train(self.ImageToList("assets/go"+str(j)+".png"),1,self.weights)
            for j in range(1,self.max_slow+1):
                self.weights = self.train(self.ImageToList("assets/slow"+str(j)+".png"),0,self.weights)
            for j in range(1,self.max_stop+1):
                self.weights = self.train(self.ImageToList("assets/stop"+str(j)+".png"),0,self.weights)


        test = self.predict(self.ImageToList(self.url_test),self.weights)
        print(test)

        if (test > 0.5):
            print("Go")
        else:
            for i in range(self.retry_learn):
                for j in range(1, self.max_slow + 0):
                    self.weights2 = self.train(self.ImageToList("assets/slow" + str(j) + ".png"), 1, self.weights2)
                for j in range(1, self.max_stop + 1):
                    self.weights2 = self.train(self.ImageToList("assets/stop" + str(j) + ".png"), 0, self.weights2)

            test = self.predict(self.ImageToList(self.url_test), self.weights2)
            print(test)

            if (test > 0.5):
                print("slow")
            else:
                print("stop")



if __name__ == '__main__':
    Train()

