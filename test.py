import math
import random

from PIL import Image
from djitellopy import Tello
from pynput import keyboard
import cv2
import threading
import json
import sys

from map import map


class AutoDrone:
    def __init__(self, map, xd, yd, xa, ya):
        self.weight1 = self.get_data("assets/data/w1.json")
        self.weight2 = self.get_data("assets/data/w2.json")
        self.name = "UAV"
        self.commands = [
            lambda t: t.takeoff(),
            lambda t: t.move_up(20)
        ]
        self.drone = Tello()
        self.drone.connect()
        print(self.drone.get_battery())
        self.drone.takeoff()
        self.run_speed = 50
        self.run_slow = 20
        self.speed_turn = 30
        self.map = map
        self.lst_position = []  # for the position of the way
        self.deg = 0  # for check the position of deg
        self.change_deg = [0, 180, 270, 90]
        self.where_is = 0  # for know the position in the lst of way
        self.x_depart = xd
        self.y_depart = yd
        self.x_arrivee = xa
        self.y_arrivee = ya
        self.parcours()
        print(self.lst_position)
        self.video_loop()

    def cmd(self):
        for c in self.commands:
            print(c, c(self.drone))

    def capture_video(self):
        self.thread1 = threading.Thread(target=self.video_loop, daemon=True)
        self.thread1.start()

    def video_loop(self):
        self.drone.streamon()
        # self.drone.takeoff()
        while True:
            picture = self.drone.get_frame_read().frame
            cv2.imshow(self.name, picture)
            cv2.waitKey(1)
            src = cv2.resize(picture, (64, 64))
            try:
                p = self.predict(self.matrixToList(src), self.weight1)
                if p > 0.5:
                    print("go")
                    self.drone.move_forward(50)
                    self.where_is += 2
                    self.verif_way()
                else:
                    p = self.predict(self.matrixToList(src), self.weight2)
                    if p > 0.5:
                        print("slow")
                        self.drone.move_forward(30)
                        self.where_is += 1
                        self.verif_way()
                    else:
                        print("stop")
                        self.drone.rotate_clockwise(40)
                        self.deg += 40

                        if self.deg > 360:
                            self.deg = 0 + (self.deg - 360)
            except:
                print("")

    def parcours(self):
        self.lst_position = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0]

    def verif_way(self):
        if self.where_is % 10:
            if self.where_is > len(self.lst_position):
                self.drone.land()
            else:
                new_deg = self.change_deg[self.lst_position[self.where_is]] - self.deg
                self.deg += new_deg
                self.drone.rotate_clockwise(new_deg)

    def get_data(self, path):
        with open(path) as fp:
            data = json.load(fp)
        return data

    def sigmoid(self, x):
        x /= 100000  # pour eviter math range error
        return 1 / (1 + math.exp(-x))

    def predict(self, input, w):
        output = 0
        for i in range(len(input)):
            output += w[i] * input[i]
        return self.sigmoid(output)

    def ImageToList(self, url):
        lst = []
        with Image.open(url) as image:
            image = image.resize((64, 64))
            array = list(image.getdata())
            for i in range(len(array)):
                for j in range(3):
                    lst.append(array[i][j])
            return lst

    def matrixToList(self, matrix):
        lst = []
        for i in matrix:
            for j in i:
                for k in j:
                    lst.append(k)
        return lst


AutoDrone(map, 4, 14, 3, 5)
