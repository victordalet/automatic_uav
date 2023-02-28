from djitellopy import Tello
from pynput import keyboard
import cv2
import threading
import sys
import os

class Drone:
    def __init__(self):
        self.name = "UAV"
        self.commands = [
            lambda t: t.takeoff(),
            lambda t: t.movee_up(20)
        ]
        self.drone = Tello()
        self.drone.connect()
        self.count = 0
        self.capture_video()
        self.keyboard()
        self.thread1.join()

    def fly(self):
        self.thread2 = threading.Thread(target=self.cmd, daemon=True)
        self.thread2.start()

    def cmd(self):
        for c in self.commands:
            print(c, c(self.drone))

    def capture_video(self):
        self.thread1 = threading.Thread(target=self.video_loop, daemon=True)
        self.thread1.start()

    def video_loop(self):
        self.drone.streamon()
        while True:
            cv2.imshow(self.name, self.drone.get_frame_read().frame)
            cv2.waitKey(1)

    def directory(self, path, extension):
        list_dir = os.listdir(path)
        for file in list_dir:
            if file.endswith(extension):
                self.count += 1
        return self.count

    def on_press(self, key):
        print(key)
        if key.char == "p":
            src = self.drone.get_frame_read().frame
            cv2.imwrite("assets/"+str(self.directory("./assets", ".png"))+".png", src)
        if key.char == "e":
            sys.exit()

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False

    def keyboard(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


Drone()