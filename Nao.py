from naoqi import ALProxy
import vision_definitions
from PIL import Image
import time
import cv2
import numpy as np
import MachineLearning.TFApi as tfapi
import os
import sys

class Nao:
    IP = "192.168.55.145"
    PORT = 9559
    tts = None
    cam = None
    resolution = vision_definitions.kQVGA
    color_space = vision_definitions.kYUVColorSpace
    fps = 30
    cam_id = ""

    def __init__(self, ip_address="192.168.55.145", port=9559):
        self.IP = ip_address
        self.PORT = port
        self.tts = ALProxy("ALTextToSpeech", "192.168.55.145", 9559)
        self.cam = ALProxy("ALVideoDevice", self.IP, self.PORT)

    def change_camera_parameters(self, resolution, color, fps):
        self.resolution = resolution
        self.color_space = color
        self.fps = fps

    def say(self, text):
        self.tts.say(text)

    def get_image(self):
        t0 = time.time()
        self.cam_id = self.cam.subscribe("python_GVM", self.resolution, self.color_space, self.fps)
        image = self.cam.getImageRemote(self.cam_id)
        self.cam.unsubscribe(self.cam_id)
        print "Image retrieve time: ", time.time() - t0
        return image

    def start_video(self):
        self.cam_id = self.cam.subscribe("python_GVM", self.resolution, self.color_space, self.fps)
        #self.cam.setParam(vision_definitions.kCameraSelectID,
        #                          self.cam_id)

    def stop_video(self):
        self.cam.unsubscribe(self.cam_id)

    def get_frame_raw(self):
        t0 = time.time()
        frame = self.cam.getDirectRawImageRemote(self.cam_id)
        print "Image retrieve time: ", time.time() - t0
        return frame

    def get_frame(self):
        t0 = time.time()
        frame = self.cam.getImageRemote(self.cam_id)
        print "Image retrieve time: ", time.time() - t0
        return frame

    def set_parameter(self, parameter_name, value):
        self.tts.setParameter(parameter_name, value)

    def display_image_pillow(self, image):
        image_width = image[0]
        image_height = image[1]
        arr = image[6]
        pillow_image = Image.frombytes("RGB", (image_width, image_height), arr)
        pillow_image.show()

    def display_image_cv2(self, image):
        cv2.imshow("main", image)
        cv2.waitKey(1)

    def convert_image_np(self, image):
        image_width = image[0]
        image_height = image[1]
        values = map(ord, list(image[6]))
        i = 0
        img = np.zeros((image_height, image_width, 3), np.uint8)

        for y in range(0, image_height):
            for x in range(0, image_width):
                img.itemset((y, x, 0), values[i + 0])
                img.itemset((y, x, 1), values[i + 1])
                img.itemset((y, x, 2), values[i + 2])
                i += 3
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



if __name__ == '__main__':
    nao = Nao(ip_address="192.168.55.145")
    try:
        model = tfapi.Model()
        model.load_frozen_model()
        model.load_label_map()
        nao.change_camera_parameters(vision_definitions.kQVGA, 11, 1)
        #nao.say("I'm taking an image.")
        nao.start_video()
        while True:
            image = nao.get_frame()
            image = nao.convert_image_np(image)
            if image is not None:
                pass
                #image = model.predict(image_np=image)
                nao.display_image_cv2(image)
            else:
                nao.say("Piip poop")
                print("shit")
    except:
        nao.stop_video()



