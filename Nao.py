from naoqi import ALProxy
import vision_definitions
from PIL import Image
import time
import cv2
import sys
import numpy as np
import MachineLearning.TFApi as tfapi
import os
import sys
import traceback
import pickle

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

        #self.cam.setParam(vision_definitions.kCameraSelectID, vision_definitions.kTopCamera)
        #self.cam.setActiveCamera("ALVideoDevice", 1)

    def connect(self):
        self.tts = ALProxy("ALTextToSpeech", self.IP, self.PORT)
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

        #self.cam.setParameter(1, self.cam_id)

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
        self.cam.releaseImage(self.cam_id)
        print "Image retrieve time: ", time.time() - t0
        return frame

    def set_parameter(self, param_id, value):
        self.tts.setParam(param_id, value)


class Utility:

    @staticmethod
    def display_image_pillow(img):
        image_width = img[0]
        image_height = img[1]
        arr = img[6]
        pillow_image = Image.frombytes("RGB", (image_width, image_height), arr)
        pillow_image.show()

    @staticmethod
    def display_image_cv2(img, wait_time=1):
        cv2.imshow("main", img)
        cv2.waitKey(wait_time)

    @staticmethod
    def convert_image_np(img):
        image_width = img[0]
        image_height = img[1]
        values = map(ord, list(img[6]))
        print(values)
        i = 0
        img_np = np.zeros((image_height, image_width, 3), np.uint8)

        for y in range(0, image_height):
            for x in range(0, image_width):
                img_np.itemset((y, x, 0), values[i + 0])
                img_np.itemset((y, x, 1), values[i + 1])
                img_np.itemset((y, x, 2), values[i + 2])
                i += 3
        return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Takes in nao robot's image object and returns BGR np array
    @staticmethod
    def nao_YUV2BGR(img_object):
        t0 = time.time()
        image_height = img_object[1]
        image_width = img_object[0]
        frame_channels = img_object[2]
        image_raw = img_object[6]
        # Take every x data from raw image and store it in the right component
        U = image_raw[0::4]
        Y1 = image_raw[1::4]
        V = image_raw[2::4]
        Y2 = image_raw[3::4]

        UV = np.empty((image_height * image_width), dtype=np.uint8)
        YY = np.empty((image_height * image_width), dtype=np.uint8)

        UV[0::2] = np.fromstring(U, dtype=np.uint8)
        UV[1::2] = np.fromstring(V, dtype=np.uint8)
        YY[0::2] = np.fromstring(Y1, dtype=np.uint8)
        YY[1::2] = np.fromstring(Y2, dtype=np.uint8)

        UV = UV.reshape((image_height, image_width))
        YY = YY.reshape((image_height, image_width))

        combined_yuv_image = cv2.merge([UV, YY])
        # YUY2 isn't the right format, but it works. Nao camera sends UYVY format (YUV422).
        bgr_img = cv2.cvtColor(combined_yuv_image, cv2.COLOR_YUV2BGR_YUY2)
        print("Image decode and conversion time: ", time.time() - t0)
        return bgr_img


if __name__ == '__main__':
    nao = Nao(ip_address="192.168.55.149")
    model = tfapi.Model()
    try:
        nao.connect()
        model.load_frozen_model()
        model.load_label_map()
        nao.change_camera_parameters(vision_definitions.kQVGA, 9, 1)
        nao.start_video()
        i = 0
        while True:
            image = nao.get_frame()
            with open("./test/raw_frame'{}'.dump".format(i), 'wb') as f:
                pickle.dump(image, f)
            image = Utility.nao_YUV2BGR(image)
            if image is not None:
                pass
                image = model.predict(image_np=image)
                t0 = time.time()
                Utility.display_image_cv2(image)
                i += 1
                print "Image retrieve time 3: ", time.time() - t0
            else:
                nao.say("Piip poop")
    except:
        print ("No connection to Nao, running image data from ./test/ folder.")
        model = tfapi.Model()
        model.load_frozen_model()
        model.load_label_map()
        i = 0
        dir_len = len(os.listdir("./test/"))
        model.start_session()
        while i < dir_len:
            with open("./test/raw_frame'{}'.dump".format(i), 'rb') as f:
                image = pickle.load(f)
            image = Utility.nao_YUV2BGR(image)
            if image is not None:
                t0 = time.time()
                image = model.predict(image_np=image)
                print("Prediction 1 time: ", time.time() - t0)
                Utility.display_image_cv2(image, 1)
                i += 1
            else:
                print("Image is None, skipping...")
    finally:
        traceback.print_stack()
        model.close_session()
        nao.stop_video()

