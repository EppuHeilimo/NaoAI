from __future__ import division
from naoqi import ALProxy
from naoqi import ALModule
from naoqi import ALBroker
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
import wave
import pyaudio
from scipy.io import wavfile
from scipy import fftpack
import matplotlib.pyplot as plt
import struct, math



WordRec = None
AudioRemote = None


class AudioRemoteModule(ALModule):
    """ Mandatory doc """
    nao = None
    name = ""
    py_audio = None
    stream = None
    data = np.empty([1])
    nb_channels = 1
    nb_samples = 2

    def __init__(self, name, nao_ref):
        ALModule.__init__(self, name)
        self.name = name
        self.nao = nao_ref
        self.py_audio = pyaudio.PyAudio()
        nao.connect_remote_audio()

    def start(self):
        nao.start_audio_stream(self.name)

    def stop(self):
        nao.stop_audio_stream(self.name)

    def processRemote(self, nb_channels, nb_samples, timestamp, buffer):
        """ Mandatory doc """
        unfiltered_buffer = np.fromstring(buffer, dtype=np.int16)
        unfiltered_buffer = np.reshape(unfiltered_buffer, (nb_channels, nb_samples), 'F')
        self.data = np.append(self.data, unfiltered_buffer)
        self.nb_samples = nb_samples
        self.nb_channels = nb_channels





class WordRecModule(ALModule):
    """ Mandatory doc """
    nao = None
    name = ""

    def __init__(self, name, nao_ref):
        ALModule.__init__(self, name)
        self.nao = nao_ref
        self.name = name
        nao.memory.subscribeToEvent("WordRecognized", "WordRec", "onWordRecognized")

    def onWordRecognized(self, *_args):
        """ Mandatory doc """
        nao.asr.pause(True)
        data = nao.memory.getData("WordRecognized")
        print data
        nao.asr.pause(False)

    def stop(self):
        nao.memory.unsubscribeToEvent("WordRecognized", "WordRec")

    def start(self):
        nao.memory.subscribeToEvent("WordRecognized", "WordRec", "onWordRecognized")


class Nao:
    IP = "192.168.55.145"
    PORT = 9559
    tts = None
    cam = None
    asr = None
    memory = None
    audio = None
    resolution = vision_definitions.kQVGA
    color_space = vision_definitions.kYUVColorSpace
    fps = 30
    cam_id = ""
    asr_id = ""
    broker = None
    audio_file = None
    sample_rate = 48000

    def __init__(self, ip_address="192.168.55.145", port=9559):
        self.IP = ip_address
        self.PORT = port
        # self.cam.setParam(vision_definitions.kCameraSelectID, vision_definitions.kTopCamera)
        # self.cam.setActiveCamera("ALVideoDevice", 1)

    def connect(self):
        self.broker = ALBroker("myBroker",
                            "0.0.0.0",  # listen to anyone
                            0,  # find a free port and use it
                            self.IP,  # parent broker IP
                            self.PORT)  # parent broker port
        self.memory = ALProxy("ALMemory", self.IP, self.PORT)
        self.tts = ALProxy("ALTextToSpeech", self.IP, self.PORT)
        self.cam = ALProxy("ALVideoDevice", self.IP, self.PORT)

    def speech_rec_connect(self):
        self.asr = ALProxy("ALSpeechRecognition", self.IP, self.PORT)
        self.asr.pause(True)
        vocabulary = ["nao", "who am i", "yes"]
        self.asr.setVocabulary(vocabulary, False)
        self.asr.pause(False)

    def change_camera_parameters(self, resolution, color, fps):
        self.resolution = resolution
        self.color_space = color
        self.fps = fps

    def say(self, text):
        self.tts.say(text)

    def connect_remote_audio(self):
        self.audio = ALProxy("ALAudioDevice", self.IP, self.PORT)

    def stop_audio_stream(self, name):
        self.audio.unsubscribe(name)
        #data = []
        #[data.append(np.fromstring(AudioRemote.data[x], dtype=np.int16)) for x in range(len(AudioRemote.data))]
        #np_data = np.array(data)
        #np_data = np_data.flatten()
        filter_size = 3000
        data = np.array(map(lambda x: 0 if filter_size > x > -filter_size
                            else Utility.reduce(x, filter_size), AudioRemote.data))

        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(data)
        plt.show()
        wavef = wave.open('test.wav', 'w')
        wavef.setnchannels(AudioRemote.nb_channels)  # mono
        print AudioRemote.nb_samples
        wavef.setsampwidth(1)
        wavef.setframerate(self.sample_rate)
        for i in range(int(len(data))):
            temp_data = struct.pack('<h', data[i])
            wavef.writeframesraw(temp_data)

        wavef.writeframes('')
        wavef.close()

        #wavfile.write('test.wav', self.sample_rate, data)
        #self.audio_file.close()



    def start_audio_stream(self, name):
        self.audio.setClientPreferences(name, self.sample_rate, 1, 0)
        self.audio.subscribe(name)


    def get_image(self):
        #t0 = time.time()
        self.cam_id = self.cam.subscribe("python_GVM", self.resolution, self.color_space, self.fps)
        image = self.cam.getImageRemote(self.cam_id)
        self.cam.unsubscribe(self.cam_id)
        #print "Image retrieve time: ", time.time() - t0
        return image

    def start_video(self):
        self.cam_id = self.cam.subscribe("python_GVM", self.resolution, self.color_space, self.fps)

    def start_sr(self):
        self.asr_id = self.asr.subscribe("python_GVM")

    def stop_sr(self):
        self.asr.unsubscribe(self.asr_id)

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
        #print "Image retrieve time: ", time.time() - t0
        return frame

    def set_parameter(self, param_id, value):
        self.tts.setParam(param_id, value)


class Utility:

    @staticmethod
    def reduce(x, filter_size):
        if x > 0:
            x = x - filter_size
        else:
            x = x + filter_size
        return x

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
        return bgr_img


if __name__ == '__main__':
    nao = Nao(ip_address="192.168.55.149")

    # ssd_mobilenet_v1_coco_11_06_2017
    # faster_rcnn_resnet101_coco_2017_11_08
    # face_ssd_mobilenet_v1
    model = tfapi.Model(model_name='face_ssd_mobilenet_v1')
    try:
        nao.connect()
        nao.speech_rec_connect()
        nao.stop_video()
        global WordRec
        #global AudioRemote
        WordRec = WordRecModule("WordRec", nao)
        #AudioRemote = AudioRemoteModule("AudioRemote", nao)
        model.load_frozen_model()
        model.load_label_map()
        nao.change_camera_parameters(vision_definitions.kVGA, 9, 1)
        nao.start_video()
        model.start_session()
        #AudioRemote.start()
        while True:
                image = nao.get_frame()
                image = Utility.nao_YUV2BGR(image)
                if image is not None:
                    image = model.predict(image_np=image)
                    Utility.display_image_cv2(image)
                else:
                    nao.say("Piip poop")

    except Exception as e:
        traceback.print_exc()
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

        print('Stopping remote audio module.')
        #AudioRemote.stop()
        print('Stopping video')
        nao.stop_video()
        print('Stopping speech recognition')
        #nao.stop_sr()
        WordRec.stop()
        print('Stopping broker.')
        nao.broker.shutdown()


