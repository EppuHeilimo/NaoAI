
# coding: utf-8

# # Imports
import os
import sys
# Adding parent folder to path so we can import from our own project
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from MachineLearning.object_detection.utils import label_map_util

from MachineLearning.object_detection.utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
# Any model exported using the `export_inference_graph.py`
#  tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here.
#  See the [detection model zoo]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
#  for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

class Model:
    model_name = ""
    model_file = ""
    num_classes = 90
    path_to_ckpt = ""
    path_to_labels = ""
    category_index = None
    detection_graph = None
    sess = None

    # ssd_mobilenet_v1_coco_11_06_2017
    # faster_rcnn_resnet101_coco_2017_11_08
    # ssd_mobilenet_v1_coco_2017_11_08
    def __init__(self, num_classes=90, download_from_tf=True, model_name='ssd_mobilenet_v1_coco_11_06_2017'):
        # What model to download.
        self.model_name = model_name
        self.model_file = self.model_name + '.tar.gz'
        self.num_classes = num_classes
        if download_from_tf:
            # Download Model
            if not os.path.isfile(os.path.join(os.getcwd(), self.model_file)):
                DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
                opener = urllib.request.URLopener()
                opener.retrieve(DOWNLOAD_BASE + self.model_file, self.model_file)
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.path_to_ckpt = self.model_name + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.path_to_labels = os.path.join('MachineLearning/object_detection/data', 'mscoco_label_map.pbtxt')

        tar_file = tarfile.open(self.model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())


    def load_frozen_model(self):
        # ## Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


    def load_label_map(self):
        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        #  we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns
        #  a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def start_session(self):
        self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def close_session(self):
        self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

    def predict(self, image_path="", image_np=None):
        if not image_path == "":
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            min_score_thresh=0.5,
            use_normalized_coordinates=True,
            line_thickness=4)
        return image_np


    def show_np_images(self, images_np):

        plt.figure(figsize=(20, 14))
        plt.imshow(images_np)
        plt.show()






