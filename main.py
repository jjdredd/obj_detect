#! /usr/bin/env python3

# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    plt.figure()
    plt.imsave(image_path + '_det_1.png', image_np)
    # plt.imshow(image_np)




MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
PATH_TO_MODELF = MODEL_NAME + '/saved_model'
PATH_TO_LABELS = ("/home/ragim/source/models/"
                  "research/object_detection/data/mscoco_label_map.pbtxt")
NUM_CLASSES = 90

print ("loading model ", MODEL_NAME)

model = tf.saved_model.load(PATH_TO_MODELF)
model = model.signatures['serving_default']

detection_model = model

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = "/home/ragim/progs/python/obj_detect/test_images/"
TEST_IMAGE_PATHS = [
    PATH_TO_TEST_IMAGES_DIR + 'image1.jpg',
    PATH_TO_TEST_IMAGES_DIR + 'image2.jpg'
]

for image_path in TEST_IMAGE_PATHS:
    print ("infering ", image_path)
    show_inference(detection_model, image_path)
