#! /usr/bin/env python3

from xml.etree import cElementTree as ElementTree
import glob

import tensorflow as tf
from object_detection.utils import dataset_util

import os
import io
from PIL import Image


annotations_dir = 'prepare_data'
image_dir = 'prepare_data'
output_dir = 'prepare_data'
output_path = os.path.join(output_dir, 'train.tfrecord')

def create_tfrecord(data_dict):
    """creating tfrecord  from a dict with all needed data"""
    with tf.io.gfile.GFile(os.path.join(image_dir, data_dict['image']), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = data_dict['image'].encode('utf8')
    image_format = b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    for i in range(len(data_dict['names'])):
        xmins.append(data_dict['xmins'][i] / width)
        xmaxs.append(data_dict['xmaxs'][i] / width)
        ymins.append(data_dict['ymins'][i] / height)
        ymaxs.append(data_dict['ymaxs'][i] / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(data_dict['names']),
        'image/object/class/label': dataset_util.int64_list_feature(data_dict['labels']),
    }))
    return tf_example

def parse_pascal_xml(fname, labelmap):
    """parse xml annotation in pascal format"""

    data_dict = {}
    root = ElementTree.parse(fname).getroot()
    data_dict['image'] =  root.find('filename').text
    data_dict['width'] = int(root.find('size').find('width').text)
    data_dict['height'] = int(root.find('size').find('height').text)
    data_dict['depth'] = int(root.find('size').find('depth').text)

    data_dict['names'] = []
    data_dict['labels'] = []
    data_dict['xmins'] = []
    data_dict['ymins'] = []
    data_dict['xmaxs'] = []
    data_dict['ymaxs'] = []
    for obj in root.findall('object'):
        data_dict['names'].append(obj.find('name').text.encode('utf8'))
        data_dict['labels'].append(labelmap[obj.find('name').text])
        data_dict['xmins'].append(int(obj.find('bndbox').find('xmin').text))
        data_dict['ymins'].append(int(obj.find('bndbox').find('ymin').text))
        data_dict['xmaxs'].append(int(obj.find('bndbox').find('xmax').text))
        data_dict['ymaxs'].append(int(obj.find('bndbox').find('ymax').text))

    return(data_dict)
    

def main():
    labelmap = {'mexpr' : 1}
    writer = tf.io.TFRecordWriter(output_path)
    for fname in glob.iglob(os.path.join(annotations_dir, '*.xml')):
        print("adding", fname)
        tfr = create_tfrecord(parse_pascal_xml(fname, labelmap))
        writer.write(tfr.SerializeToString())

    writer.close()
        



if __name__ == '__main__':
    main()
