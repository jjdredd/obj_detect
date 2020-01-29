#! /usr/bin/env python3

from xml.etree import cElementTree as ElementTree
import glob

import tensorflow as tf
from object_detection.utils import dataset_util

import io
from PIL import Image


def create_tfrecord(data_dict):
    """creating tfrecord  from a dict with all needed data"""
    with tf.io.gfile.GFile(data_dict['image'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = data_dict['image']
    image_format = b'png'

    for i in len(data_dict['names']):
        xmins.append(data_dict['xmins'][i] / width)
        xmaxs.append(data_dict['xmaxs'][i] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

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
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def parse_pascal_xml(fname, labelmap):
    """parse xml annotation in pascal format"""

    data_dict = {}
    root = ElementTree.parse(fname).getroot()
    data_dict['image'] =  root.find('path').text
    data_dict['width'] = int(root.find('size').find('width').text)
    data_dict['height'] = int(root.find('size').find('height').text)
    data_dict['depth'] = int(root.find('size').find('depth').text)

    data_dict['names'] = []
    data_dict['xmins'] = []
    data_dict['ymins'] = []
    data_dict['xmaxs'] = []
    data_dict['ymaxs'] = []
    for obj in root.findall('object'):
        data_dict['names'].append(obj.find('name').text)
        data_dict['labels'].append(labelmap[obj.find('name').text])
        data_dict['xmins'].append(int(obj.find('bndbox').find('xmin').text))
        data_dict['ymins'].append(int(obj.find('bndbox').find('ymin').text))
        data_dict['xmaxs'].append(int(obj.find('bndbox').find('xmax').text))
        data_dict['ymaxs'].append(int(obj.find('bndbox').find('ymax').text))

    return(data_dict)
    

def main(path, labelmap):
    labelmap = {'mexpr' : 1}
    for fname in glob.iglob(path + '/*.xml'):
        print(fname)
        print(parse_pascal_xml(fname, labelmap))
        



if __name__ == '__main__':
    main('prepare_data')
