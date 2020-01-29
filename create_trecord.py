#! /usr/bin/env python3

from xml.etree import cElementTree as ElementTree
import glob

import tensorflow as tf
from object_detection.utils import dataset_util

import io
from PIL import Image


def create_tfrecord(data_dict):
    """creating tfrecord  from a dict with all needed data"""

def parse_pascal_xml(fname):
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
        data_dict['xmins'].append(int(obj.find('bndbox').find('xmin').text))
        data_dict['ymins'].append(int(obj.find('bndbox').find('ymin').text))
        data_dict['xmaxs'].append(int(obj.find('bndbox').find('xmax').text))
        data_dict['ymaxs'].append(int(obj.find('bndbox').find('ymax').text))

    return(data_dict)
    

def main(path):
    for fname in glob.iglob(path + '/*.xml'):
        print(fname)
        # create_tfrecord(parse_pascal_xml(fname))
        print(parse_pascal_xml(fname))
        



if __name__ == '__main__':
    main('prepare_data')
