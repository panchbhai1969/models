# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import json
import csv

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
flags = tf.app.flags
# flags.DEFINE_string('reports_dir', '', 'Root directory to reports.')
# flags.DEFINE_string('images_dir', '', 'Root directory to images.')
flags.DEFINE_string('dataset_path', '', 'Path to dataset csv')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('num_shards', 2, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       img_path):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding json fields for a single image 
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.


  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  # filename = data['name'].split('-')[1]

  # img_path = os.path.join(image_subdirectory, filename)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  wi, hi = image.size
  # if image.format != 'PNG':
  #   raise ValueError('Image format not PNG')
#   key = hashlib.sha256(encoded_png).hexdigest()


  height = hi
  width = wi
  filename = img_path.split('/')[-1]
  encoded_image_data = encoded_png
  image_format = image.format.encode('utf8')
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box  (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  for bounding_box in data['detections']:
      # xmins.append(float(bounding_box['bounding_box']['minimum']['c']) / width)
      # ymins.append(float(bounding_box['bounding_box']['minimum']['r']) / height)
      # xmaxs.append(float(bounding_box['bounding_box']['maximum']['c']) / width)
      # ymaxs.append(float(bounding_box['bounding_box']['maximum']['r']) / height)
      xmins.append(float(bounding_box['xmin']))
      ymins.append(float(bounding_box['ymin']))
      xmaxs.append(float(bounding_box['xmax']))
      ymaxs.append(float(bounding_box['ymax']))
      class_name =bounding_box['pretemp_diction'].encode('utf8')
      classes_text.append(class_name)
      classes.append(label_map_dict[bounding_box['pretemp_diction']])

  if len(xmaxs)==0:
    return None
  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
  }
  

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))

      json_report = ''
      with open(example[2]) as f:
        json_report = json.load(f)
      img_path = example[1]

      try:
        tf_example = dict_to_tf_example(
            json_report,
            label_map_dict,
            img_path)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', example['image']['pathname'])


def dataset_csv_to_example_list(dataset_path):
  examples_list = []
  i=0
  with open(dataset_path) as f:
    csv_f = csv.reader(f)
    
    for row in csv_f:
      if i==0: # skipping first row
        i = 1 
        continue
      examples_list.append(row)
  return examples_list


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  # reports_dir = FLAGS.reports_dir
  # images_dir = FLAGS.images_dir
  dataset_path = FLAGS.dataset_path
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from CBC dataset.')
  

  examples_list = dataset_csv_to_example_list(dataset_path)
  


  # for report in os.listdir(reports_dir):
  #     with open(os.path.join(reports_dir, report)) as json_file:
  #         example = json.load(json_file)
  #     examples_list.append(example)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  # random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)

  print("Length of examples_list : ", len(examples_list))

  num_examples = len(examples_list)
  num_train = int(0.95 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]

  print("No. of Train Examples : ", len(train_examples))
  print("No. of Validation Examples : ", len(val_examples))

  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
  images_dir = ''

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      images_dir,
      train_examples)
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      images_dir,
      val_examples)


if __name__ == '__main__':
  tf.app.run()