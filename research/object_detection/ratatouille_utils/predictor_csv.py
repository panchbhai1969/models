###############################################
print("Importing Libraries.......")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import json
import csv 
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import stage_predict
sys.path.append("..")
from object_detection.utils import ops as utils_ops
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
print("Libraries Imported....")
###############################################
flags=tf.app.flags
flags.DEFINE_string('model_dir','','Path to directory with frozen graph')
flags.DEFINE_string('nn_dir','','Path to directory with neural network weights. Weights should follow name convention.')
flags.DEFINE_bool('two_stage_classification',False,'Wether to do a single stage classification or two stage classification')
flags.DEFINE_string('roi_label_path','','Path to roi label map')
flags.DEFINE_string('stg_label_path','','Path to stage level label map')
flags.DEFINE_string('out_images_dir','','Path to Directory with analysed images')
flags.DEFINE_string('out_json_dir','','Path to directory where you want to dump the JSON files')
flags.DEFINE_string('csv_path','','Path to csv of format s.no,image_path,report_path')
flags.DEFINE_string('out_csv_dir','','Directory where compare_json.csv will be created. compare_json.csv is of format (analysed_json_path,actual_json_path)')
flags.DEFINE_float('roi_threshold',0.3,'Threshold for roi detections')
flags.DEFINE_float('stage_threshold',0.2,'Threshold for stage wise detections')
flags.DEFINE_bool('save_analysed_images',False,'Wether to save analysed images. Argument out_images_dir required if this is True')
FLAGS=flags.FLAGS

###############################################
def load_csv(CSV_PATH):
  """
  Loads CSV containing image_path and reports_path. Format (s.no,image_path,report_path)
  """
  print("Loading CSV file...")
  TEST_IMAGE_PATHS=[]
  ACTUAL_REPORTS_PATHS=[]
  with open(CSV_PATH) as csvfile:
    f_csv = csv.reader(csvfile) 
    headers = next(f_csv) 
    for row in f_csv:
      TEST_IMAGE_PATHS.append(row[1])
      ACTUAL_REPORTS_PATHS.append(row[2])

  return TEST_IMAGE_PATHS,ACTUAL_REPORTS_PATHS
###############################################
print("Loading detection graph and label maps...........")
MODEL_NAME=FLAGS.model_dir
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME,'frozen_inference_graph.pb')
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
PATH_TO_LABELS=FLAGS.roi_label_path
PATH_TO_STAGE_LABELS=FLAGS.stg_label_path
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
if FLAGS.two_stage_classification==True:
  category_index_stages=label_map_util.create_category_index_from_labelmap(PATH_TO_STAGE_LABELS, use_display_name=True)

TEST_IMAGE_PATHS,ACTUAL_REPORTS_PATHS = load_csv(FLAGS.csv_path)
print("Loading Done..........")
#############HELPER FUNCTIONS###################

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def return_output_dict(image_path,detection_graph):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    return image_np,output_dict

def get_detections(image_path,detection_graph,thres):
    detection_count=1
    detections=[]
    scores=[]
    stages=[]
    image_np,output_dict=return_output_dict(image_path,detection_graph)
    #im_width, im_height = image_np.shape[1],image_np.shape[0]
    for i in range(0,100):
        if output_dict['detection_scores'][i]>=thres:
            detections.append(output_dict['detection_boxes'][i])
            scores.append(output_dict['detection_scores'][i])
            stages.append(output_dict['detection_classes'][i])
    return image_np,scores,stages,detections

def single_image_single_stage_detections(image_path,detection_graph,category_index,thres_stage):
    image_np,scores,stages,detections=get_detections(image_path,detection_graph,thres_stage)
    scores=np.array(scores)
    scores=scores.reshape(-1,)
    stages=np.array(stages)
    stages=stages.reshape(-1,)
    detections=np.array(detections)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,detections,stages,scores,category_index,use_normalized_coordinates=True,min_score_thresh=thres_stage,line_thickness=3)
    return image_np,scores,stages,detections

  
def single_image_two_stage_detections(image_path,detection_graph,resnet_model,nn_model,category_index_stages,thres_roi,thres_stage):
    scores=[]
    stages=[]
    detections=[]
    image_np,_,_,pre_detections=get_detections(image_path,detection_graph,thres_roi)
    im_width, im_height = image_np.shape[1],image_np.shape[0]
    for detection in pre_detections:
        ymin, xmin, ymax, xmax=detection
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),int(ymin * im_height), int(ymax * im_height))
        detection_box=image_np[top:bottom,left:right,:]
        box_features=stage_predict.extract_features(detection_box,resnet_model)
        score,stage=stage_predict.predict_stage(box_features,nn_model)
        if score>=thres_stage:
          detections.append(detection)
          scores.append(score)
          stages.append(stage)
    scores=np.array(scores)
    scores=scores.reshape(-1,)
    stages=np.array(stages)
    stages=stages.reshape(-1,)
    detections=np.array(detections)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,detections,stages,scores,category_index_stages,use_normalized_coordinates=True,min_score_thresh=thres_stage,line_thickness=3)
    return image_np,scores,stages,detections

def index_to_label_name(index_map,label_index):
  for index,index_dict in index_map.items():
      if index_dict['id']==label_index:
        return index_dict['name']

def result_to_json(image_path,analysed_output_dir,scores,stages,detections,index_map,json_dir):
    """
    Function that creates json file for the prediction. Also returns analysed_json_path for furthur use
    """
    out_dict={}
    image_name=image_path.rsplit('/', 1)[-1]
    out_dict["name"]=image_name
    out_dict["number_of_detections"]=detections.shape[0]
    out_dict["author"]='MachineLearning : Logy.AI'
    detection_list=[]
    for i,detection in enumerate(detections):
        detection_dict={}
        ymin, xmin, ymax, xmax=detection
        detection_dict["xmin"]=str(xmin)
        detection_dict["xmax"]=str(xmax)
        detection_dict["ymin"]=str(ymin)
        detection_dict["ymax"]=str(ymax)
        detection_dict["pretemp_diction"]=index_to_label_name(index_map,stages[i])
        detection_dict["probability"]=str(scores[i])
        detection_list.append(detection_dict)
    out_dict["detections"]=detection_list
    out_dict["original_image"]=image_path
    out_dict["analysed_image"]=os.path.join(analysed_output_dir,image_name)
    analysed_json_path=os.path.join(json_dir,image_name.rsplit('.', 1)[0]+'.json')
    with open(analysed_json_path, 'w') as fp:
        json.dump(out_dict, fp)

    return analysed_json_path

#########################FUNCTION FOR SINGLE STAGE CLASSIFICATION####################
def single_stage_class():
  #Loading the resnet model
  print("STARTING SINGLE STAGE CLASSIFICATION....")
  OUT_CSV_DATA=[["analysed_json_path","actual_json_path"]]
  for i,image_path in enumerate(TEST_IMAGE_PATHS):
      image_name=image_path.rsplit('/', 1)[-1]
      analysed_image,scores,stages,detections=single_image_single_stage_detections(image_path,detection_graph,category_index,thres_stage=FLAGS.stage_threshold)
      print(image_name+"-->Number of detections="+str(detections.shape[0])+"------> Done")
      if FLAGS.save_analysed_images:
        output_dir=FLAGS.out_images_dir
        cv2.imwrite(os.path.join(output_dir,image_name),analysed_image)
      analysed_json_path=result_to_json(image_path,FLAGS.out_images_dir,scores,stages,detections,category_index,FLAGS.out_json_dir)
      data=[]
      data.append(analysed_json_path)
      data.append(ACTUAL_REPORTS_PATHS[i])
      OUT_CSV_DATA.append(data)

  print("Detections Done")
  print("Writing compare_json.csv file")
  with open(os.path.join(FLAGS.out_csv_dir,'compare_json.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(OUT_CSV_DATA)

#########################FUNCTION FOR TWO STAGE CLASSIFICATION####################
def two_stage_class():
  #Loading the resnet model
  print("STARTING TWO STAGE CLASSIFICATION....")
  print("Loading the resnet model for feature extraction........")
  resnet_model=stage_predict.load_resnet()
  print("Loaded the resnet model")
  #Loading the NN model
  print("Loading the Neural Network model......")
  NN_PATH=FLAGS.nn_dir
  nn_model=stage_predict.load_NN(NN_PATH)
  print("Loaded the Neural Network model.......")
  OUT_CSV_DATA=[["analysed_json_path","actual_json_path"]]
  print("Starting image analysis...............")
  for i,image_path in enumerate(TEST_IMAGE_PATHS):
      image_name=image_path.rsplit('/', 1)[-1]
      analysed_image,scores,stages,detections=single_image_two_stage_detections(image_path,detection_graph,resnet_model,nn_model,category_index_stages,thres_roi=FLAGS.roi_threshold,thres_stage=FLAGS.stage_threshold)
      print(image_name+"-->Number of detections="+str(detections.shape[0])+"------> Done")
      if FLAGS.save_analysed_images:
        output_dir=FLAGS.out_images_dir
        cv2.imwrite(os.path.join(output_dir,image_name),analysed_image)
      analysed_json_path=result_to_json(image_path,FLAGS.out_images_dir,scores,stages,detections,category_index_stages,FLAGS.out_json_dir)
      data=[]
      data.append(analysed_json_path)
      data.append(ACTUAL_REPORTS_PATHS[i])
      OUT_CSV_DATA.append(data)
  print("Detections Done")
  print("Writing compare_json.csv file")
  with open(os.path.join(FLAGS.out_csv_dir,'compare_json.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(OUT_CSV_DATA)

def main(_):
  if FLAGS.two_stage_classification==True:
    two_stage_class()
  else:
    single_stage_class()
    
if __name__ == "__main__":
      tf.app.run()
#############################################################

