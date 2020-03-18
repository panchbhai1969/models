###############################################
print("Importing Libraries.......")
import tensorflow as tf
import numpy as np
import json
import csv
from sklearn.metrics import classification_report
from object_detection.utils import label_map_util
print("Libraries Imported....")
###############################################
flags=tf.app.flags
flags.DEFINE_string('csv_path','','Path to csv of format analysed_json_path,actual_json_paths')
flags.DEFINE_string('stg_label_path','','Path to stage level label map')
flags.DEFINE_float('iou_thres',0.0,'Threshold value for IOU')
FLAGS=flags.FLAGS
###############################################
def calc_area(xmin,xmax,ymin,ymax):
  return (xmax-xmin)*(ymax-ymin)

def calc_iou(box_1,box_2):
  """
  Calculated iou between two boxes
  """
  xmin_1,xmax_1,ymin_1,y_max1=box_1
  xmin_2,xmax_2,ymin_2,y_max2=box_2
  area_1=calc_area(float(xmin_1),float(xmax_1),float(ymin_1),float(y_max1))
  area_2=calc_area(float(xmin_2),float(xmax_2),float(ymin_2),float(y_max2))
  intersection_xmin=max(xmin_1,xmin_2)
  intersection_xmax=min(xmax_1,xmax_2)
  intersection_ymin=max(ymin_1,ymin_2)
  intersection_ymax=min(y_max1,y_max2)
  area_intersection=calc_area(float(intersection_xmin),float(intersection_xmax),float(intersection_ymin),float(intersection_ymax))
  area_union=area_1+area_2-area_intersection
  return float(area_intersection/area_union)

def get_class_list(index_map):
  class_list=[]
  for index,index_dict in index_map.items():
        class_list.append(index_dict['name'])
  return class_list

def get_label_list(index_map):
  label_list=[]
  for index,index_dict in index_map.items():
        label_list.append(index_dict['id'])
  return label_list

def label_name_to_index(index_map,label_name):
  for index,index_dict in index_map.items():
      if index_dict['name']==label_name:
        return index_dict['id']

def get_predictons(analysed_report,original_report,category_index_stages,iou_thres):
  """
  Returns class list for ground truth and predictions
  """
  analysed_detections=analysed_report["detections"]
  actual_detections=original_report["detections"]
  if analysed_report["name"]!=original_report["name"]:
    raise ValueError('Image name mismatch')
  predictions=[]
  ground_truth=[]
  for detection_1 in analysed_detections:
    box_1=detection_1["xmin"],detection_1["xmax"],detection_1["ymin"],detection_1["ymax"]
    class_1=label_name_to_index(category_index_stages,detection_1["pretemp_diction"])
    if class_1==None:
      raise ValueError("Class name \""+detection_1["pretemp_diction"]+"\" not found in label map")
    for detection_2 in actual_detections:
      box_2=detection_2["xmin"],detection_2["xmax"],detection_2["ymin"],detection_2["ymax"]
      class_2=label_name_to_index(category_index_stages,detection_2["pretemp_diction"])
      if class_2==None:
        raise ValueError("Class name \""+detection_2["pretemp_diction"]+"\" not found in label map")
      if(calc_iou(box_1,box_2)>=iou_thres):
        predictions.append(class_1)
        ground_truth.append(class_2)

  return ground_truth,predictions

def get_classification_report(analysed_json_paths,actual_json_paths,category_index_stages,iou_thres):
  """
  Prints the classification report for all detections above iou_thres
  """
  predictions_all=[]
  ground_truth_all=[]
  num_reports=len(analysed_json_paths)
  for i in range(num_reports):
    with open(analysed_json_paths[i]) as f:
        analysed_report = json.load(f)
    with open(actual_json_paths[i]) as f:
        original_report = json.load(f)
    
    print("Analysing {} report of {} reports".format(i+1,num_reports))
    ground_truth,predictions=get_predictons(analysed_report,original_report,category_index_stages,iou_thres)
    ground_truth_all.extend(ground_truth)
    predictions_all.extend(predictions)
  
  label_list=get_label_list(category_index_stages)
  class_list=get_class_list(category_index_stages)

  print("Analysis for all reports complete")
  print("Printing classification report")
  print(classification_report(y_true=ground_truth_all,y_pred=predictions_all,labels=label_list,target_names=class_list))

def load_csv(CSV_PATH):
  """
  Loads CSV containing paths for analysed jsons and actual jsons
  """
  print("Loading CSV file...")
  analysed_json_paths=[]
  actual_json_paths=[]
  with open(CSV_PATH) as csvfile:
    f_csv = csv.reader(csvfile) 
    headers = next(f_csv) 
    for row in f_csv:
      analysed_json_paths.append(row[0])
      actual_json_paths.append(row[1])

  return analysed_json_paths,actual_json_paths

def main(_):
  CSV_PATH=FLAGS.csv_path
  PATH_TO_STAGE_LABELS=FLAGS.stg_label_path
  IOU_THRES=FLAGS.iou_thres
  category_index_stages=label_map_util.create_category_index_from_labelmap(PATH_TO_STAGE_LABELS, use_display_name=True)
  analysed_json_paths,actual_json_paths=load_csv(CSV_PATH)
  get_classification_report(analysed_json_paths,actual_json_paths,category_index_stages,IOU_THRES)

if __name__ == "__main__":
    tf.app.run()