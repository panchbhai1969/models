import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pickle
import cv2
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from object_detection.utils import label_map_util
import json
#import sys
#log = open("status.log", "a")
#sys.stdout = log
#######################################
flags=tf.app.flags
flags.DEFINE_string('images_dir','','Path to directory containing images')
flags.DEFINE_string('reports_dir','','Path to directory containing reports')
flags.DEFINE_string('output_dir','','Path to directory where you want to save NN model weights')
flags.DEFINE_string('label_map_path','','Path to label map')
flags.DEFINE_integer('save_weights','1','Bool for saving weights')
FLAGS=flags.FLAGS
######################################
def load_data(IMAGES_PATH,REPORTS_PATH):
    print("Loading data")
    reports=os.listdir(REPORTS_PATH)
    labels=[]
    images=[]
    for report in reports:
        JASON_PATH=REPORTS_PATH+'/'+report
        with open(JASON_PATH) as json_file:
            data = json.load(json_file)
        image_name=data['name']
        image_path=IMAGES_PATH+'/'+image_name
        img=cv2.imread(image_path,1)
        detections=data['detections']
        for detection in detections:
            label=detection['pretemp_diction']
            labels.append(label)
            im_width, im_height = img.shape[1],img.shape[0]
            xmin=float(detection["xmin"])
            xmax=float(detection["xmax"])
            ymin=float(detection["ymin"])
            ymax=float(detection["ymax"])
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width)
                                          ,int(ymin * im_height), int(ymax * im_height))
            detection_box=img[top:bottom,left:right,:]
            images.append(detection_box)
    set_labels=set(labels)
    labels=np.array(labels)
    print("Data loaded successfully. Data report :-")
    print("Images found : {}".format(len(images)))
    return images,labels

def image_preprocessing(images):                                            #ONLY RESIZING THE IMAGE AS OF NOW
    print("Processing images")
    for i,image in enumerate(images):
        images[i]=cv2.resize(image,(100,100))
    images=np.array(images)
    print("Images processed, Image shape : {}".format(images.shape))
    return images

def extract_features(images):
    print("Loading RESNET for feature extraction....")
    model=ResNet50(include_top=False,weights='imagenet',input_shape=(100,100,3),pooling="avg")
    print("Extracting features using RESNET")
    images=model.predict(images)
    print("Feature extraction successfull,final shape : {}".format(images.shape))
    return images

def export_label_map(LABELMAP_PATH):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    label_list=[]
    for category in label_map_dict.keys():
        label_list.append(category)
    label_list=sorted(label_list)
    label_array=np.array(label_list)
    return label_array

def one_hot_encode(labels,LABELMAP_PATH):
    print("Converting labels to one hot vectors")
    label_array=export_label_map(LABELMAP_PATH)
    num_categories=label_array.shape[0]
    enc=OneHotEncoder(categories=[label_array],dtype=int)
    enc.fit(labels.reshape(-1,1))
    print("Encoder categories:-")
    print(enc.categories_)
    labels=enc.transform(labels.reshape(-1,1)).toarray()
    print(labels.shape)
    print("Labels converted successfully")
    return labels,num_categories

from keras.layers import Flatten,Dense,Input
from keras.models import Model
def create_model(num_categories):
    print("Making model Instance")
    inp=Input(shape=(2048,))
    x=Dense(128,activation="relu")(inp)
    x=Dense(64,activation="relu")(x)
    y=Dense(num_categories,activation="softmax")(x)
    model=Model(inputs=inp,output=y)
    print("Model created successfully")
    return model

def train_nn_model(images,labels,num_categories,test_size=0.1,epochs=50,batch_size=64,save_weights=FLAGS.save_weights):
    print("Making Train/Test split")
    x_train,x_val,y_train,y_val=train_test_split(images,labels,test_size=test_size)
    model=create_model(num_categories)
    print("Starting the training")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val),epochs=epochs,batch_size=batch_size)
    print("Training Completed")

    if save_weights:
        OUTPUT_DIR=FLAGS.output_dir
        print("Saving model weights and architecture")
        weight_name='model_weights'+'.h5'
        architecture_name='model_architecture'+'.json'
        model.save_weights(os.path.join(OUTPUT_DIR,weight_name))
        with open(os.path.join(OUTPUT_DIR,architecture_name), 'w') as f:
            f.write(model.to_json())
        print("Model weights and Architecture saved successfully")

    print("Printing classification report")
    y_pred=model.predict(x_val)
    y_pred=np.argmax(y_pred,axis=1)
    y_true=np.argmax(y_val,axis=1)
    print(classification_report(y_true,y_pred))


def main(_):
    IMAGES_PATH=FLAGS.images_dir
    REPORTS_PATH=FLAGS.reports_dir
    images,labels=load_data(IMAGES_PATH,REPORTS_PATH)
    images=image_preprocessing(images)
    images=extract_features(images)
    labels,num_categories=one_hot_encode(labels,FLAGS.label_map_path)
    train_nn_model(images,labels,num_categories)

if __name__ == '__main__':
  tf.app.run()