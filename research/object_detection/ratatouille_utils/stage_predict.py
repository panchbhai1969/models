import numpy as np
import os
from tensorflow import keras
import pickle
import cv2
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import model_from_json

def load_resnet():
    resnet_model=ResNet50(include_top=False,weights='imagenet',input_shape=(100,100,3),pooling="avg")
    return resnet_model

def extract_features(image,resnet_model):
    image=cv2.resize(image,(100,100))
    image=np.expand_dims(image,axis=0)
    image_features=resnet_model.predict(image)
    return image_features

def load_NN(NN_PATH):
    weight_path=os.path.join(NN_PATH,"model_weights.h5")
    architecture_path=os.path.join(NN_PATH,"model_architecture.json")
    with open(architecture_path, 'r') as f:
        nn_model = model_from_json(f.read())
    nn_model.load_weights(weight_path)
    return nn_model

def predict_stage(box_features,nn_model):
    prediction_scores=nn_model.predict(box_features)
    prediction_class=np.argmax(prediction_scores,axis=1)
    return prediction_scores[0][prediction_class],prediction_class+1
