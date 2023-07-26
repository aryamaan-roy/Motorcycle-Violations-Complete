import argparse
import math
import random
import subprocess
import sys
import time


import numpy as np
import torch
import torch.nn as nn
import yaml
import torchvision

from models.yolo import Model
from utils.general import intersect_dicts
from utils.general import non_max_suppression

import cv2
import os
import pickle
import matplotlib.pyplot as plt
from pprint import pprint

def get_model(hyp, weights, cfg, device):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict


    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=2, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor']
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load

    model.hyp = hyp  # attach hyperparameters to model
    model.eval()

    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
hyps = "./data/hyps/hyp.scratch-low.yaml"
rm_weights = "../../../Weights/yolov5/rider_motor_cl/weights/last.pt"
hnh_weights = "../../../Weights/yolov5/h_nh_run/weights/last.pt"
cfg = "models/yolov5m.yaml"

rider_motor_model = get_model(hyps, rm_weights, cfg, device)
helmet_no_helmet_model = get_model(hyps, hnh_weights, cfg, device)
trapez_model = pickle.load(open('../../../Weights/Trapezium_Prediction_Weights.pickle', 'rb'))

frames = []
frames2 = []
gt_boxes = []
gt_classes = []
# Storing all the Validation Images 
# Images are all files with .jpg in data/validation_data_234Images
path = "../../../../data/testing_data_310Images/final_test_set/"
for file in os.listdir(path):
    if file.endswith('.jpg') or file.endswith('.jpeg') :
        # read the image and storing 2 copies of it for annotation purposes 
        img1 = cv2.imread(os.path.join(path, file))
        img2 = img1.copy()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        frames.append(img1)
        frames2.append(img2)
        file_name = path + "../default/" + file.split('.')[0] + ".txt"
        f = open(file_name, "r")
        lines = f.readlines()
        boxes = []
        labels = []
        for line in lines:
            list1 = []
            words = line.split()
            for i, word in enumerate(words[1:]):
                if(i % 2 == 0):
                    list1.append(float(word) * img1.shape[1])
                else:
                    list1.append(float(word) * img1.shape[0])
            boxes.append(list1)
            labels.append(int(words[0]))
        gt_boxes.append(boxes)
        gt_classes.append(labels)

# # limit to first 10 images
frames = frames[:]
frames2 = frames2[:]
gt_boxes = gt_boxes[:]
gt_classes = gt_classes[:]


input_size = 640
all_batch_data = []
for i, frame in enumerate(frames):
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    image_data = torch.from_numpy(image_data)
    image_data = image_data.permute((0,3,1,2))
    all_batch_data.append(image_data)

all_rider_motor_bbox = []
count = 0

for batch_data in all_batch_data:
    rider_motor_bbox = rider_motor_model(batch_data)
    print(count)
    count += 1
    all_rider_motor_bbox.append(rider_motor_bbox)

iou = 0.45
score = 0.50

allowed_classes = [0,1]

# This stores the rider & motorcycle boxes, scores, classes and number of objects detected for each image in the validation set
all_boxes_R_M = []
all_scores_R_M = []
all_classes_R_M = []
all_num_objects_R_M = []

for rider_motor_bbox in all_rider_motor_bbox:

    preds = non_max_suppression(rider_motor_bbox,
                            score,
                            iou,
                            labels=(),
                            multi_label=True,
                            agnostic=False,
                            max_det=300)

    ind = torchvision.ops.nms(preds[0][:,:4], preds[0][:,4], 0.6)
    preds[0] = preds[0][ind]
    preds = preds[0]
    print(preds)

    num_objects_R_M = preds.shape[0]
    boxes_R_M = preds[:,:4].detach().numpy()
    boxes_R_M = boxes_R_M[:] / input_size
    print(boxes_R_M)
    scores_R_M = preds[:,4].detach().numpy()
    scores_R_M = scores_R_M[:]
    classes_R_M = preds[:,5].detach().numpy()
    classes_R_M = classes_R_M[:]

    deleted_indx = []
    for i in range(num_objects_R_M):
        class_indx = int(classes_R_M[i])
        if class_indx not in allowed_classes:
            deleted_indx.append(i)
    boxes_R_M = np.delete(boxes_R_M, deleted_indx, axis=0)
    scores_R_M = np.delete(scores_R_M, deleted_indx, axis=0)
    classes_R_M = np.delete(classes_R_M, deleted_indx, axis=0)
    num_objects_R_M = len(classes_R_M)
    # print(boxes_R_M)
    # print(scores_R_M)
    # MOTORCYCLE CLASS CHANGED TO 3 FROM 1
    classes_R_M[classes_R_M == 1] = 3

    all_boxes_R_M.append(boxes_R_M)
    all_scores_R_M.append(scores_R_M)
    all_classes_R_M.append(classes_R_M)
    all_num_objects_R_M.append(num_objects_R_M)

import pandas as pd

# This stores the rider and motorcycle dataframes for each image in the validation set
all_rider = []
all_motorcycle = []

# Getting the rider motorcycle dataframe 

for i, frame in enumerate(frames):
    # Bounding boxes are in normalized ymin, xmin, ymax, xmax
    original_h, original_w, _ = frame.shape
    classes_R_M = all_classes_R_M[i]
    boxes_R_M = all_boxes_R_M[i]

    #getting rider, motorcycle dataframe
    df = pd.DataFrame(classes_R_M, columns=['class_id'])
    ymin = boxes_R_M[:, 1]
    xmin = boxes_R_M[:, 0]
    ymax = boxes_R_M[:, 3]
    xmax = boxes_R_M[:, 2]
    df['x'] = pd.DataFrame(xmin + (xmax-xmin)/2, columns=['x'])
    df['y'] = pd.DataFrame(ymin + (ymax-ymin)/2, columns=['y'])
    df['w'] = pd.DataFrame(xmax-xmin, columns=['w'])
    df['h'] = pd.DataFrame(ymax-ymin, columns=['h'])
    rider = df.loc[df['class_id']==0]
    motorcycle = df.loc[df['class_id']==3]
    all_rider.append(rider)
    all_motorcycle.append(motorcycle)

# assign instance id to each rider AND motorcycle
import sys
sys.path.append('../../../Testing-Images/')
from core.association import motor_rider_iou, motor2_rider_iou, area

def get_instance(rider, motorcycle, iou_threshold):
    """
    args:
    rider, motorcycle : pd.DataFrame

    output:
    rider, motorycle : pd.DataFrame with a column named 'instance_id'
    """
    rider['instance_id'] = -1
    motorcycle['instance_id'] = -1
    
    for i in range(len(motorcycle)):
        motorcycle.iat[i,motorcycle.columns.get_loc('instance_id')] = i
        for j in range(len(rider)):
            if (motor_rider_iou(motorcycle.iloc[i], rider.iloc[j]) > iou_threshold):
                if (rider.iloc[j]['instance_id'] == -1):
                    rider.iat[j,rider.columns.get_loc('instance_id')] = i
                else:
                    print("df")
                    instance = int(rider.iloc[j]['instance_id'])
                    instance_final = motor2_rider_iou(motorcycle.iloc[i], motorcycle.iloc[instance], rider.iloc[j], i, instance)
                    rider.iat[j,rider.columns.get_loc('instance_id')] = instance_final
    return rider, motorcycle

for i in range(len(all_rider)):
    all_rider[i], all_motorcycle[i] = get_instance(all_rider[i], all_motorcycle[i], 0.01)
    # print("\n\nRider and Motorcycle Dataframe for image ", i)


## Helper Functions

def heuristic_on_pred(a, motor, rider_ins):
    no_of_bbox = len(motor) + len(rider_ins)
    if (motor['w']==0):
        no_of_bbox = no_of_bbox - 1

    mean_w = (rider_ins['w'].sum() + motor['w'].sum())/no_of_bbox
    mean_x = (rider_ins['x'].sum() + motor['x'].sum())/no_of_bbox

    if (a[4]<mean_w):
        a[4] = motor['w'].mean()
    if (a[0] < mean_x - mean_w/2):
        a[0] = rider_ins['x'].mean()
    if (a[0] > mean_x + mean_w/2):
        a[0] = rider_ins['x'].mean()
    return a

def corner_condition(y, xmax, ymax):
    if (y[0]<0):
        y[0] = 0
    if (y[0]>xmax):
        y[0] = xmax
    if (y[1]<0):
        y[1] = 0
    if (y[1]>ymax):
        y[1] = ymax
    if (y[2]<0):
        y[2] = 0
    if (y[2]>xmax):
        y[2] = xmax
    if (y[3]<0):
        y[3] = 0
    if (y[3]>ymax):
        y[3] = ymax
    if (y[4]<0):
        y[4] = 0
    if (y[4]>xmax):
        y[4] = xmax
    if (y[5]<0):
        y[5] = 0
    if (y[5]>ymax):
        y[5] = ymax
    if (y[6]<0):
        y[6] = 0
    if (y[6]>xmax):
        y[6] = xmax
    if (y[7]<0):
        y[7] = 0
    if (y[7]>ymax):
        y[7] = ymax
    
    return y

all_frames_trap_boxes = []

for image_idx in range(len(frames)):
    motorcycle = all_motorcycle[image_idx]
    rider = all_rider[image_idx]
    trapezium_boxes = np.zeros((len(motorcycle), 8))
    trapezium_idx = 0
    for i in range(len(motorcycle)):
        input = []
        motor = motorcycle.iloc[i]
        instance = motor['instance_id']
        input.extend([float (motor['x']),float (motor['y']),float (motor['w']),float (motor['h'])])

        rider_ins = rider.loc[rider['instance_id']==instance]
            
        for j in range(len(rider_ins)):
            input.extend([float (rider_ins.iloc[j]['x']),float (rider_ins.iloc[j]['y']),float (rider_ins.iloc[j]['w']),float (rider_ins.iloc[j]['h'])])
        
        x=np.zeros((1,24))
        x[0,:len(input)] = np.array(input).reshape((1,-1))
        predict = trapez_model.predict(x)
 
        a = predict[0]
        a = heuristic_on_pred(a, motor, rider_ins)
        trapezium_boxes[trapezium_idx][0], trapezium_boxes[trapezium_idx][1], trapezium_boxes[trapezium_idx][2], trapezium_boxes[trapezium_idx][3], trapezium_boxes[trapezium_idx][4], trapezium_boxes[trapezium_idx][5],trapezium_boxes[trapezium_idx][6] ,trapezium_boxes[trapezium_idx][7]  = (a[0] - a[4]/2)*original_w,(a[5]+x[0][1]-x[0][3]/2)*original_h, (a[0] - a[4]/2)*original_w,(a[2]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[3]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[6]+x[0][1]-x[0][3]/2)*original_h
        
        trapezium_boxes[trapezium_idx] = corner_condition(trapezium_boxes[trapezium_idx], original_w, original_h)
        trapezium_idx = trapezium_idx+1
            
    trapezium_boxes = trapezium_boxes[:trapezium_idx,:]
    trapezium_boxes = trapezium_boxes.astype(int)

    all_frames_trap_boxes.append(trapezium_boxes)

new = []

for i in range(len(all_frames_trap_boxes)):
    l = []
    for j in range(len(all_frames_trap_boxes[i])):
        obj = all_frames_trap_boxes[i][j]
        if(area([(obj[0], obj[1]), (obj[2], obj[3]), (obj[4], obj[5]), (obj[6], obj[7])]) != 0):
            l.append(obj)
    new.append(l)

all_frames_trap_boxes = new

from core.association import iou

def trapez_rider_iou(y_, rider):

    y_ = [[y_[0], y_[1]], [y_[2], y_[3]], [y_[4], y_[5]], [y_[6], y_[7]]]
    x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
    rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]

    return iou(y_, rider)

def get_instance_with_trapez(rider, trapezium, iou_threshold):
    """
    args:
    rider, trapezium : pd.DataFrame

    output:
    rider, trapezium : pd.DataFrame with a column named 'instance_id'
    """
    # print info of the rider and trapezium dataframes
    # print("Rider dataframe info:")
    # print(rider.info())
    trapezium_instance_ids = np.zeros(len(trapezium))
    for i in range(len(trapezium)):
        trapezium_instance_ids[i] = i
        for j in range(len(rider)):
            if (trapez_rider_iou(trapezium[i], rider.iloc[j]) > iou_threshold):
                if (rider.iloc[j]['instance_id'] == -1):
                    rider.iat[j,rider.columns.get_loc('instance_id')] = i
                else:
                    instance = int(rider.iloc[j]['instance_id'])
                    if (trapez_rider_iou(trapezium[instance], rider.iloc[j]) < trapez_rider_iou(trapezium[i], rider.iloc[j])):
                        rider.iat[j,rider.columns.get_loc('instance_id')] = i
                    else:
                        rider.iat[j,rider.columns.get_loc('instance_id')] = instance
    return rider, trapezium_instance_ids

all_frames_trap_classes = []

for image_idx in range(len(frames)):
    rider, trapez_instance_ids = get_instance_with_trapez(all_rider[image_idx], all_frames_trap_boxes[image_idx], 0.01)
    trapezium_classes = np.zeros(len(all_frames_trap_boxes[image_idx]))
    # The class of a trapezium is 0 if there are less than 3 riders in it, or else 1
    for i in range(len(all_frames_trap_boxes[image_idx])):
        trapezium_classes[i] = len(rider.loc[rider['instance_id']==i])
        if (trapezium_classes[i] >= 3):
            trapezium_classes[i] = 1
        else :
            trapezium_classes[i] = 0
    all_frames_trap_classes.append(trapezium_classes)

def list_of_tuples(obj):
    '''
    obj is of shape 8
    '''
    return [(obj[0], obj[1]), (obj[2], obj[3]), (obj[4], obj[5]), (obj[6], obj[7])]

def iou_mat(gt, pred):
    '''
    returns the iou mat of shape (gt.shape[0], pred.shape[0])
    '''
    mat = np.zeros((len(gt), len(pred)))
    for i in range(len(gt)):
        for j in range(len(pred)):
            mat[i,j] = iou(list_of_tuples(gt[i]), list_of_tuples(pred[j]))
    return mat

predicted_boxes = all_frames_trap_boxes
predicted_classes = all_frames_trap_classes

fp = 0
fn = 0
tp = 0
iou_thresh = 0.5
for i in range(len(gt_boxes)):
    # if(i!=3):
    #     continue
    gt_box = gt_boxes[i]
    gt_class = gt_classes[i]
    pred_box = predicted_boxes[i]
    pred_class = predicted_classes[i]
    mat = iou_mat(gt_box, pred_box)
    ind = np.argsort(mat, axis = -1)
    sorted_mat = np.sort(mat, axis = -1)

    gt_assign = np.ones((len(gt_box))) * -1
    gt_assign_class = np.ones((len(gt_box))) * -1

    for i in range(len(gt_box)):
        if(sorted_mat.shape[1] > 0):
            if(sorted_mat[i,-1] > iou_thresh and pred_class[ind[i, -1]] == gt_class[i]):
                gt_assign[i] = ind[i, -1]

    val = gt_assign[gt_assign != -1]

    if (len(np.unique(val)) != len(val)):
        print("yoyo")
    else:
        fn += len(gt_assign[gt_assign == -1])
        fp += len(pred_box) - len(val)
        tp += len(val)


print(fn)
print(fp)
print(tp)
print(tp/(tp + fp))
print(tp/(tp + fn))
