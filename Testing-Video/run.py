import os
from operator import index

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import tensorflow.keras as keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.video import det_to_vid, interpolation
from core.association import *
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pickle
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './ Weights/yolov4-416',
#                    'path to weights file')
flags.DEFINE_string('weights_L4', '../Weights/yolov4/rider_motor_512','path to weights file')
flags.DEFINE_string('weights_RHNH', '../Weights/yolov4/helmet_no_helmet_512','path to weights file')
flags.DEFINE_string('classes', '../data/classes/4_class_detector.names','path to manes file')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_float('rider_pred_threshold', 1.5, 'IOU/NIOU area threshold')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('trapezium_pred_model', '../Weights/trapezium_regressor/Trapezium_Prediction_Weights.pickle', 'add the model weights for predicting trapezium bounding box as a post-processing step')
flags.DEFINE_string('video', '../data/Videos/3idiots.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './outputs/detections/3idiots.mp4', 'path to raw output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('interpolation', True, 'interpolate the missing bounding boxes based on future frames')


# Given a frame, the rider and motorcycle dataframes, the function returns the ROI frames along with its original position in the frame
def extract_roi(frame, rider, motorcycle) :
    """
    args:
    frame : np.array
    rider, motorcycle : pd.DataFrame

    output:
    roi_instances : list of np.array
    """
    roi_instances = []
    for i in range(len(motorcycle)):
        motor = motorcycle.loc[motorcycle['instance_id']==i]
        instance = motorcycle.iloc[i]['instance_id']
        ride = rider.loc[rider['instance_id']== instance]

        if (len(ride)==0):
            continue
        
        xmax = max(float(motor['x'] + motor['w']/2), max(ride['x'] + ride['w']/2))
        xmin = min(float(motor['x'] - motor['w']/2), min(ride['x'] - ride['w']/2))
        ymax = max(float(motor['y'] + motor['h']/2), max(ride['y'] + ride['h']/2))
        ymin = min(float(motor['y'] - motor['h']/2), min(ride['y'] - ride['h']/2))

        w = xmax - xmin
        h = ymax - ymin

        xmax = xmax + 0.05*w
        xmin = xmin - 0.05*w

        ymax = ymax + 0.05 * h
        ymin = ymin - 0.05 * h

        if (xmin < 0):
            xmin=0
        if (xmax >1):
            xmax=1
        if (ymax>1):
            ymax =1
        if(ymin<0):
            ymin =0

        t = int(ymin*frame.shape[0])
        l = int(xmin*frame.shape[1])
        b = int(ymax*frame.shape[0])
        r = int(xmax*frame.shape[1])

        if t<0 or l<0 or b<0 or r<0:
            continue
        roi_frame = frame[t:b, l:r]
        # roi_frame = frame
        original_position = (t,l,b,r)
        roi_dict = {'frame':roi_frame, 'original_position':original_position}
        roi_instances.append(roi_dict)

    return roi_instances

# Given y : Trapezium coordinates (x,y, offsets ...) and a single rider (x,y,w,h), the function returns the IOU between the two
def trapez_rider_iou(y_, rider):

    y_ = [[y_[0], y_[1]], [y_[2], y_[3]], [y_[4], y_[5]], [y_[6], y_[7]]]
    x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
    rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]

    return iou(y_, rider)

# Given the trapeziums and riders of a frame, the function compares IOU between all trapeziums and riders 
# and assigns instance IDs to the riders based on the IOU threshold
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

# This function is not used as of now. If function call is uncommented in main, it will basically annotate the frame with the boxes and labels and store it in the output folder
def save_annotated_frame(frame, bboxes, classes, scores, num_objects, frame_num):
    """
    args:
    frame : np.array
    bboxes : np.array
    classes : np.array
    scores : np.array
    num_objects : int
    frame_num : int

    output:
    None
    """
    # save annotated frame

    output_path = 'outputs/frames'
    colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i in range(num_objects):
        # save bbox
        # convert bbox from yolo format to opencv format
        current_class = int(classes[i])
        bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if current_class == 1 or current_class == 2:
            bbox = [bboxes[i][1] - bboxes[i][3] / 2, bboxes[i][0] - bboxes[i][2] / 2, bboxes[i][1] + bboxes[i][3] / 2, bboxes[i][0] + bboxes[i][2] / 2]
        bbox = [int(bbox[0] * frame.shape[0]), int(bbox[1] * frame.shape[1]), int(bbox[2] * frame.shape[0]), int(bbox[3] * frame.shape[1])]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (bbox[1], bbox[0]), (bbox[3], bbox[2]), colors[current_class], 2)
        # save class
        cv2.putText(frame, str(current_class), (int(bboxes[i][1]), int(bboxes[i][0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # save score
        cv2.putText(frame, str(scores[i]), (int(bboxes[i][1]), int(bboxes[i][2])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_path + '/frame_' + str(frame_num) + '.jpg', frame)
    # save file
    output_path = 'outputs/annotations'
    with open(output_path + '/frame_' + str(frame_num) + '.txt', 'w') as f:
        for i in range(num_objects):
            # convert bbox from yolo format to opencv format
            bbox = [bboxes[i][1] - bboxes[i][3] / 2, bboxes[i][0] - bboxes[i][2] / 2, bboxes[i][1] + bboxes[i][3] / 2, bboxes[i][0] + bboxes[i][2] / 2]
            bbox = [int(bbox[0] * frame.shape[0]), int(bbox[1] * frame.shape[1]), int(bbox[2] * frame.shape[0]), int(bbox[3] * frame.shape[1])]
            # convert to int
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            f.write(str(int(classes[i])) + ' ' + str(scores[i]) + ' ' + str(bbox[1]) + ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + '\n')


# assigns a unique instance id to each motorcycle. Then loops over all riders and assigns the same instance id to the rider if the IOU is greater than the threshold.
# If the rider is already assigned to a motorcycle, then the iou of rider with the 2 motorcycles is calculated and the rider is assigned to the motorcycle with the higher iou.
# Updated rider and motorcycle dataframes are returned.
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
                    instance = int(rider.iloc[j]['instance_id'])
                    instance_final = motor2_rider_iou(motorcycle.iloc[i], motorcycle.iloc[instance], rider.iloc[j], i, instance)
                    rider.iat[j,rider.columns.get_loc('instance_id')] = instance_final

    return rider, motorcycle


# below functions are used as helper functions in the trapezium regressor. They handle corner cases for the predicted trapezium

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

# This function basically takes a bbox and matches the motorcycle with the bbox with the least distance. It returns the instance id of that motorcycle and the number of riders on that motorcycle. Used in tracking
def find(bbox, instance, bboxes, classes):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    flag = 10000
    idx = -1
    for i in range(len(bboxes)):
        if (classes[i]== 'Motorcycle'):
            b = bboxes[i]
            val = abs(xmin-b[0]) + abs(ymin-b[1]) + abs(xmax-b[0]-b[2]) + abs(ymax-b[1]-b[3])
            if (val<flag):
                flag = val
                idx = i
    if (idx != -1):
        num, num_riders = instance[idx][0], instance[idx][1]
    else:
        num, num_riders,val = -1, -1, flag
    return num, num_riders, val



def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    # otherwise load standard tensorflow saved model
    else:
        # The M+R model and the RHNH model are loaded here
        infer_rider_motor = keras.models.load_model(FLAGS.weights_L4)
        infer_RHNH = keras.models.load_model(FLAGS.weights_RHNH)
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    trapez_model = pickle.load(open(FLAGS.trapezium_pred_model, 'rb'))




    triple_rider_violation = {}
    triple_rider_violated = []


    HNH_violation = {}
    HNH_violations = 0
    HNH_violated = []

    column_names = ["frame_id", "class_name", "track_id", "trapez_0", "trapez_1", "trapez_2", "trapez_3", "trapez_4", "trapez_5", "trapez_6", "trapez_7", "bbox_0", "bbox_1", "bbox_2", "bbox_3"]

    detections_tovid = []


    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            detections_tovid = pd.DataFrame(detections_tovid, columns=column_names)
            dir_path_ = os.path.dirname(FLAGS.output)
            path = os.path.join(dir_path_ ,os.path.basename(FLAGS.output).split(".")[0]+".csv")
            detections_tovid.to_csv(path, index=False)
            interpolated_detections=interpolation(path)
            print('Video has ended or failed, try a different video format!')
            print('==================================')
            if FLAGS.interpolation:
                print('Improving tracking quality by interpolation process....')
                dir_path_ = os.path.dirname(FLAGS.output)
                path_ = os.path.join(dir_path_, os.path.basename(FLAGS.output).split(".")[0]+"_interpolated."+os.path.basename(FLAGS.output).split(".")[1])
                det_to_vid(FLAGS.video, interpolated_detections, path_)
            else:
                print('Generation of camera ready video in progress....')
                dir_path_ = os.path.dirname(FLAGS.output)
                path_ = os.path.join(dir_path, os.path.basename(FLAGS.output).split(".")[0]+"_interpolated"+os.path.basename(FLAGS.output).split(".")[1])
                det_to_vid(FLAGS.video, path, path_)
            break
        frame_num +=1
        # newline
        print('\n\nFrame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            # The Rider and Motorcycle boxes are predicted on the frame
            batch_data = tf.constant(image_data)
            rider_motor_bbox = infer_rider_motor.predict(batch_data)
            for value in rider_motor_bbox:
                temp_value = np.expand_dims(value, axis=0)
                boxes_R_M = temp_value[:, :, 0:4]
                pred_conf = temp_value[:, :, 4:]
            
        # Non-max suppression is applied to the R+M boxes

        boxes_R_M, scores_R_M, classes_R_M, valid_detections_R_M = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes_R_M, (tf.shape(boxes_R_M)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # Valid Detection boxes are filtered and converted to numpy arrays
        num_objects_R_M = valid_detections_R_M.numpy()[0]
        boxes_R_M = boxes_R_M.numpy()[0]
        boxes_R_M = boxes_R_M[0:int(num_objects_R_M)]
        scores_R_M = scores_R_M.numpy()[0]
        scores_R_M = scores_R_M[0:int(num_objects_R_M)]
        classes_R_M = classes_R_M.numpy()[0]
        classes_R_M = classes_R_M[0:int(num_objects_R_M)]

        deleted_indx = []
        allowed_classes = [0,1]
        for i in range(num_objects_R_M):
            class_indx = int(classes_R_M[i])
            if class_indx not in allowed_classes:
                deleted_indx.append(i)
        boxes_R_M = np.delete(boxes_R_M, deleted_indx, axis=0)
        scores_R_M = np.delete(scores_R_M, deleted_indx, axis=0)
        classes_R_M = np.delete(classes_R_M, deleted_indx, axis=0)
        num_objects_R_M = len(classes_R_M)

        # MOTORCYCLE CLASS CHANGED TO 3 FROM 1
        classes_R_M[classes_R_M == 1] = 3

        # Getting the rider motorcycle dataframe 

        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        original_h, original_w, _ = frame.shape

        #getting rider, motorcycle dataframe
        df = pd.DataFrame(classes_R_M, columns=['class_id'])
        ymin = boxes_R_M[:, 0]
        xmin = boxes_R_M[:, 1]
        ymax = boxes_R_M[:, 2]
        xmax = boxes_R_M[:, 3]
        df['x'] = pd.DataFrame(xmin + (xmax-xmin)/2, columns=['x'])
        df['y'] = pd.DataFrame(ymin + (ymax-ymin)/2, columns=['y'])
        df['w'] = pd.DataFrame(xmax-xmin, columns=['w'])
        df['h'] = pd.DataFrame(ymax-ymin, columns=['h'])
        rider = df.loc[df['class_id']==0]
        motorcycle = df.loc[df['class_id']==3]

        # Assigning Instance IDs to the rider and motorcycle
        rider, motorcycle = get_instance(rider, motorcycle, 0.01)

        print(rider)
        print(motorcycle)

        # Getting the ROI instances in a frame (to be used for helmet detection)
        roi_instances = extract_roi(frame, rider, motorcycle)

        print("Number of roi: ", len(roi_instances))

              
        all_batch_data = []
        frame_size_ROI = []
        # Preparing the ROI instances for input to the helmet detection model
        for i in range(len(roi_instances)):
            instance_frame = roi_instances[i]['frame']
            frame_size_ROI.append(instance_frame.shape)
            image_data = cv2.resize(instance_frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            batch_data = tf.constant(image_data)
            all_batch_data.append(batch_data)

        print("ROI frame size: ", frame_size_ROI)

        # Infering the HNH boxes on the ROI instances
        all_instance_H_NH_boxes = []

        for i in range(len(all_batch_data)):
            batch_data = all_batch_data[i]
            H_NH_boxes = infer_RHNH.predict(batch_data)
            all_instance_H_NH_boxes.append(H_NH_boxes)

        all_final_HNH_boxes = []
        all_pred_conf_HNH = []

        # Filtering bbox coordinates from the HNH boxes (as was done for M+R boxes)
        for i in range(len(all_instance_H_NH_boxes)):
            single_instance_boxes = all_instance_H_NH_boxes[i]
            for value in single_instance_boxes:
                temp_value = np.expand_dims(value, axis=0)
                boxes = temp_value[:, :, 0:4]
                conf = temp_value[:, :, 4:]
            all_final_HNH_boxes.append(boxes)
            all_pred_conf_HNH.append(conf)

        all_bboxes_HNH = []
        all_scores_HNH = []
        all_classes_HNH = []
        all_num_objects_HNH = []
        
        # Non-max suppression on the HNH boxes (as was done for M+R boxes)
        for i in range(len(all_final_HNH_boxes)):
            boxes = all_final_HNH_boxes[i]
            conf = all_pred_conf_HNH[i]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(conf, (tf.shape(conf)[0], -1, tf.shape(conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.5
            )

            # convert data to numpy arrays and slice out unused elements (as was done for M+R boxes)
            num_objects_HNH = valid_detections.numpy()[0]
            bboxes_HNH = boxes.numpy()[0]
            bboxes_HNH = bboxes_HNH[0:int(num_objects_HNH)]
            scores_HNH = scores.numpy()[0]
            scores_HNH = scores_HNH[0:int(num_objects_HNH)]
            classes_HNH = classes.numpy()[0]
            classes_HNH = classes_HNH[0:int(num_objects_HNH)]

            deleted_indx = []
            allowed_classes = [0, 1]
            for i in range(num_objects_HNH):
                class_indx = int(classes_HNH[i])
                if class_indx not in allowed_classes:
                    deleted_indx.append(i)
            bboxes_HNH = np.delete(bboxes_HNH, deleted_indx, axis=0)
            scores_HNH = np.delete(scores_HNH, deleted_indx, axis=0)
            classes_HNH = np.delete(classes_HNH, deleted_indx, axis=0)
            num_objects_HNH = len(classes_HNH)

            classes_HNH[classes_HNH == 1] = 2
            classes_HNH[classes_HNH == 0] = 1

            all_bboxes_HNH.append(bboxes_HNH)
            all_scores_HNH.append(scores_HNH)
            all_classes_HNH.append(classes_HNH)
            all_num_objects_HNH.append(num_objects_HNH)
        
        # Contains bboxes, scores, classes and number of objects for each ROI instance in a frame
        print("HNH --- ROI")
        print(all_bboxes_HNH)
        print(all_num_objects_HNH)
        print(all_classes_HNH)
        print(all_scores_HNH)

        final_bboxes_HNH = []
        final_scores_HNH = []
        final_classes_HNH = []
        final_num_objects_HNH = 0

        # We need to transpose the bbox coordinates in the ROI instances to the original frame
        # convert the bounding box to the original image in yolo format and store it in final_bboxes_HNH
        for i in range(len(roi_instances)):
            original_position = roi_instances[i]['original_position']
            full_frame_width = frame.shape[1]
            full_frame_height = frame.shape[0]
            for j in range(all_num_objects_HNH[i]):
                xmin = int(all_bboxes_HNH[i][j][1] * frame_size_ROI[i][1])
                ymin = int(all_bboxes_HNH[i][j][0] * frame_size_ROI[i][0])
                xmax = int(all_bboxes_HNH[i][j][3] * frame_size_ROI[i][1])
                ymax = int(all_bboxes_HNH[i][j][2] * frame_size_ROI[i][0])
                xmin = xmin + original_position[1]
                xmax = xmax + original_position[1]
                ymin = ymin + original_position[0]
                ymax = ymax + original_position[0]
                # convert to yolo format 
                x_center = (xmin + xmax) / (2 * full_frame_width)
                y_center = (ymin + ymax) / (2 * full_frame_height)
                width = (xmax - xmin) / full_frame_width
                height = (ymax - ymin) / full_frame_height
                final_bboxes_HNH.append([x_center, y_center, width, height])
                final_scores_HNH.append(all_scores_HNH[i][j])
                final_classes_HNH.append(all_classes_HNH[i][j])
                final_num_objects_HNH += 1


        # Concatenating bboxes, scores and classes for M+R and HNH into a single structure which would be passed to deepsort
        if final_num_objects_HNH > 0:
            bboxes = np.concatenate((boxes_R_M, final_bboxes_HNH))
            scores = np.concatenate((scores_R_M, final_scores_HNH))
            classes = np.concatenate((classes_R_M, final_classes_HNH))
            num_objects = final_num_objects_HNH + num_objects_R_M
        else:
            bboxes = boxes_R_M
            scores = scores_R_M
            classes = classes_R_M
            num_objects = num_objects_R_M

        print("CONCATENATED")
        print("Classes", classes)
        print("Scores", scores)
        print("Bboxes", bboxes)
        print("Num objects", num_objects)

        # save_annotated_frame(frame, bboxes, classes, scores, num_objects, frame_num)

        if FLAGS.count:
            #cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(num_objects))

        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        original_h, original_w, _ = frame.shape


        # Predicting the Trapeziums
        # y would store the trapezium coordinates for all the trapeziums in the frame
        y = np.zeros((len(motorcycle), 8))
        num = 0

        # Looping over all the motorcycles in the frame and predicting the trapezium (given the motorcycle and rider coordinates)
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
            y[num][0], y[num][1], y[num][2], y[num][3], y[num][4], y[num][5],y[num][6] ,y[num][7]  = (a[0] - a[4]/2)*original_w,(a[5]+x[0][1]-x[0][3]/2)*original_h, (a[0] - a[4]/2)*original_w,(a[2]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[3]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[6]+x[0][1]-x[0][3]/2)*original_h
            
            y[num] = corner_condition(y[num], original_w, original_h)
            num = num+1

        # Trapeziums stored in the trapez_bboxes variable
        trapez_bboxes = y[:num,:]
        # Rider and trapeziums assigned a instance id (IOU Based Rider Counting)
        rider, trapez_instance_ids = get_instance_with_trapez(rider, trapez_bboxes, 0.01)
        
        # tracker_instance_trapez stores the instance id and the number of riders in the trapezium
        tracker_instance_trapez = []
        for i in range(len(trapez_bboxes)):
            instance = trapez_instance_ids[i]
            rider_ins = rider.loc[rider['instance_id']==instance]
            if (len(rider_ins)==0):
                tracker_instance_trapez.append([-1, -1])
                continue
            tracker_instance_trapez.append([instance, len(rider_ins)])

        # To delete the motorcycles which are not in the trapezium
        deleted_indx = []
        instance = np.zeros((len(bboxes), 2), dtype = int)
        k = 0
        for i in range(len(bboxes)):
            if (classes[i]==3):
                if (int(tracker_instance_trapez[k][1]) == -1):
                    deleted_indx.append(i)
                else:
                    instance[i][:] = int(tracker_instance_trapez[k][0]), int(tracker_instance_trapez[k][1])
                    k = k+1
            else:
                instance[i][:] = -1, -1

        print("Instance", instance)
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        classes = np.delete(classes, deleted_indx, axis=0)
        num_objects = len(classes)

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w, classes)
        print("Bboxes after denormalize", bboxes)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        print("Tracker tracks")

        # update tracks
        num = 0
        j=0

        track_ids = []
        nums = [] 
        riders = []
        vals = []

        # Loops over all the tracks . If track is a motorcycle, then it finds the associated instance id, 
        # number of riders and min distnace (val) of the nearest motorcycle bbox.
        # It appends the above to the track_ids, nums, riders and vals list.
        # If the instance id of the nearest motorcycle bbox ṭhat the find function returns is already present in the nums list. 
        # Then it checks if the val returned by the find function is less than the val already present in the vals list. 
        # If yes, then it replaces the val and the corresponding track id and number of riders in the lists.

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if (class_name != 'Motorcycle'):
                continue
            num, num_riders, val = find(bbox, instance, bboxes, classes)
            if (num==-1):
                continue
            if (num not in nums):
                track_ids.append(track.track_id)
                nums.append(num)
                riders.append(num_riders)
                vals.append(val)
                continue
            idx = nums.index(num)
            if (val<=vals[idx]):
                track_ids[idx] = track.track_id
                vals[idx] = val
                riders[idx] = num_riders
            
            
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = [255, 0, 0]
            # print the track info
            print("ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            if (class_name == 'No-Helmet'):
                if track not in HNH_violation:
                    HNH_violation[track] = 1

                if track in HNH_violation:
                    HNH_violation[track]+=1
                
                if HNH_violation[track]==2:
                    HNH_violated.append(track)

            if track in HNH_violated:
                color = [255, 0, 0]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(5+len(str(track.track_id)))*17, int(bbox[1])),(255,0,0), -1)
                cv2.putText(frame,  "ID:" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                detections_tovid.append([str(frame_num), "No-Helmet", str(track.track_id), "0", "0", "0", "0", "0", "0", "0", "0", str(int(bbox[0])), str(int(bbox[1])), str(int(bbox[2])), str(int(bbox[3])) ])
                continue

            if (class_name == "Helmet"):
                color = [0,255, 0]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(5+len(str(track.track_id)))*17, int(bbox[1])),(0,255,0), -1)
                cv2.putText(frame,  "ID:" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                detections_tovid.append([str(frame_num), "Helmet", str(track.track_id), "0", "0", "0", "0", "0", "0", "0", "0", str(int(bbox[0])), str(int(bbox[1])), str(int(bbox[2])), str(int(bbox[3])) ])
                continue

            if class_name == 'Motorcycle':
                if (track.track_id not in track_ids):
                    continue
                idx = track_ids.index(track.track_id)
                num = nums[idx]
                num_riders = riders[idx]
                if (num==-1):
                    continue
                if track not in  triple_rider_violation:
                    triple_rider_violation[track] = 0
                
                if (num_riders>2):
                    triple_rider_violation[track] += 1

                if ( triple_rider_violation[track]>=3):
                    if (track not in triple_rider_violated):
                        triple_rider_violated.append(track)
                
                trapez = trapez_bboxes[int(num)]
                if (len(trapez)==0):
                    continue
                pts = np.array([[[trapez[0], trapez[1]], [trapez[2], trapez[3]], [trapez[4], trapez[5]], [trapez[6], trapez[7]]]], np.int32)
                pts = pts.reshape((-1, 1, 2))

                if (track in triple_rider_violated):
                    color = [255, 0, 0]
                    cv2.polylines(frame, [pts], True , color, 2)
                    cv2.putText(frame, "ID:"+ str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    detections_tovid.append([str(frame_num), "Triple_Rider", str(track.track_id), str(int(trapez[0])), str(int(trapez[1])), str(int(trapez[2])), str(int(trapez[3])), str(int(trapez[4])), str(int(trapez[5])), str(int(trapez[6])), str(int(trapez[7])), "0", "0", "0", "0"])
                elif (num_riders<=2):
                    color = [0,255, 0]
                    cv2.polylines(frame, [pts], True , color, 2)
                    cv2.putText(frame, "ID:"+ str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    detections_tovid.append([str(frame_num), "safe_rider", str(track.track_id), str(int(trapez[0])), str(int(trapez[1])), str(int(trapez[2])), str(int(trapez[3])), str(int(trapez[4])), str(int(trapez[5])), str(int(trapez[6])), str(int(trapez[7])), "0", "0", "0", "0"])
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.rectangle(frame,(0,0),(425,50),(255,0,0),-1)
        cv2.putText(frame, 'Triple Rider Violations:' + str(len(triple_rider_violated)),(30,30),0, 1, (255,255,255),2)
        cv2.rectangle(frame,(original_w - 425,0),(original_w,50),(255,0,0),-1)
        cv2.putText(frame, 'Helmet-Violations : ' + str(len(HNH_violated)),(original_w - 425 + 30,30),0, 1, (255,255,255),2)
        cv2.putText(frame, 'frame_num : ' + str(frame_num),(original_w-850 + 30,30),0, 1, (0,0,0),2)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
