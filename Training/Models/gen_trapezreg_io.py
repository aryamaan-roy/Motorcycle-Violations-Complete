# The directory for trapezium annotations containing the xml files
trapezium_xml_folder = "/Users/keshavgupta/desktop/data/training_data_817Images/Trapezium_instance_boxes"
# The directory for motorcycle and rider annotations corresponding to the trapzium annotations
input_txt_folder = "/Users/keshavgupta/desktop/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing"

######################################################################
# The following files will give error when producing the output files:

# training_data_817Images/Trapezium_instance_boxes/0000585_13865.xml
# training_data_817Images/Trapezium_instance_boxes/0000752_6624.xml 

# Go to the dataset and modify these files
# Under the <object> tag, wherever you see <type>Bounding Box</type>
# and remove that line
######################################################################

import os
import pandas as pd
import glob
from os.path import basename
import numpy as np
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
pd.options.mode.chained_assignment = None
def distance(x1, y1, x2, y2):
  return (((x2-x1)**2 + (y2-y1)**2)**0.5)

def iou(bbox1, bbox2):
    """
    args:
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]

    output:
    iou: (float)
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    poly3 = poly2.intersection(poly1)

    Ar1 = float(poly1.area)
    Ar2 = float(poly2.area)
    Ar_of_int = float(poly3.area)

    iou = Ar_of_int / (Ar1 + Ar2 - Ar_of_int)

    return iou

def union(bbox1, bbox2):
    """
    args:
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]

    output:
    iou: (float)
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    poly3 = poly2.union(poly1)


    return Polygon(poly3)

def motor_rider_iou(motorcycle, rider):
  x, y, w, h = motorcycle['x'], motorcycle['y'], motorcycle['w'], motorcycle['h']
  motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
  x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
  rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
  return iou(motor, rider)

def motor2_rider_iou(motorcycle_now, motorcycle_before, rider, instance_now, instance_before):
  if (motor_rider_iou(motorcycle_now, rider) > motor_rider_iou(motorcycle_before, rider)):
    return instance_now
  else:
    return instance_before
  
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

def xml_to_df(file_path):
  """
  args:
  file_path(str): the file should be in xml and the format should be Labelme 3.0 (preferably downloaded from CVAT)

  output:
  data : pd.DataFrame
  """
  tree = ET.parse(file_path)
  root = tree.getroot()
  width = float (root[3][1].text)
  height = float (root[3][0].text)
  x1 = []
  y1 = []
  x2 = []
  y2 = []
  x3 = []
  y3 = []
  x4 = []
  y4 = []

  for i in range(4, len(root)):
    print(i)
    x1.append(float (root[i][7][0][0].text)/width)
    y1.append(float (root[i][7][0][1].text)/height)
    x2.append(float (root[i][7][1][0].text)/width)
    y2.append(float (root[i][7][1][1].text)/height)
    x3.append(float (root[i][7][2][0].text)/width)
    y3.append(float (root[i][7][2][1].text)/height)
    x4.append(float (root[i][7][3][0].text)/width)
    y4.append(float (root[i][7][3][1].text)/height)

  data = pd.DataFrame({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"x3":x3,"y3":y3,"x4":x4,"y4":y4})
  return data

def motorandrider_bigbox_iou (motorandrider, big_bbox):
  bbox = [[big_bbox['x1'], big_bbox['y1']],[big_bbox['x2'], big_bbox['y2']],[big_bbox['x3'], big_bbox['y3']],[big_bbox['x4'], big_bbox['y4']]]
  return iou(motorandrider, bbox)

def motorandrider2_bigbox_iou(motorandriderbefore, motorandridernow, big_bbox, i, instance):
  if (motorandrider_bigbox_iou(motorandriderbefore,big_bbox )>motorandrider_bigbox_iou(motorandridernow,big_bbox )):
    return instance
  else:
    return i
  
def union_of_rider_motor(motorcycle, rider, instance_id):
  rider = rider.loc[rider['instance_id']==instance_id]

  motorcycle = motorcycle.iloc[instance_id]
  x, y, w, h = motorcycle['x'], motorcycle['y'], motorcycle['w'], motorcycle['h']
  motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
  if(len(rider)==0):
    return union(motor, motor)


  x, y, w, h = motorcycle['x'], motorcycle['y'], motorcycle['w'], motorcycle['h']
  motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]

  for i in range(len(rider)):
    if (i==0):
      x, y, w, h = rider.iloc[i]['x'], rider.iloc[i]['y'], rider.iloc[i]['w'], rider.iloc[i]['h']
      rider1 = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
      poly = union(motor, rider1)
    else:
      x, y, w, h = rider.iloc[i]['x'], rider.iloc[i]['y'], rider.iloc[i]['w'], rider.iloc[i]['h']
      rider1 = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
      poly = union(poly, rider1)
  return poly


def get_output (rider, motorcycle, big_bbox, iou_threshold):
  for i in range(len(motorcycle)):
    for j in range(len(big_bbox)):
      motorandrider = union_of_rider_motor(motorcycle, rider, i)
      if (motorandrider_bigbox_iou(motorandrider, big_bbox.iloc[j]) > iou_threshold):
        if (big_bbox.iloc[j]['instance_id']==-1):
          big_bbox.iat[j, big_bbox.columns.get_loc('instance_id')] = i
        else:
          instance = int(big_bbox.iloc[j]['instance_id'])

          motorandriderbefore = union_of_rider_motor(motorcycle, rider, instance)
          motorandridernow = union_of_rider_motor(motorcycle, rider, i)
          instance_final = motorandrider2_bigbox_iou(motorandriderbefore, motorandridernow, big_bbox.iloc[j], i, instance)
          big_bbox.iat[j, big_bbox.columns.get_loc('instance_id')] = instance_final
  return big_bbox


x = np.zeros((4000, 24))
y = np.zeros((4000, 8))
num = 0

for file_path in glob.glob(input_txt_folder+"/*txt"):
  print(file_path)
  detections = pd.read_csv(file_path, header=None, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])

  rider = detections.loc[detections['class_id']==0]
  motorcycle = detections.loc[detections['class_id']==3]

  rider, motorcycle = get_instance(rider, motorcycle, 0.01)

  file_path = os.path.join(trapezium_xml_folder, basename(file_path).split('.')[0] + '.xml' )

  big_bbox = xml_to_df(file_path)
  big_bbox['instance_id'] = -1
  big_bbox = get_output(rider, motorcycle, big_bbox, 0.01)


  for i in range(len(big_bbox)):
    input = []
    output = []
    instance = big_bbox.iloc[i]['instance_id']
    # print("instance", instance)
    if (instance==-1):
      continue

    motor = motorcycle.loc[motorcycle['instance_id']==instance]
    rider_ins = rider.loc[rider['instance_id']==instance]
    bb = big_bbox.loc[big_bbox['instance_id']==instance]

    if (len(bb)==1):
      output.extend([bb['x1'], bb['y1'], bb['x2'], bb['y2'], bb['x3'], bb['y3'], bb['x4'], bb['y4']])
      y[num, :len(output)] = np.array(output).reshape((8,))
      input.extend([motor['x'], motor['y'], motor['w'], motor['h']])
      for j in range(len(rider_ins)):
        input.extend([rider_ins.iloc[j]['x'], rider_ins.iloc[j]['y'], rider_ins.iloc[j]['w'], rider_ins.iloc[j]['h']])
      x[num, :len(input)] = np.array(input).reshape((1,-1))
      num+=1


y = y[:num]
x = x[0:num]

input = pd.DataFrame(x)
output = pd.DataFrame(y)
input.to_csv("./input.csv")
output.to_csv("./output.csv")