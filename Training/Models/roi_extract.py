import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import os
from os.path import basename
import cv2
import glob
import warnings

warnings.filterwarnings("ignore")

# Put the path to the directory where you have saved the images and the 
# corresponding labels (all_class_labels)
img_folder = r"/Users/keshavgupta/desktop/data/validation_data_234Images/M_R_H_NH/"
txt_folder = r"/Users/keshavgupta/desktop/data/validation_data_234Images/M_R_H_NH/"

# Put the path to the directory where you want to save the extracted ROI's and the 
# corresponding labels
roi_folder = r"/Users/keshavgupta/desktop/data/new_val/"

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


def iou2(rider, head):
    xr1, xr2, yr1, yr2 = rider['x'] - rider['w']/2, rider['x'] + rider['w']/2, rider['y']-rider['h']/2, rider['y']+rider['h']/2
    xh1, xh2, yh1, yh2 = head['x'] - head['w']/2, head['x'] + head['w']/2, head['y']-head['h']/2, head['y']+head['h']/2
    # xr1 = xr1[0]
    # xr2 = xr2[0]
    # yr1 = yr1[0]
    # yr2 = yr2[0]
    if ((xr2<xh1) or (xr1>xh2) or (yr2<yh1) or (yr1>yh2)):
        overlap = 0
    else:
        x_coor = np.sort(np.array([xr1, xr2, xh1, xh2]))
        y_coor = np.sort(np.array([yr1, yr2, yh1, yh2]))
        width = x_coor[1] - x_coor[2]
        height = y_coor[1]-y_coor[2]
        overlap = width * height
        
    Ar1 = rider['w']*rider['h']
    Ar2 = head['w']*head['h']
    NIOU = round((Ar2 - overlap),4)/round((Ar1+Ar2-overlap),4)
    IOU = round(overlap,4)/round((Ar1+Ar2-overlap),4)
    return IOU, NIOU

def coeff(rider, head):
    IOU, NIOU = iou2(rider, head)
    if (round(NIOU,4)==0):
        val = 10000
    else:
        val = IOU/NIOU
    return (val)

def get_tlbr(arr):
    head = arr[:,:]
    xc = arr[:,1]
    yc = arr[:,2]
    w = arr[:,3]
    h = arr[:,4]
    head[:,1] = xc - w/2
    head[:,2] = yc - h/2
    head[:,3] = xc + w
    head[:,4] = yc + h
    return head

for files in glob.glob(txt_folder+"/*.txt"):
    # files = "/Users/keshavgupta/desktop/data/validation_data_234Images/M_R_H_NH/0006066_9425.txt"
    df = pd.read_csv(files, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])
    rider = df.loc[df['class_id']==0]
    motorcycle = df.loc[df['class_id']==3]
    helmet = df.loc[df['class_id']==1]
    no_helmet = df.loc[df['class_id']==2]

    head = pd.concat([helmet, no_helmet])
    head = head.to_numpy()
    tlbr_head = get_tlbr(head)
    rider, motorcycle = get_instance(rider, motorcycle, 0.01)
    path = os.path.join(img_folder, basename(files).split('.')[0] + ".jpg")
    img = cv2.imread(path)

    motorcycle_ins = len(motorcycle)

    for i in range(len(motorcycle)):
        motor = motorcycle.loc[motorcycle['instance_id']==i]
        instance = motorcycle.iloc[i]['instance_id']
        ride = rider.loc[rider['instance_id']== instance]
        # print(head.iloc[0])

        if (len(ride)==0):
            continue
        
        # print(type(motor['x'] + motor['w']/2))
        # print(motor['x'] + motor['w']/2)
        # print(type(max(ride['x'] + ride['w']/2)))
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

        count = 0
        t = int(ymin*img.shape[0])
        l = int(xmin*img.shape[1])
        b = int(ymax*img.shape[0])
        r = int(xmax*img.shape[1])

        if t<0 or l<0 or b<0 or r<0:
            continue

        head_rois = []

        for elem in (tlbr_head):
            head_xmin = int(elem[1]*img.shape[1])
            head_ymin = int(elem[2]*img.shape[0])
            head_xmax = int(elem[3]*img.shape[1])
            head_ymax = int(elem[4]*img.shape[0])
            
            if head_xmin > r or head_xmax < l or head_ymax < t or head_ymin > b:
                continue
            if (head_xmin < l):
                head_xmin = l
            if (head_xmax > r):
                head_xmax = r 
            if (head_ymin < t):
                head = t
            if (head_ymax > b):
                head_ymax = b
            
            new_imgw = r - l
            new_imgh = b - t

            head_xmax -= l
            head_xmin -= l
            head_ymax -= t
            head_ymin -= t

            head_xmin /= new_imgw
            head_ymin /= new_imgh
            head_xmax /= new_imgw
            head_ymax /= new_imgh

            head_xc = (head_xmin + head_xmax) / 2
            head_yc = (head_ymin + head_ymax) / 2
            head_w = (-head_xmin + head_xmax)
            head_h = (-head_ymin + head_ymax)
            class_idx = 0
            if(elem[0] == 2):
                class_idx = 1
            line = str(class_idx) + " " + str(head_xc) + " " + str(head_yc) + " " + str(head_w) + " " + str(head_h) + "\n"
            # print(line)
            head_rois.append(line)
            count += 1

        if(count >= 1):
            patch = img[t:b, l:r]
            new_path_img = os.path.join(roi_folder, basename(files).split('.')[0] +"_"+str(i)+ ".jpg")
            new_path_txt = os.path.join(roi_folder, basename(files).split('.')[0] +"_"+str(i)+ ".txt")
            cv2.imwrite(os.path.join(roi_folder, basename(files).split('.')[0] +"_"+str(i)+ ".jpg"), patch)
            file = open(new_path_txt, "x")
            for line in head_rois:
                file.write(line)
            # cv2.rectangle(patch, [head_xmin, head_ymin], [head_xmax, head_ymax], [255,0,0], 1, 0)
            # print(patch)
            # cv2.imshow("sfd", patch)
            # cv2.waitKey(0)
