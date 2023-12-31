# Steps for training on YOLOv4 and YOLOv5 : 

### Here first the steps on YOLOv4 are given and then on YOLOv5

### Additional instructions for Curriculum learning are given at the end. Please read them before executing any of the normal commands as well


# YOLOv4

## 1) Clone the original YOLOv4 repo first by
```
git clone https://github.com/AlexeyAB/darknet
```

## 2) Run the following command
```
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
```
### If you want to build and run on CUDA GPU then you also need to make the following changes:

(You can check if you have cuda or not and the cuda version :
/usr/local/cuda/bin/nvcc --version)
```
sed -i 's/GPU=0/GPU=1/' Makefile

sed -i 's/CUDNN=0/CUDNN=1/' Makefile

sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```


## 3) Run 
```
make
```
## 4) Download pretrained weights:
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```
## 5) Get the data in YOLOv4 format
(class_id, X_center, Y_center, width, height)

You must have the training data, validation data with you in some directory (not necessarily in the current YOLOv4 directory) along with the labels in the YOLOv4 format in the same directory as the image folder.
You then need to create 2 text files (one for training and one for validation) in which all the paths to the images are there (Absolute paths)

For ex : 
Assume your YOLOv4 directory is present in your Motorcycle-Violations-Complete/Training/Models folder:
In context of the IDD dataset your home could look like:
```
Motorcycle-Violations-Complete/Training/Models:
    YOLOv4
    final_test_set M_R_H_NH_after_preprocessing:
        img1.jpg
        img1.txt
        img2.jpg
        img2.txt
        .
        .
        .
    validation_data_234Images:
        val1.jpg
        val1.txt
        val2.jpg
        val2.txt
        .
        .
        .
```
Run
```
cd final_test_set M_R_H_NH_after_preprocessing/
readlink -f *.jpg > ../train.txt
cd ..
cd M_R_H_NH
readlink -f *.jpg > ../val.txt
```
Now in your home directory you will have 2 files 
train.txt
val.txt
that you will need in the future steps

(For curriculum training, please refer to the end)
(For Helmet No-Helmet training, please refer to the end)

## 6) Run the following command
```
cp cfg/yolov4-custom.cfg cfg/yolov4-obj.cfg
```
This is the configuration file for training where you specify the network architecture details and also the hyperparameters for training.

You need to change the file according to the number of classes you have

If you have 'c' classes
then in the configuration file you need to set 
filters = (c + 5) * 3
just before the [yolo] section
and also the number of classes in the yolo section.
Refer to the yolov4-obj.cfg file in the repo here.
Note that there are multiple [yolo] sections in the config file and you need to change all of them. 

## 7) The next step is to edit the files in the data directory

Run
```
cd data
```
Now modify the obj.data file or create one if not there

Change the classes to the number of classes you want to train on
Give the paths to the train, validation files that you created above relative to the root of the YOLOv4 directory
Specify the backup folder (where all the weights will be stored)
(For the example above :
A sample obj.data file will look like:
classes = 2
train = ../train.txt
valid = ../val.txt
names = data/obj.names
backup = ./backup
)

Now modify the obj.names file or create one if not there

Add the names of the classes of your dataset to this file

Run
```
cd ..
```
## 8) The general training command is: 
```
./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show
```
Here if you named the files according to the above procedure then,
```
./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show
```

## 10) In order to see the metrics of your training:
```
./darknet detector map data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_1000.weights
```

## 11) Access the weights file

The weights file are present in the 
./backup folder

## 12) Convert the darknet weights to tensorflow weights that can run with this repo

Now, currently the weights you have are darknet weights and in order for you to use them in this repo, you need to convert them to tensorflow weights

There is a folder in this directory named:
```
Convert_Darknet_YOLO_to_TensorFlow
```

Copy the weights in YOLOv4/backup to Convert_Darknet_YOLO_to_TensorFlow/data

Run
```
cd Convert_Darknet_YOLO_to_TensorFlow
```

Change the save_model.py a bit as:

Modify 
```
flags.DEFINE_string('weights', './data/<Path to weights file>', 'path to weights file')
```
and 
```
flags.DEFINE_string('output', '../../Weights/<output folder>', 'path to output')
```

# For curriculum training:

The current Motorcycle-Violations-Complete/Training/Models looks like:
```
Motorcycle-Violations-Complete/Training/Models:
    YOLOv4
    train.txt
    val.txt
    final_test_set M_R_H_NH_after_preprocessing:
        img1.jpg
        img1.txt
        img2.jpg
        img2.txt
        .
        .
        .
    M_R_H_NH:
        val1.jpg
        val1.txt
        val2.jpg
        val2.txt
        .
        .
        .
```

Now you need to change all the img*.txt files corresponding to the annotations for the first round of curriculum training
(For ex-> all the annotations to only the motorcycle annotations)

In context of the IDD Dataset, a helper script has been put in this repo, that will do just that:

1) Copy the helper script in the train_data and the val_data folder

2) Run :
```
cd training_data_817Images
mkdir all_class_labels
mkdir motor_class
mkdir rider_motor_class
cp final_test_set M_R_H_NH_after_preprocessing/*.txt all_class_labels
cp all_class_labels/* motor_class
cp all_class_labels/* rider_motor_class
```
Run corresponding commands for the val folder as well

3)Change the path name at the beggining of class_utils.py to 

path = "motor_class"

You also need to change lines
```
13, 15, 18, 20
```
For only motor annotations we want
```
line 13 -> if(words[0] == '3'):
line 15 -> new_line[0] = '1'
line 18 -> if(words[0] == '5'):
line 20 -> new_line[0] = '2'
```
4) Run :
```
python3 class_utils.py
```
5) Run
```
cp motor_class/* final_test_set M_R_H_NH_after_preprocessing
```
This will change the annotations to just that of motor_class

You need to switch the training weights after the first step of training is over for curriculum training.

After the training is over the weights will be in backup folder
Now for curriculum training you need to change the annotations for your train and val dataset. You dont need to modify the train.txt and val.txt files. 

For the second round,

Change the class_utils.py again

    path = "rider_motor_class" 

You also need to change lines
```
13, 15, 18, 20
```
For rider-motor annotations we want
```
line 13 -> if(words[0] == '3'):
line 15 -> new_line[0] = '1'
line 18 -> if(words[0] == '0'):
line 20 -> new_line[0] = '0'
```
Run
```
python3 class_utils.py
```
and 
```
cp rider_motor_class/* final_test_set M_R_H_NH_after_preprocessing
```
and train again normally as given in the instructions above

# Helmet No-Helmet Training

The procedure for training on extracted ROI's is as follows:

1) Open roi_extract.py and change the img_folder (absolute path) and txt_folder path (absolute path) to the ones containing training or validation images.
2) Make sure you have labels corresponding to all_class_labels in the folder you mention here.
3) Change roi_folder to point to the path (absolute path) where you want to save the extracted ROIs and the corresponding labels.
4) Run
   ```
   python3 extract.py
   ```
5) The extracted labels and images will be present in the folder specified in roi_folder
6) You need to create a train.txt and a val.txt for the training.
7) Refer to step 5 above on how to do that
8) Follow the rest of the steps (5, ...) to train the model.
   
# YOLOv5

The YOLOv5 repo is already here

Run

```
cd yolov5
```

#### Assuming you have already downloaded the dataset (Refer to step 5 of YOLOv4 training)
We need to change data/idd.yaml
Add the absolute paths of train.txt and val.txt to this file 
and change the classes section according to your custom dataset.

Run 
```
python3 train.py --data idd.yaml --weights yolov5s.pt
```
for training. 

The weights will be stored in runs folder in the directory (if not already present then the directory will be created).
For curriculum learning refer to the same instructions above in the curriculum learning section





