# Steps for training on YOLOv4 : 

#### Additional instructions for Curriculum learning are given at the end. Please read them before executing any of the normal commands as well

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
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
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
    training_data_817Images:
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
cd train_data
readlink -f *.jpg > ../train.txt
cd ..
cd val_data
readlink -f *.jpg > ../val.txt
```
Now in your home directory you will have 2 files 
train.txt
val.txt
that you will need in the future steps

(For curriculum training, please refer to the end)

## 6) Run the following command
```
cp cfg/yolov4-custom.cfg cfg/yolov4-obj.cfg
```
This is the configuration file for training where you specify the network architecture details and also the hyperparameters for training.

You need to change the file according to the number of classes you have

If you have 'c' classes
then in the configuration file you need to set 
filters = (c + 5) * 3
Note that there are multiple filters keyword in the config file and you need to change all of them. 

## 7) The next step is to edit the files in the data directory

Run
```
cd data
```
Now modify the obj.data file

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

Now modify the obj.names file

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


# For curriculum training:

The current Motorcycle-Violations-Complete/Training/Models looks like:
```
Motorcycle-Violations-Complete/Training/Models:
    YOLOv4
    train.txt
    val.txt
    training_data_817Images:
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
cp training_data_817Images/*.txt all_class_labels
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
