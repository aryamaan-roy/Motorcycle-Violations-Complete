# get 4 class MAP given the ground truth annotations and the predicted bounding boxes, scores and classes in yolo format
# the ground truth annotations are in the format: [class, x, y, w, h]
# the predicted bounding boxes are in the format: [x, y, w, h, score, class]
# the classes are : 0,1,2,3
# the scores are between 0 and 1
# the MAP is calculated for each class and then averaged over the 4 classes

def getMAP(gt, pred):
    # gt is a list of lists of lists of the form: [class, x, y, w, h]
    # pred is a list of lists of lists of the form: [x, y, w, h, score, class]
    # the classes are : 0,1,2,3
    # the scores are between 0 and 1
    # the MAP is calculated for each class and then averaged over the 4 classes

    # get the number of images
    num_images = len(gt)

    # get the number of classes
    num_classes = 4

    # initialize the APs for each class
    APs = [0] * num_classes

    # loop over the classes
    for c in range(num_classes):
        