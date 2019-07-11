# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import pandas as pd
import os.path
import pickle
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences

# Initialize the parameters
#confThreshold = 0.5  #Confidence threshold
#nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "classes.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "/home/santoshganti/learnopencv/YOLOv3-Training-Snowman-Detector/darknet-yolov3.cfg";
modelWeights = "/home/santoshganti/learnopencv/YOLOv3-Training-Snowman-Detector/weights/darknet-yolov3_final.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def draw_max(max_conf,my_det,my_classId,frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
     #print("my_det ", my_det)
    if len(my_det) == 0:
        return [0, 0, 0, 0, 0]
    center_x = int(my_det[0] * frameWidth)
    center_y = int(my_det[1] * frameHeight)
    width = int(my_det[2] * frameWidth)
    height = int(my_det[3] * frameHeight)
    left = int(center_x - width / 2)
    top = int(center_y - height / 2)
    classIds.append(my_classId)
    confidences.append(float(max_conf))
    #boxes.append([left, top, width, height])
    
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    #indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    drawPred(classIds[0], confidences[0], left, top, left + width, top + height)
    return[classIds[0], left, top, left + width, top + height]
# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, object_class):
    #print ("outs :", outs)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    max_conf = 0
    my_det = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            #print(classId)
            #print("object class:",object_class)
            if (object_class==classId and confidence > max_conf):
                #print("confidence ", confidence)
                max_conf = confidence
                my_det = detection 
    return [max_conf,my_det,object_class,frame]
    

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
outputFile = "yolo_out_py.avi"

txtFile = open(args.image,'r')
txt = txtFile.readlines()

label_line = txt.pop(0)

new_text = label_line.split(' ')

with open('/home/santoshganti/ai/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(new_text)
text_input = []
for i in range(len(sequences)):
    text_input.extend(sequences[i])

text_input = [text_input]
MAX_SEQUENCE_LENGTH=16
data = pad_sequences(text_input, maxlen=MAX_SEQUENCE_LENGTH)

model = load_model('/home/santoshganti/ai/glovecnn.h5')
object_class = np.argmax(model.predict(data))
print("object",object_class)
path=args.image[:-4]
list_main=[]
count=0
max_confidance=0
max_conf_list=[]
out_dir=''
counter=0
for img in txt:
    counter=counter+1
    img=img[:-1]
    image_link = str(img)
    image_name=image_link.split("/")[-1][:-4]
    
    if (args.image):
        # Open the image file
        if not os.path.isfile(image_link):
            print("Input image file ", image_link, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(image_link)
        outputFile = path+"/"+image_name+'_yolo_out_py.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_yolo_out_py.avi'
    else:
        # Webcam input
        cap = cv.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if (not args.image):
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            #print("Done processing !!!")
            #print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        
        list1 = postprocess(frame, outs,object_class)
        if(list1[0]>max_confidance):
            max_conf_list=list1
            out_dir=outputFile
            
        print("length",out_dir)
        if(counter==len(txt)):
            if(len(max_conf_list)!=0):
                print("in write ",out_dir)
                draw_max(max_conf_list[0],max_conf_list[1],max_conf_list[2],max_conf_list[3])
                t, _ = net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                cv.imwrite(out_dir, max_conf_list[3].astype(np.uint8))
                cv.imshow(winName, max_conf_list[3])
            else:
                print("No Image with given discription")
                
                
                
'''if( len(list_main)  %10 == 0):
            df = pd.DataFrame(list_main)
            print("done")
            df.to_csv('out.csv')

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8));
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)


df = pd.DataFrame(list_main)
print("done")
df.to_csv('out.csv')'''


