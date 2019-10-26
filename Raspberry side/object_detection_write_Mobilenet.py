# coding: utf-8

# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
from PIL import Image
print('done')
# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
# Import utilites
from utils import label_map_util
freq = cv2.getTickFrequency()
PATH_TO_CKPT = './frozenpb/frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = './data/BDD100k_label_map.pbtxt'

# Number of classes the object detector can identify
NUM_CLASSES = 10
txtpath = '../Myssdlite/outputtxt4_22/'
imgpath = '/home/pi/Pictures/img/'
## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
def load_image_into_numpy_array(image):
  (w,h)=image.size
  return np.array(image.getdata()).reshape(
      (h, w, 3)).astype(np.uint8)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
t1 = cv2.getTickCount() 
#print(t1)
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

t2 = cv2.getTickCount()
#print(t2)
#print((t2-t1)/freq,'\n')
# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

### Picamera ###
while(True):
    # Initialize Picamera and grab reference to the raw capture
    '''
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)
    '''
    for image_path in os.listdir(imgpath):
      path1 = imgpath+image_path
      print(path1)
      t1 = cv2.getTickCount()  
      image = Image.open(path1)

      frame = load_image_into_numpy_array(image)

    
      # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
      # i.e. a single-column array, where each item in the column has the pixel RGB value
      #frame = np.copy(frame.array)
      #frame.setflags(write=1)
      frame_expanded = np.expand_dims(frame, axis=0)

      t3 = cv2.getTickCount()
      time2 = (t3-t1)/freq
      #print(time2)
      # Perform the actual detection by running the model with the image as input
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})
      
      i=0
      car = 0
      bus = 0
      truck = 0
      #print(classes,scores)
      #print(scores[0])
      #frame_rate_calc = 1/time1
      f = open('/home/pi/Desktop/result.txt','w')
      while(i<len(classes[0])):
        #print(scores[i])
        if scores[0][i] > 0.5:
            if classes[0][i]==1:
                car+=1
            elif classes[0][i]==5:
                truck+=1
            elif classes[0][i]==7:
                bus+=1
            
        i = i + 1
      # Press 'q' to quit
      f.write(image_path+' '+str(car)+' '+str(truck)+' '+str(bus))
      f.close()
      t2 = cv2.getTickCount()
      time1 = (t2-t3)/freq
      #print(time1)
