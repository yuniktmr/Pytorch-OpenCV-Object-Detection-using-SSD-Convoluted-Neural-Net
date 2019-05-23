# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:38:53 2019
@author: Yunik
"""
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#defining a function for detections

#1. frame by frame detection

def detect(frame, net, transform):
    height, width = frame.shape[:2] #0 = height, 1 = width, 2 represents img channel
    #apply transfromation 1
    frame_t = transform (frame)[0] 
#convert numpy array to torch tensor
    x = torch.from_numpy(frame_t).permute(2,0,1)
#permute/ transformation to convert rbg to grb (nerural net specific)
#then add fake dimension to batch (net only accept batches)
#unsqueezed function (0 for batch)
    x = Variable(x.unsqueeze(0))  #the pytorch Variable is used in dynamic graphs whcih makes backward prop easier
    y = net(x) #feed data to network
#create new tensor containing required values
    detections = y.data
#scale tensor with dimensions w,h (scale of values in upper left of rectangle ),w,h (for lower right): for normalization betweeen 0 and 1.
    scale = torch.Tensor([width, height, width, height])
#detections Tensor contains detections = [batch, number of classes (objects to be detected), no of occurances of the class, (score, x0, y0, x1, y1) ]
    #for each occurence we got score and coordinates of upper left and lower right corner; if score>0.6 object is considered to be found, else not found
    for i in range (detections.size(1)):
        j = 0
        while detections[0, i ,j, 0] >= 0.6: #last 0 is index of the score
            pt = (detections[0 , i, j, 1:] * scale).numpy()
            #rectangle works with and openCv works with nmupy arrays. So, convert to numpy
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            j+=1
    return frame

#creating the SSD neural net
net = build_ssd('test') #args: trained or test phase
#load wts of already pretained net
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = 'cpu')) #torch.load opens tensor function containing the wts        

#supply the parameters for the detect function
#transform the stuff as fit for the net
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) #net.size : target size of imges to feed to net, 2nd argument is triplet of numbers enabling col value at right scale i.e the scale in which net was trained

#doing obj detection on the test video

#1. open the video 
reader = imageio.get_reader('funny_dog.mp4')

#get the fps frequency
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect (frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()        
            
            
