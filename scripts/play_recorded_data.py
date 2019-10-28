#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import os
import glob
import numpy as np
import rospy
import cv2
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from atg_autoencoder_mixture.srv import CurrentObservation, CurrentObservationResponse

from common import *
from utils import *
class RecordedObservationServer:
    NODE_NAME = "RecordedObservation"

    def __init__(self, record_path, size_of_dataset=1353):
        rospy.init_node(self.NODE_NAME)

        self.bridge = CvBridge()
        self.service = rospy.Service('recorded_observation', CurrentObservation, self.get_current_observation)
        self.current_obs_index = 1
        self.record_path = record_path
        self.size_of_dataset =size_of_dataset
    def get_current_observation(self, request):
        if self.current_obs_index > self.size_of_dataset:
            self.current_obs_index = 1
            print('REPLAY........')
        current_image_path  = self.record_path + str(self.current_obs_index) + '.jpg'
        image = Image.open(current_image_path)
        image  = cv_to_ros(pil_to_cv(image), self.bridge)
        self.current_obs_index +=1
        print(current_image_path)
        return CurrentObservationResponse(image)

if __name__=="__main__":
    record_path = '/home/daniel/Desktop/atg/catkin_ws/src/atg_autoencoder_mixture/data/real_aspects/Aspect-Raw'
    frs = RecordedObservationServer(record_path, size_of_dataset=1353)
    try:
        rospy.spin()
    except:
        rospy.loginfo("error!!")
