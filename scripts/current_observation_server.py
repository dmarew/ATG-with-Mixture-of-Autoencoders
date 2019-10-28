#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import os
import numpy as np
import rospy
import cv2
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from atg_autoencoder_mixture.srv import CurrentObservation, CurrentObservationResponse

class CurrentObservationServer:
    NODE_NAME = "current_observation"

    def __init__(self):
        rospy.init_node(self.NODE_NAME)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.service = rospy.Service('current_observation', CurrentObservation, self.get_current_observation)
        self.image = None

    def callback(self,data):
        self.image = data
    def get_current_observation(self, request):
        return CurrentObservationResponse(self.image)

if __name__=="__main__":
	frs = CurrentObservationServer()
	try:
		rospy.spin()
	except:
		rospy.loginfo("error!!")
