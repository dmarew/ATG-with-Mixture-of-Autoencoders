#!/usr/bin/env python
from __future__ import print_function
#ROS staff
import roslib
import sys
import os
import numpy as np
import rospy
import cv2
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from atg_autoencoder_mixture.srv import *

#pytorch stuff
from common import *
from random_transformer import random_transformer
from models import *
from utils import *




NODE_NAME = 'aspect_transition_graph'

class AspectTransitionGraph:


    def __init__(self, rate=0.5):

        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.wait_for_service('current_observation')
        self.action_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.bridge = CvBridge()
        self.observation_count = 0
        self.action_count = 0
        self.get_current_observation = rospy.ServiceProxy('current_observation', CurrentObservation)
        self.rate = rospy.Rate(rate)

        self.aspect_count = 0
        self.autoencoder_mixture = {}
        self.autoencoder_mixture[aspect_count] = {}
        self.autoencoder_mixture[aspect_count]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
        self.autoencoder_mixture[aspect_count]['recon_error'] = 0
        self.final_aspect_nodes = []

    def build_atg(self, obs_save_dir=None, obs_rate = 0.5):

        while not rospy.is_shutdown():

            ros_image = self.get_current_observation().img
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

            recon_loss, recon_loss_norm = get_reconstruction_loss_with_all_ae(cv_image,
                                                                 self.autoencoder_mixture,
                                                                 loss_fn = torch.nn.functional.mse_loss)

            if obs_save_dir is not None:
                obs_save_dir_full = '/home/daniel/Desktop/atg/catkin_ws/src/atg_autoencoder_mixture/' + obs_save_dir
                if not os.path.exists(obs_save_dir_full):
                    os.makedirs(obs_save_dir_full)

                write_loc = os.path.join(obs_save_dir_full, 'obs_' + str(self.observation_count) + '.jpg')
                cv2.imwrite(write_loc , cv_image)
            self.observation_count += 1
            self.rate.sleep()

def main(args):
    at = AspectTransitionGraph()

    try:
        at.build_atg('data/testing')
    except KeyboardInterrupt:
        print("Shutting down")
if __name__ == '__main__':
    main(sys.argv)
