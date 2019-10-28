#!/usr/bin/env python
from __future__ import print_function
#ROS staff
import roslib
import sys
import os
import numpy as np
import rospy
from std_msgs.msg import String, Header, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from atg_autoencoder_mixture.srv import *
from atg_autoencoder_mixture.msg import AspectNodes
#pytorch stuff
from common import *
from random_transformer import random_transformer
from models import *
from utils import *

from data_loader import *



NODE_NAME = 'aspect_transition_graph'

class AspectTransitionGraph:


    def __init__(self, rate=0.5):

        rospy.init_node(NODE_NAME, anonymous=False)
        rospy.wait_for_service('current_observation')
        self.atg_pub = rospy.Publisher("/atg", Float64MultiArray, queue_size=10)
        self.aspect_nodes_pub = rospy.Publisher("/aspect_nodes", AspectNodes, queue_size=10)

        self.action_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.bridge = CvBridge()
        self.observation_count = 0
        self.action_count = 0
        self.get_current_observation = rospy.ServiceProxy('current_observation', CurrentObservation)
        self.rate = rospy.Rate(rate)

        self.aspect_count = 0
        self.autoencoder_mixture = {}
        self.autoencoder_mixture[self.aspect_count] = {}
        self.autoencoder_mixture[self.aspect_count]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
        self.autoencoder_mixture[self.aspect_count]['recon_error'] = 0
        self.aspect_images = []
        self.atg = {}
        self.prev_aspect_node = None
    def build_atg(self, obs_save_dir=None):
        action = 0
        while not rospy.is_shutdown():

            ros_image = self.get_current_observation().img
            cv_image = self.bridge.imgmsg_to_cv2(ros_image)
            if NEED_CROP:
                cv_image = cv_image[CROP_I:CROP_I+CROP_H, CROP_J:CROP_J + CROP_W]
            if obs_save_dir is not None:
                obs_save_dir_full = '/home/daniel/Desktop/atg/catkin_ws/src/atg_autoencoder_mixture/' + obs_save_dir
                if not os.path.exists(obs_save_dir_full):
                    os.makedirs(obs_save_dir_full)

                write_loc = os.path.join(obs_save_dir_full, 'obs_' + str(self.observation_count) + '.jpg')
                cv2.imwrite(write_loc , cv_image)
            image = to_var(cv_to_tensor(cv_image, image_size=IMAGE_SIZE))
            recon_loss, recon_loss_norm = get_reconstruction_loss_with_all_ae(image,
                                                                 self.autoencoder_mixture,
                                                                 loss_fn = torch.nn.functional.mse_loss)

            c_aspect_node = current_aspect_node(recon_loss, self.aspect_count)

            atg_mat_fma = Float64MultiArray()
            if recon_loss_norm.min() < RECONSTRUCTION_TOLERANCE:
                print('[Observation: %d  was matched with Aspect node: %d]'%(self.observation_count, c_aspect_node))
            else:




                print('Observation: %d not matched hence Creating aspect_%d'%(self.observation_count, self.aspect_count))
                print()
                self.aspect_images.append(tensor_to_ros(image.data, self.bridge))

                self.autoencoder_mixture[self.aspect_count] = {}
                self.autoencoder_mixture[self.aspect_count]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
                gen_images = generate_random_versions_of_image(image.squeeze(0), random_transformer, n_versions=300)
                ds = AutoEncoderDataset(gen_images, aspect_image=image)
                optimizer = optim.Adam(self.autoencoder_mixture[self.aspect_count]['autoencoder'].parameters(), lr=1e-3)
                criterion = nn.BCELoss()
                data_loader = DataLoader(ds, batch_size=4, shuffle=True)

                train_autoencoder(self.autoencoder_mixture[self.aspect_count]['autoencoder'],
                                  optimizer,
                                  criterion,
                                  data_loader,
                                  number_of_epochs=5,
                                  name='aspect_autoencoder_' + str(self.aspect_count), verbose=True)

                test_image = to_var(gen_images[0].unsqueeze(0))
                test_image_recon = self.autoencoder_mixture[self.aspect_count]['autoencoder'](test_image)

                recon_error = torch.nn.functional.mse_loss(test_image_recon, test_image)
                self.autoencoder_mixture[self.aspect_count]['recon_error'] = recon_error.data.sum()

                #fig=plt.figure(figsize=(18, 16), dpi= 100, facecolor='w', edgecolor='k')
                #imshow(make_grid(test_recon_and_image), True)
                self.aspect_count += 1

            if self.prev_aspect_node is not None:
                atg_entry_key = (self.prev_aspect_node, action,  c_aspect_node)


                if atg_entry_key in self.atg.keys():
                    self.atg[atg_entry_key] += 1
                else:
                    self.atg[atg_entry_key] = 1

                print('ATG: ', self.atg, self.aspect_count)
                atg_mat = atg_dict_to_mat(self.atg, self.aspect_count, 6)
                print(atg_mat)
                atg_mat_layout = MultiArrayLayout()
                dim1 = MultiArrayDimension()
                dim2 = MultiArrayDimension()
                dim3 = MultiArrayDimension()
                dim1.label = 's'
                dim1.size  = atg_mat.shape[0]

                dim2.label = 'a'
                dim2.size  = atg_mat.shape[1]

                dim3.label = 's_prime'
                dim3.size  = atg_mat.shape[2]

                dims = [dim1, dim2, dim3]

                atg_mat_layout.dim = dims
                atg_mat_layout.data_offset = 0

                atg_mat_fma.layout = atg_mat_layout
                atg_mat_fma.data = atg_mat.reshape(1, -1).tolist()[0]

            aspect_nodes = AspectNodes()
            aspect_nodes.data = self.aspect_images

            self.aspect_nodes_pub.publish(aspect_nodes)
            self.atg_pub.publish(atg_mat_fma)


            self.observation_count += 1
            self.prev_aspect_node = c_aspect_node
            action = simulate_action()
            print('taking action %d'%(action))
            self.rate.sleep()
            print('Done taking action')


def main(args):
    at = AspectTransitionGraph(rate=1)

    try:
        at.build_atg('data/hia')
    except KeyboardInterrupt:
        print("Shutting down")
if __name__ == '__main__':
    main(sys.argv)
