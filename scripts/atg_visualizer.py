#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import os
import numpy as np
import rospy
import cv2
from std_msgs.msg import String, Header, Float64MultiArray
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.patches as patches

from common import *
from utils import *
from atg_autoencoder_mixture.msg import AspectNodes

NODE_NAME = 'atg_visualizer'

class ATGVisulizer:

    def __init__(self, fig):
        self.fig = fig
        rospy.init_node(NODE_NAME)
        self.aspect_nodes = None
        self.bridge = CvBridge()
        rospy.Subscriber("/atg", Float64MultiArray, self.atg_callback)
        rospy.Subscriber("/aspect_nodes", AspectNodes, self.aspect_nodes_callback)

    def atg_callback(self, data):
        plt.clf()
        if len(data.data)==0:return


        atg_shape = (data.layout.dim[0].size, data.layout.dim[1].size, data.layout.dim[2].size)
        atg_mat = np.array(data.data).reshape(atg_shape)
        atg_n_states = atg_shape[0]
        n_nodes = len(self.aspect_nodes)
        if self.aspect_nodes is not None:
            G, pos, labels, num_edges = generate_atg_graph(atg_mat, self.aspect_nodes, self.bridge)
            G.graph['edge'] = {'arrowsize': '10', 'splines': 'curved'}
            G.graph['graph'] = {'scale': '3'}

            ax=plt.subplot(111)
            ax.set_aspect('equal')
            #nx.draw_networkx_edges(G, pos, width=15.0, alpha=1., ax=ax)

            if(n_nodes>1):
                #edge_weights = []
                #edge_list = []
                '''
                for edge in nx.generate_edgelist(G, data=['weight']):
                    edge_ = edge.split(' ')
                    node_1, node_2, edge_weight = edge_[0], edge_[1], int(edge_[2])
                    edge_weights.append(edge_weights)
                    edge_list.append([node_1, node_2])
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=[edge_weights], ax=ax, edge_cmap=plt.cm.Blues)

                '''
                edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)

            trans=ax.transData.transform
            trans2=self.fig.transFigure.inverted().transform

            piesize=0.2 # this is the image size
            p2=piesize/2.0
            for n in G:
                xx,yy=trans(pos[n]) # figure coordinates
                xa,ya=trans2((xx,yy)) # axes coordinates
                a = plt.axes([xa-p2,ya-p2, piesize, piesize])
                a.set_aspect('equal')
                im = a.imshow(G.node[n]['image'])
                #im = ax.imshow(image)
                patch = patches.Circle((IMAGE_SIZE//2, IMAGE_SIZE//2), radius=IMAGE_SIZE//2-4, transform=a.transData)
                im.set_clip_path(patch)
                a.axis('off')
            ax.axis('off')

            #for n in range(n_nodes):

            #nx.draw_networkx_nodes(G,pos)
            #nx.draw_networkx_labels(G,pos,labels,font_size=16)
            #nx.draw_networkx_edges(G,pos, arrowstyle='->', arrowsize=10, edge_colors = range(2, num_edges + 2),
            #                   edge_cmap=plt.cm.Blues, width=2)

        plt.draw_all()

    def aspect_nodes_callback(self, data):
        self.aspect_nodes = data.data


def main(args):
    fig = plt.figure()
    at = ATGVisulizer(fig)

    plt.title('Aspect Transition Graph')
    plt.axis('off')
    plt.show(block=True)
if __name__ == '__main__':
    main(sys.argv)
