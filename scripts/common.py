import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
import time
import pickle
import argparse
import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import scipy.stats
VERBOSE =  True
USE_ASPECT_IMAGE = False
VIZ_GEN_IMAGES  = False
ATG_NODE_RATE = 0.5
RECONSTRUCTION_TOLERANCE = 0.01
RECONSTRUCTION_TOLERANCE_LL = 0.1
NUMBER_OF_EPOCHS = 10
N_VERSIONS=128
COLLECT_DATA = False
NUM_DATA_POINTS = 50
CONVERGENCE_THRESHOLD = 50
MAX_QUEUE_SIZE = 10
IMAGE_SIZE = 128
NEED_CROP = True
ACTION_SELECTION_MODE = 'RANDOM'
EXPERIMENT_NAME = 'can_online_testing'
NUMBER_OF_EXPERIMENTS = 1
CUDA_VAILABLE = torch.cuda.is_available()
ACTION_PARAMETER_SPACE = np.arange(-2.61, 2.61, .785)
NUM_ACTIONS = len(ACTION_PARAMETER_SPACE)
print('ACTION_PARAMETER_SPACE: ', ACTION_PARAMETER_SPACE)
