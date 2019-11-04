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

VERBOSE =  True
ATG_NODE_RATE = 0.5
RECONSTRUCTION_TOLERANCE = 0.01
NUMBER_OF_EPOCHS = 10
CONVERGENCE_THRESHOLD = 20
MAX_QUEUE_SIZE = 10
IMAGE_SIZE = 64
NEED_CROP = True
ACTION_SELECTION_MODE = 'RANDOM'
EXPERIMENT_NAME = 'cup_random'
CUDA_VAILABLE = torch.cuda.is_available()
ACTION_PARAMETER_SPACE = np.arange(-2.61, 2.61, .785)
NUM_ACTIONS = len(ACTION_PARAMETER_SPACE)
print('ACTION_PARAMETER_SPACE: ', ACTION_PARAMETER_SPACE)
