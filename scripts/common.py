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

ATG_NODE_RATE = 0.5
RECONSTRUCTION_TOLERANCE = 0.01
CONVERGENCE_THRESHOLD = 10
IMAGE_SIZE = 128
NEED_CROP = True
CUDA_VAILABLE = torch.cuda.is_available()
ACTION_PARAMETER_SPACE = np.arange(-2.61, 2.61, .785)
print('ACTION_PARAMETER_SPACE: ', ACTION_PARAMETER_SPACE)
