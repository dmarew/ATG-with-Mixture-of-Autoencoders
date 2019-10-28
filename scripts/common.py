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

import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

RECONSTRUCTION_TOLERANCE = 0.015
IMAGE_SIZE = 64
NEED_CROP = True
CROP_I = 250
CROP_J = 300
CROP_H = 100
CROP_W = 80
