__author__ = 'Mariam'

# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import numpy as np
import math
import scipy
import scipy.io as sio
from scipy.spatial import distance
from scipy import sparse
from scipy import signal
import pickle
import time
import matplotlib
import matplotlib.patches as patches
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import axes3d, Axes3D
import inspect
import copy
import subprocess
import math
import struct
import random
import CC_vtk


#Column morphology for different species: human/rats/mice


class Column:
    def __init__(self,Type=0):
        #Type: 0 for human, 1 for rats, 2 for mice
        type=Type
        self.L = {  # Column length on um [Defelipe et al. 2002] doi:10.1023/A:1024130211265
            '0': 2622,  # human
            '1': 1827,  # rat
            '2': 1210,  # mouse
        }

        self.L1_d={
            '0':235,
            '1':123,
            '2':69,
        }


        self.L23_d={
            '0':295+405+370,
            '1':457,
            '2':235,
        }

        self.L4={
            '0':285,
            '1':152,
            '2':208,
        }

        self.L5={
            '0':552,
            '1':321+209,
            '2':248,
        }

        self.L6={
            '0':480,
            '1':565,
            '2':451,
        }




