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
    def __init__(self,type=1):
        #Type: 0 for human, 1 for rats, 2 for mice
        Type=str(type)
        self.Len = {  # Column length on um [Defelipe et al. 2002] doi:10.1023/A:1024130211265
            '0': 2622,  # human
            '1': 1827,  # rat
            '2': 1210,  # mouse
        }
        #Thickness of each layer in um [Defelipe et al. 2002] doi:10.1023/A:1024130211265
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

        self.L4_d={
            '0':285,
            '1':152,
            '2':208,
        }

        self.L5_d={
            '0':552,
            '1':321+209,
            '2':248,
        }

        self.L6_d={
            '0':480,
            '1':565,
            '2':451,
        }
        # density of neurons in the different layers in neurons/mm3 [[Defelipe et al. 2002] doi:10.1023/A:1024130211265]

        self.Dens1={
            '0': 8333,
            '1': 3472,
            '2': 18229,
        }

        self.Dens23={
            '0': 27205,
            '1': 61670,
            '2': 137645,
        }

        self.Dens4={
            '0': 46167,
            '1': 90965,
            '2': 181362,
        }

        self.Dens5={
            '0':  23076,
            '1': 40202,
            '2': 77765,
        }

        self.Dens6={
            '0':  16774,
            '1': 64286,
            '2': 122092,
        }



        self.Syn_NB={
            '0': 29807,
            '1':18018,
            '2':21133,
        }
        # column radius in um
        self.R={
            '0':300,
            '1':210,
            '2':210,
        }
        #column length
        self.L=self.Len[Type]
        #Column diameter
        self.D=2*self.R[Type]
        #thickness of each layer of the column
        self.L_th=np.array([self.L1_d[Type], self.L23_d[Type], self.L4_d[Type], self.L5_d[Type], self.L6_d[Type]])
        #Number of synapses per cell for each column
        self.NB_Syn =self.Syn_NB[Type]
        #cell density in each layer
        self.Celldens=np.array([self.Dens1[Type], self.Dens23[Type], self.Dens4[Type], self.Dens5[Type], self.Dens6[Type]])*1e-9 #in neurons/um3
        #xortical column volume
        self.Volume=self.L_th*np.pi*self.R[Type]*self.R[Type]
        #cell number in each layer
        self.Layer_nbCells=self.Celldens* self.L_th*np.pi*self.R[Type]*self.R[Type]
        ##reduced numbers for simulation
        self.Layer_nbCells=self.Layer_nbCells/30
        #from Markram et al. 2015 rat model
        self.PYRpercent = np.array([0, 0.7, 0.9, 0.8, 0.9])
        self.INpercent = 1 - self.PYRpercent
        self.PVpercent = np.array([0, 0.3, 0.55, 0.45, 0.45])
        self.SSTpercent = np.array([0, 0.2, 0.25, 0.40, 0.35])
        self.VIPpercent = np.array([0, 0.40, 0.20, 0.15, 0.20])
        self.RLNpercent = np.array([1, 0.1, 0, 0, 0])
        self.GetCelltypes()
        # self.Layer_nbCells = np.array([322,7524, 4656, 6114,
        #                           12651]) / d  # markram et al. 2015total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)


        self.List_Lambda_s= np.array(
            [[0.032, 0, 0, -0.032, 0],  # layer II/III-> TPC,UPC,IPC,BPC,SSC
             [0.048, 0.048, 0, 0, -0.012],  # layer IV-> TPC,UPC,IPC,BPC,SSC
             [0.137, 0.048, 0, 0, 0],  # layer V-> TPC,UPC,IPC,BPC,SSC
             [0.117, 0, -0.104,0.024, 0]  # layer VI-> TPC,UPC,IPC,BPC,SSC
             ], dtype=np.float)

        self.List_Lambda_d= np.array(
            [[-0.120, 0, 0,0.120, 0],  # layer II/III-> TPC,UPC,IPC,BPC,SSC
             [-0.095, -0.095, 0, 0, 0],  # layer IV-> TPC,UPC,IPC,BPC,SSC
             [-0.071, -0.076, 0, 0, 0],  # layer V-> TPC,UPC,IPC,BPC,SSC
             [-0.046, 0, 0.047,0.019, 0]  # layer VI-> TPC,UPC,IPC,BPC,SSC
             ], dtype=np.float)

        self.PCsubtypes_Per=np.array([[0,0,0,0,0],[0.9*self.NB_PYR[1],0,0.1*self.NB_PYR[1],0,0],[0.5*self.NB_PYR[2],0.36*self.NB_PYR[2],0,0,0.14*self.NB_PYR[2]],[self.NB_PYR[3]*0.81,self.NB_PYR[3]*0.19,0,0,0],[self.NB_PYR[4]*0.39,self.NB_PYR[4]*0.17,self.NB_PYR[4]*0.20,self.NB_PYR[4]*0.24,0]]) #TPC,UPC,IPC,BPC,SSC
        self.List_celltypes = np.array([np.array([0]*self.Layer_nbCells_pertype[0][l] + [1]*self.Layer_nbCells_pertype[1][l] + [2]*self.Layer_nbCells_pertype[2][l] + [3]*self.Layer_nbCells_pertype[3][l] + [4]*self.Layer_nbCells_pertype[4][l]).astype(int) for l in range(len(self.Layer_nbCells))])


    def GetCelltypes(self):
        NBPYR = self.PYRpercent * self.Layer_nbCells
        self.NB_PYR = NBPYR.astype(int)
        NB_IN = self.INpercent * self.Layer_nbCells
        self.NB_IN = NB_IN.astype(int)
        NB_PV = self.PVpercent * self.NB_IN
        self.NB_PV = NB_PV.astype(int)
        NB_PV_BC = 0.7 * self.NB_PV
        self.NB_PV_BC = NB_PV_BC.astype(int)
        self.NB_PV_ChC = (NB_PV - NB_PV_BC).astype(int)
        NB_SST =self.SSTpercent * self.NB_IN
        self.NB_SST = NB_SST.astype(int)
        NB_VIP = self.VIPpercent * self.NB_IN
        self.NB_VIP = NB_VIP.astype(int)
        NB_RLN = self.RLNpercent * self.NB_IN
        self.NB_RLN = NB_RLN.astype(int)
        self.Layer_nbCells = self.NB_PYR+self.NB_SST+self.NB_PV+self.NB_VIP+self.NB_RLN  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)
        self.Layer_nbCells_pertype=[self.NB_PYR,self.NB_PV,self.NB_SST,self.NB_VIP,self.NB_RLN]



if __name__ == '__main__':
    Column = Column(type=0)





