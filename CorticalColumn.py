__author__ = 'Mariam, Maxime'

import cProfile
import pstats
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
import pandas as pd
import time
from mpl_toolkits.mplot3d import axes3d, Axes3D
import inspect
import copy
import subprocess
import struct
import random
import CC_vtk
import SST_neo
import PV_neo
import VIP_neo
import RLN_neo
import PC_neo3
#import pyscenarios
import csv
#import pyedflib
import datetime
from numba import guvectorize,vectorize, float64, int64
from numba.experimental import jitclass
from numba import jit, njit, types, typeof
import Cell_morphology
# import circos
#import bigfloat
#bigfloat.exp(5000,bigfloat.precision(100))
from PyQt5.QtCore import *
import matplotlib as plt


@njit
def find_intersection(r1, r2, d):
    rad1sqr = r1 ** 2
    rad2sqr = r2 ** 2

    if d == 0:
        return math.pi * min(r1, r2) ** 2

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * r1 * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * r2 * d)

    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = math.acos(angle1) * 2
        theta2 = math.acos(angle2) * 2
        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))

        return area1 + area2
    elif angle1 < -1 or angle2 < -1:
        return math.pi * min(r1, r2) ** 2
    return 0

class SenderObject(QObject):
    something_happened = pyqtSignal(np.float)

class SenderObjectInt(QObject):
    something_happened = pyqtSignal(np.int)

spec1 = [
    ('PreSynaptic_Cell_AMPA'    , types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Cell_GABA'    , types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_AMPA'    , types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_GABA_d'    , types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_GABA_s', types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_GABA_a', types.List(typeof(np.array([], dtype=np.int32)))),
    ('ExternalPreSynaptic_Cell_AMPA_DPYR'   , types.List(typeof(np.array([], dtype=np.int32)))),
    ('ExternalPreSynaptic_Cell_AMPA_Th'     , types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynapticWeight_AMPA'               , types.List(typeof(np.array([], dtype=np.float64)))),
    ('PreSynapticPos_AMPA', types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynapticWeight_GABA'               , types.List(typeof(np.array([], dtype=np.float64)))),
    ('PreSynapticPos_GABA'               , types.List(typeof(np.array([], dtype=np.int32)))),

]


@jitclass(spec1)
class presynaptic_class():
    def __init__(self, PreSynaptic_Cell_AMPA,
                 PreSynaptic_Cell_GABA,
                 PreSynaptic_Soma_AMPA,
                 PreSynaptic_Soma_GABA_d,
                 PreSynaptic_Soma_GABA_s,
                 PreSynaptic_Soma_GABA_a,
                 ExternalPreSynaptic_Cell_AMPA_DPYR,
                 ExternalPreSynaptic_Cell_AMPA_Th,
                 PreSynapticWeight_AMPA,
                 PreSynapticPos_AMPA,
                 PreSynapticWeight_GABA,
                 PreSynapticPos_GABA):
        self.PreSynaptic_Cell_AMPA = PreSynaptic_Cell_AMPA
        self.PreSynaptic_Cell_GABA = PreSynaptic_Cell_GABA
        self.PreSynaptic_Soma_AMPA = PreSynaptic_Soma_AMPA
        self.PreSynaptic_Soma_GABA_d = PreSynaptic_Soma_GABA_d
        self.PreSynaptic_Soma_GABA_s = PreSynaptic_Soma_GABA_s
        self.PreSynaptic_Soma_GABA_a = PreSynaptic_Soma_GABA_a
        self.ExternalPreSynaptic_Cell_AMPA_DPYR =ExternalPreSynaptic_Cell_AMPA_DPYR
        self.ExternalPreSynaptic_Cell_AMPA_Th = ExternalPreSynaptic_Cell_AMPA_Th
        self.PreSynapticWeight_AMPA = PreSynapticWeight_AMPA
        self.PreSynapticPos_AMPA = PreSynapticPos_AMPA
        self.PreSynapticWeight_GABA = PreSynapticWeight_GABA
        self.PreSynapticPos_GABA = PreSynapticPos_GABA


@njit
def Model_compute(nbEch,
                 dt,
                 tps_start,
                 Layer_nbCells,
                 NB_PYR,
                 NB_PV,
                 NB_SST,
                 NB_VIP,
                 NB_RLN,
                 NB_DPYR,
                 NB_Th,
                 inputNB,
                 List_PYR,
                 List_PV,
                 List_SST,
                 List_VIP,
                 List_RLN,
                 List_DPYR,
                 List_Th,
                 Stim_Signals,
                 Stim_InputSignals,
                 Stim_EField,
                 PS,
                 pyrVs,
                 pyrVd,
                 pyrVa,
                 pyrPPSE_d1,
                 pyrPPSE_d23,
                 pyrPPSE_d4,
                 pyrPPSE_d5,
                 pyrPPSE_d6,
                 pyrPPSI_d1,
                 pyrPPSI_d23,
                 pyrPPSI_d4,
                 pyrPPSI_d5,
                 pyrPPSI_d6,
                 pyrPPSI_s,
                 pyrPPSI_a,
                 PV_Vs,
                 SST_Vs,
                 VIP_Vs,
                 RLN_Vs,
                 DPYR_Vs,
                 Th_Vs,
                 layerS,
                 typeS,
                 indexS,
                 t,
                 seed):

    if not seed == 0:
        np.random.seed(seed)
    # else:
    #     np.random.seed()
    #print(PS.PreSynaptic_Cell_AMPA)
    for i in range(NB_DPYR):  # Distant Pyramidal cells
        List_DPYR[i].dt = dt
    for i in range(NB_Th):  # Thalamus
        List_Th[i].dt = dt

    # initialize PYR cells:
    for i in range(np.sum(NB_PYR)):  # All PYR cells
        List_PYR[i].dt = dt
    for i in range(np.sum(NB_PV)):
        List_PV[i].dt = dt
    for i in range(np.sum(NB_SST)):
        List_SST[i].dt = dt
    for i in range(np.sum(NB_VIP)):
        List_VIP[i].dt = dt
    for i in range(np.sum(NB_RLN)):
        List_RLN[i].dt = dt

    for tt, tp in enumerate(t):
        # if np.mod(tp[tt]*1.,1.01)<dt:
        #     print(tps_start)
        tps_start += dt
        for i in range(NB_DPYR):  ###distant cortex
            List_DPYR[i].init_I_syn()
            List_DPYR[i].updateParameters()
        for i in range(NB_Th):  ###thalamus
            List_Th[i].init_I_syn()
            List_Th[i].updateParameters()

        for i in range(np.sum(NB_PYR)):
            List_PYR[i].init_I_syn()
            List_PYR[i].updateParameters()
        for i in range(np.sum(NB_PV)):
            List_PV[i].init_I_syn()
            List_PV[i].updateParameters()
        for i in range(np.sum(NB_SST)):
            List_SST[i].init_I_syn()
            List_SST[i].updateParameters()
        for i in range(np.sum(NB_VIP)):
            List_VIP[i].init_I_syn()
            List_VIP[i].updateParameters()
        for i in range(np.sum(NB_RLN)):
            List_RLN[i].init_I_syn()
            List_RLN[i].updateParameters()

        ####################### Add stim to external cells #######################
        for i in range(NB_DPYR):
            List_DPYR[i].I_soma_stim = Stim_Signals[i, tt]
        for i in range(NB_Th):
            List_Th[i].I_soma_stim = Stim_InputSignals[i, tt]
        for i in range(np.sum(NB_PYR)):# for EField
            List_PYR[i].update_EField_Stim(Stim_EField[tt])

        ########################################
        curr_pyr = 0
        for i in range(np.sum(Layer_nbCells)):
            nbstim = 0
            nbth = 0
            # print(i)
            #get cell type/layer/index per type
            # layerS, typeS, indexS = self.All2layer(i, Layer_nbCells,NB_PYR, NB_PV, NB_SST, NB_VIP, NB_RLN, List_celltypes)
            neurone = indexS[i]

            ###Get  Cell's Synaptic input
            #print(typeS[i])

            if len(PS.PreSynaptic_Cell_AMPA[i]) > 0 or len(PS.ExternalPreSynaptic_Cell_AMPA_DPYR[i]) > 0 or len(PS.ExternalPreSynaptic_Cell_AMPA_Th[i]) > 0:

                Cell_AMPA = PS.PreSynaptic_Cell_AMPA[i]
                Weight= PS.PreSynapticWeight_AMPA[i]
                if not len(Weight) == 0:
                    # nWeight = (Weight - np.min(Weight)) / (np.max(Weight) - np.min(Weight))
                    nWeight =  Weight / np.max(Weight)
                External_cell_Dpyr = PS.ExternalPreSynaptic_Cell_AMPA_DPYR[i]
                External_cell_Th = PS.ExternalPreSynaptic_Cell_AMPA_Th[i]


                W = np.ones(len(Cell_AMPA) + len(External_cell_Dpyr) + len(External_cell_Th))
                Vpre_AMPA = np.zeros(len(Cell_AMPA) + len(External_cell_Dpyr) + len(External_cell_Th))  # nb of external +internal AMPA inputs
                if not len(Weight)==0:
                    W[0:len(Weight)] = nWeight

                for k, c in enumerate(Cell_AMPA):  # switch case afferences AMPA ---> PC
                    #Get type/layer/ index per type of AMPA inputs == PCs
                    # layer, type, index = All2layer(c,  Layer_nbCells, NB_PYR, NB_PV, NB_SST, NB_VIP, NB_RLN, List_celltypes)
                    Vpre_AMPA[k] = List_PYR[indexS[c]].VAis()

                #add external input
                if not len(External_cell_Dpyr)==0:
                    for k, c in enumerate(External_cell_Dpyr):  # External afferences from DPYR
                        Vpre_AMPA[k+len(Cell_AMPA)] = List_DPYR[c].VAis()
                if not len(External_cell_Th)==0:
                    for k, c in enumerate(External_cell_Th):  # External afferences from DPYR
                        Vpre_AMPA[k+len(Cell_AMPA)+len(External_cell_Dpyr)] = List_Th[c].VAis()


                #switch cell's type to add AMPA input

                if typeS[i] == 0:  # PC
                    # add stim
                    I_AMPA = List_PYR[neurone].I_AMPA2(Vpre_AMPA)*W
                    I_NMDA = List_PYR[neurone].I_NMDA2(Vpre_AMPA, tps_start)*W

                    List_PYR[neurone].add_I_synDend_Bis(I_NMDA)

#                    pyrPPSE_d[curr_pyr, tt] = (np.sum(I_AMPA) + np.sum(I_NMDA)) / List_PYR[neurone].Cm_d
                    x=I_AMPA + I_NMDA

                    c = np.argwhere(PS.PreSynapticPos_AMPA[i]==5).flatten()
                    pyrPPSE_d1[curr_pyr, tt] = np.sum(x[c])/ List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_AMPA[i]==4).flatten()
                    pyrPPSE_d23[curr_pyr, tt] = np.sum(x[c])/ List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_AMPA[i]==3).flatten()
                    pyrPPSE_d4[curr_pyr, tt] = np.sum(x[c])/ List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_AMPA[i]==2).flatten()
                    pyrPPSE_d5[curr_pyr, tt] = np.sum(x[c])/ List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_AMPA[i]==1).flatten()
                    pyrPPSE_d6[curr_pyr, tt] = np.sum(x[c])/ List_PYR[neurone].Cm_d


                elif typeS[i] == 1:  # PV

                        #print(List_PYR[neurone].I_AMPA2(Vpre_AMPA)*W)
                    I_AMPA = List_PV[neurone].I_AMPA2(Vpre_AMPA)*W
                    I_NMDA = List_PV[neurone].I_NMDA2(Vpre_AMPA, tps_start)*W

                    List_PV[neurone].add_I_synSoma(I_AMPA)
                    List_PV[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 2:  # SST

                    I_AMPA = List_SST[neurone].I_AMPA2(Vpre_AMPA)*W
                    I_NMDA = List_SST[neurone].I_NMDA2(Vpre_AMPA, tps_start)*W

                    List_SST[neurone].add_I_synSoma(I_AMPA)
                    List_SST[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 3:  # VIP

                    I_AMPA = List_VIP[neurone].I_AMPA2(Vpre_AMPA)*W
                    I_NMDA = List_VIP[neurone].I_NMDA2(Vpre_AMPA, tps_start)*W

                    List_VIP[neurone].add_I_synSoma(I_AMPA)
                    List_VIP[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 4:  # RLN
                    I_AMPA = List_RLN[neurone].I_AMPA2(Vpre_AMPA)*W
                    I_NMDA = List_RLN[neurone].I_NMDA2(Vpre_AMPA, tps_start)*W

                    List_RLN[neurone].add_I_synSoma(I_AMPA)
                    List_RLN[neurone].add_I_synSoma(I_NMDA)



            ########################################################################
            ##GABA
            if len(PS.PreSynaptic_Cell_GABA[i]) > 0:
                Cell_GABA = PS.PreSynaptic_Cell_GABA[i]
                Vpre_GABA = np.zeros(len(Cell_GABA))
                # Weight= PS.PreSynapticWeight_GABA[i]
                # if not len(Weight) == 0:
                #     # nWeight = (Weight - np.min(Weight)) / (np.max(Weight) - np.min(Weight))
                #     nWeight =  Weight / np.max(Weight)
                for k, c in enumerate(Cell_GABA):  # switch afferences
                    # layer, type, index = All2layer(c,  Layer_nbCells,NB_PYR, NB_PV, NB_SST, NB_VIP, NB_RLN, List_celltypes)

                    if typeS[c] == 1:  # PV
                        Vpre_GABA[k] = List_PV[indexS[c]].VsOutput()
                    elif typeS[c] == 2:  # SST
                        Vpre_GABA[k] = List_SST[indexS[c]].VsOutput()
                    elif typeS[c] == 3:  # VIP
                        Vpre_GABA[k] = List_VIP[indexS[c]].VsOutput()
                    elif typeS[c] == 4:  # RLN
                        Vpre_GABA[k] = List_RLN[indexS[c]].VsOutput()
                    else:
                        print('error')

                if typeS[i] == 0:  # neurone is a PC
                    I_GABA = List_PYR[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_GABA_d[i], PS.PreSynaptic_Soma_GABA_s[i], PS.PreSynaptic_Soma_GABA_a[i])
                    List_PYR[neurone].add_I_synDend(I_GABA, PS.PreSynaptic_Soma_GABA_d[i])
                    List_PYR[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_GABA_s[i])
                    List_PYR[neurone].add_I_synAis(I_GABA, PS.PreSynaptic_Soma_GABA_a[i])



###################################Add presynaptic currents per layer for dendrites#################

                    x=I_GABA * PS.PreSynaptic_Soma_GABA_d[i]

                    x = I_AMPA + I_NMDA

                    c = np.argwhere(PS.PreSynapticPos_GABA[i] == 5).flatten()
                    pyrPPSI_d1[curr_pyr, tt] = np.sum(x[c]) / List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_GABA[i] == 4).flatten()
                    pyrPPSI_d23[curr_pyr, tt] = np.sum(x[c]) / List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_GABA[i] == 3).flatten()
                    pyrPPSI_d4[curr_pyr, tt] = np.sum(x[c]) / List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_GABA[i] == 2).flatten()
                    pyrPPSI_d5[curr_pyr, tt] = np.sum(x[c]) / List_PYR[neurone].Cm_d

                    c = np.argwhere(PS.PreSynapticPos_GABA[i] == 1).flatten()
                    pyrPPSI_d6[curr_pyr, tt] = np.sum(x[c]) / List_PYR[neurone].Cm_d


                    pyrPPSI_s[curr_pyr, tt] = np.sum(I_GABA*PS.PreSynaptic_Soma_GABA_s[i]) / List_PYR[neurone].Cm
                    pyrPPSI_a[curr_pyr, tt] = np.sum(I_GABA*PS.PreSynaptic_Soma_GABA_a[i]) / List_PYR[neurone].Cm

                elif typeS[i] == 1:  # interneuron PV
                    I_GABA = List_PV[neurone].I_GABA2(Vpre_GABA)
                    #print(neurone)
                    #print(Vpre_GABA)
                    #print(I_GABA)
                    List_PV[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 2:  # interneuron SST
                    I_GABA = List_SST[neurone].I_GABA2(Vpre_GABA)
                    #print(I_GABA)

                    List_SST[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 3:  # interneuron VIP
                    I_GABA = List_VIP[neurone].I_GABA2(Vpre_GABA)
                    List_VIP[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 4:  # RLN
                    I_GABA = List_RLN[neurone].I_GABA2(Vpre_GABA)
                    List_RLN[neurone].add_I_synSoma(I_GABA)
            if typeS[i] == 0:
                curr_pyr += 1
        #############################################
        ########Range Kutta#########################
        for i in range(NB_DPYR):
            List_DPYR[i].rk4()

            # print(List_DPYR[i].y[0])

        for i in range(NB_Th):
            List_Th[i].rk4()
        for i in range(np.sum(NB_PYR)):
            List_PYR[i].rk4()
        for i in range(np.sum(NB_PV)):
            #(List_PV[i].I_synSoma)
            List_PV[i].rk4()
        for i in range(np.sum(NB_SST)):
            List_SST[i].rk4()
        for i in range(np.sum(NB_VIP)):
            List_VIP[i].rk4()
        for i in range(np.sum(NB_RLN)):
            List_RLN[i].rk4()

        #######Get membrane potential variation#######

        for i in range(np.sum(NB_PYR)):
            pyrVs[i, tt] = List_PYR[i].y[0]
            pyrVd[i, tt] = List_PYR[i].y[1]
            pyrVa[i, tt] = List_PYR[i].y[28]
        for i in range(np.sum(NB_PV)):
            PV_Vs[i, tt] = List_PV[i].y[0]
        for i in range(np.sum(NB_SST)):
            SST_Vs[i, tt] = List_SST[i].y[0]
        for i in range(np.sum(NB_VIP)):
            VIP_Vs[i, tt] = List_VIP[i].y[0]
        for i in range(np.sum(NB_RLN)):
            RLN_Vs[i, tt] = List_RLN[i].y[0]


        for i in range(NB_DPYR):
            DPYR_Vs[i, tt] = List_DPYR[i].y[0]
        for i in range(NB_Th):
            Th_Vs[i, tt] = List_Th[i].y[0]



    tps_start += (t[-1] + dt)
    return t,pyrVs, pyrVd, pyrVa,pyrPPSE_d1,pyrPPSE_d23,pyrPPSE_d4,pyrPPSE_d5,pyrPPSE_d6,  pyrPPSI_d1,  pyrPPSI_d23,  pyrPPSI_d4,  pyrPPSI_d5,  pyrPPSI_d6,  pyrPPSI_s,  pyrPPSI_a, PV_Vs, SST_Vs, VIP_Vs, RLN_Vs, DPYR_Vs, Th_Vs

class CorticalColumn:
    updateTime = SenderObject()
    updateCell = SenderObjectInt()
    def __init__(self,d=30):
        self.D = 210 * 2  # neocortical column diameter in micrometers
        self.L = 2082  # neocortical column length in micrometers #Markram et al. 2015
        self.C = 2000
        # Layer_d=np.array([165,353+149,190,525,700]) #layers' thicknesses (L1-L2/3-L4-L5-L6)
        self.Layer_d = np.array([165, 353 + 149, 190, 525, 700])  # layers' thicknesses (L1-L2/3-L4-L5-L6)
        self.Layertop_pos=np.cumsum(self.Layer_d[::-1])
        # Layer_nbCells=np.array([338/3,7524/3,4656/3,6114/3,12651/3])  #total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)
        self.Layer_nbCells = np.array([322,7524, 4656, 6114,
                                  12651]) / d  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)

        self.Nbcells = int(np.sum(self.Layer_nbCells))
        self.PYRpercent = np.array([0,0.7, 0.9, 0.8, 0.9])
        self.INpercent = 1 - self.PYRpercent
        self.PVpercent= np.array([0,0.3, 0.55, 0.45, 0.45])
        self.SSTpercent= np.array([0,0.2, 0.25, 0.40, 0.35])
        self.VIPpercent= np.array([0,0.40, 0.20, 0.15, 0.20])
        self.RLNpercent= np.array([1,0.1, 0, 0, 0])


        NBPYR = self.PYRpercent * self.Layer_nbCells
        self.NB_PYR = NBPYR.astype(int)
        print('PYR',self.NB_PYR)
        #subtypes of PYR cells
        self.PCsubtypes_Per=np.array([[0,0,0,0,0],[0.9*self.NB_PYR[1],0,0.1*self.NB_PYR[1],0,0],[0.5*self.NB_PYR[2],0.36*self.NB_PYR[2],0,0,0.14*self.NB_PYR[2]],[self.NB_PYR[3]*0.81,self.NB_PYR[3]*0.19,0,0,0],[self.NB_PYR[4]*0.39,self.NB_PYR[4]*0.17,self.NB_PYR[4]*0.20,self.NB_PYR[4]*0.24,0]]) #TPC,UPC,IPC,BPC,SSC
        print(self.PCsubtypes_Per)


        NB_IN = self.INpercent * self.Layer_nbCells
        self.NB_IN = NB_IN.astype(int)
        print('IN',self.NB_IN)
        NB_PV = self.PVpercent * self.NB_IN
        self.NB_PV = NB_PV.astype(int)
        print('PV',self.NB_PV)
        NB_PV_BC = 0.7 * self.NB_PV
        self.NB_PV_BC = NB_PV_BC.astype(int)
        self.NB_PV_ChC = (NB_PV - NB_PV_BC).astype(int)
        NB_SST =self.SSTpercent * self.NB_IN
        self.NB_SST = NB_SST.astype(int)
        print('SST',self.NB_SST)
        NB_VIP = self.VIPpercent * self.NB_IN
        self.NB_VIP = NB_VIP.astype(int)
        print('VIP',self.NB_VIP)
        NB_RLN = self.RLNpercent * self.NB_IN
        self.NB_RLN = NB_RLN.astype(int)
        print('RLN',self.NB_RLN)
        #mini_Layer_nbcells = Layer_nbCells / 310
        self.Layer_nbCells = self.NB_PYR+self.NB_SST+self.NB_PV+self.NB_VIP+self.NB_RLN  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)
        print(self.Layer_nbCells)
        #####External afferences
        self.NB_DPYR=int(np.sum(self.NB_PYR)*0.07)
        self.NB_Th=int(np.sum(self.NB_PYR)*0.07)
        self.List_celltypes = np.array([np.array([0]*self.NB_PYR[l] + [1]*self.NB_PV[l] + [2]*self.NB_SST[l] + [3]*self.NB_VIP[l] + [4]*self.NB_RLN[l]).astype(int) for l in range(len(self.Layer_nbCells))])
        self.List_C = np.array(
            [[1, 1, 1, 1, 0],  # PC -> PC,PV,SST,VIP ,RLN  affinités de connexion entre cellules
             [1, 1, 0, 0, 0],  # PV -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 1, 1],  # SST -> PC,PV,SST,VIP ,RLN
             [0, 0, 1, 0, 0],   #VIP --> PC,PV,SST,VIP ,RLN
             [1, 1, 1, 1, 1]    #RLN --> PC,PV,SST,VIP ,RLN
             ], dtype=np.float)
        # self.inputNB=int(np.sum(self.Layer_nbCells)/20) ## /2 fro realistic purpose and /10 for repetitive connections see Denoyer et al. 2020
        self.update_inputNB()
        df = pd.read_excel('afferences.xlsx',engine='openpyxl')
        self.Afferences = df.to_numpy()
        self.update_connections(self.Afferences)
        # self.inputpercent=df.to_numpy()*self.inputNB/100
        #print(self.inputpercent)
        self.Allconnexions=[]

        # Sim param
        self.Fs = 50
        self.tps_start = 0
        self.T = 200 #simduration
        self.dt = 1 / self.Fs
        self.nbEch = int(self.T / self.dt)
        self.Stim_InputSignals  = []
        self.Stim_Signals  = []

        self.ImReady = False
        self.seed = 0

        self.tanpix40p180 = math.tan(40 * math.pi / 180)
        self.tanpix50p180 = math.tan(50 * math.pi / 180)

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


    def update_samples(self,Fs,T):
        self.Fs = Fs
        self.T = T
        self.dt = 1 / self.Fs
        self.nbEch = int(self.T / self.dt)

    def updateTissue(self, D, L , C, Layer_d):
        self.D = D
        self.L = L
        self.C = C
        self.Layer_d = Layer_d

    def update_cellNumber(self,Layer_nbCells,
                             PYRpercent,
                             PVpercent,
                             SSTpercent,
                             VIPpercent,
                             RLNpercent,

                            PCsubtypes_Per = None,
                            NB_PYR=None,
                            NB_PV_BC=None,
                            NB_PV_ChC=None,
                            NB_IN = None,
                            NB_PV = None,
                            NB_SST = None,
                            NB_VIP = None,
                            NB_RLN = None  ):

        self.Layer_nbCells = Layer_nbCells  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)

        self.Nbcells = int(np.sum(self.Layer_nbCells))
        self.PYRpercent = PYRpercent
        self.INpercent = 1 - self.PYRpercent
        self.PVpercent = PVpercent
        self.SSTpercent = SSTpercent
        self.VIPpercent = VIPpercent
        self.RLNpercent = RLNpercent

        if NB_PYR is None:
            NB_PYR = self.PYRpercent * self.Layer_nbCells
            self.NB_PYR = NB_PYR.astype(int)
            print('PYR', self.NB_PYR)
        else:
            self.NB_PYR = NB_PYR
        # subtypes of PYR cells
        if PCsubtypes_Per is None:
            self.PCsubtypes_Per = np.array([[0, 0, 0, 0, 0],
                                            [0.9 * self.NB_PYR[1], 0, 0.1 * self.NB_PYR[1], 0, 0],
                                            [0.5 * self.NB_PYR[2], 0.36 * self.NB_PYR[2], 0, 0, 0.14 * self.NB_PYR[2]],
                                            [self.NB_PYR[3] * 0.81, self.NB_PYR[3] * 0.19, 0, 0, 0],
                                            [self.NB_PYR[4] * 0.39, self.NB_PYR[4] * 0.17, self.NB_PYR[4] * 0.20,self.NB_PYR[4] * 0.24, 0]])  # TPC,UPC,IPC,BPC,SSC
            print(self.PCsubtypes_Per)
        else:
            self.PCsubtypes_Per = PCsubtypes_Per

        if NB_IN is None:
            NB_IN = self.INpercent * self.Layer_nbCells
            self.NB_IN = NB_IN.astype(int)
            print('IN', self.NB_IN)
        else:
            self.NB_IN = NB_IN
        if NB_PV is None:
            NB_PV = self.PVpercent * self.NB_IN
            self.NB_PV = NB_PV.astype(int)
            print('PV', self.NB_PV)
        else:
            self.NB_PV = NB_PV
        if NB_PV_BC is None:
            NB_PV_BC = 0.7 * self.NB_PV
            self.NB_PV_BC = NB_PV_BC.astype(int)
        else:
            self.NB_PV_BC = NB_PV_BC
        if NB_PV_ChC is None:
            NB_PV_BC = 0.7 * self.NB_PV
            self.NB_PV_ChC = self.NB_PV - NB_PV_BC
        else:
            self.NB_PV_BC = NB_PV_BC

        if NB_SST is None:
            NB_SST = self.SSTpercent * self.NB_IN
            self.NB_SST = NB_SST.astype(int)
            print('SST', self.NB_SST)
        else:
            self.NB_SST = NB_SST
        if NB_VIP is None:
            NB_VIP = self.VIPpercent * self.NB_IN
            self.NB_VIP = NB_VIP.astype(int)
            print('VIP', self.NB_VIP)
        else:
            self.NB_VIP = NB_VIP
        if NB_RLN is None:
            NB_RLN = self.RLNpercent * self.NB_IN
            self.NB_RLN = NB_RLN.astype(int)
            print('RLN', self.NB_RLN)
        else:
            self.NB_RLN = NB_RLN
        # mini_Layer_nbcells = Layer_nbCells / 310
        self.Layer_nbCells = self.NB_PYR + self.NB_SST + self.NB_PV + self.NB_VIP + self.NB_RLN  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)
        print(self.Layer_nbCells)
        #####External afferences
        self.NB_DPYR = int(np.sum(self.NB_PYR) * 0.07)
        self.NB_Th = int(np.sum(self.NB_PYR) * 0.07)
        self.List_celltypes = np.array([np.array([0]*self.NB_PYR[l] + [1]*self.NB_PV[l] + [2]*self.NB_SST[l] + [3]*self.NB_VIP[l] + [4]*self.NB_RLN[l]).astype(int) for l in range(len(self.Layer_nbCells))])
        self.List_C = np.array(
            [[1, 1, 1, 1, 0],  # PC -> PC,PV,SST,VIP ,RLN  affinités de connexion entre cellules
             [1, 1, 0, 0, 0],  # PV -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 1, 1],  # SST -> PC,PV,SST,VIP ,RLN
             [0, 0, 1, 0, 0],  # VIP --> PC,PV,SST,VIP ,RLN
             [1, 1, 1, 1, 1]  # RLN --> PC,PV,SST,VIP ,RLN
             ], dtype=np.float)
        # self.inputNB = int(np.sum(
        #     self.Layer_nbCells) / 20)  ## /2 fro realistic purpose and /10 for repetitive connections see Denoyer et al. 2020
        self.update_inputNB()
        self.ImReady = False
        self.List_Lambda_s = np.array(
            [[0.032, 0, 0, -0.032, 0],  # layer II/III-> TPC,UPC,IPC,BPC,SSC
             [0.048, 0.048, 0, 0, -0.012],  # layer IV-> TPC,UPC,IPC,BPC,SSC
             [0.137, 0.048, 0, 0, 0],  # layer V-> TPC,UPC,IPC,BPC,SSC
             [0.117, 0, -0.104, 0.024, 0]  # layer VI-> TPC,UPC,IPC,BPC,SSC
             ], dtype=np.float)

        self.List_Lambda_d = np.array(
            [[-0.120, 0, 0, 0.120, 0],  # layer II/III-> TPC,UPC,IPC,BPC,SSC
             [-0.095, -0.095, 0, 0, 0],  # layer IV-> TPC,UPC,IPC,BPC,SSC
             [-0.071, -0.076, 0, 0, 0],  # layer V-> TPC,UPC,IPC,BPC,SSC
             [-0.046, 0, 0.047, 0.019, 0]  # layer VI-> TPC,UPC,IPC,BPC,SSC
             ], dtype=np.float)

    def update_inputNB(self):
            self.inputNB = int(np.sum(self.Layer_nbCells) / 20)
    def update_connections(self,matrice, fixed = False):
        self.Afferences = matrice
        if fixed:
            self.inputpercent = matrice.astype(int)
        else:
            self.inputpercent=np.ceil(matrice*self.inputNB/100)
            self.inputpercent=self.inputpercent.astype(int)
        self.Allconnexions=[]



    def PlaceCell_func(self, type='Cylinder', seed = 0):
        if not seed == 0:
            np.random.seed(seed)
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        def polar2cart3D(r, theta, phi):
            return [
                r * math.sin(theta) * math.cos(phi),
                r * math.sin(theta) * math.sin(phi),
                r * math.cos(theta)
            ]

        def asCartesian(r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return [x, y, z]
        # self.updateCell.something_happened.emit(-2)
        Cellpositionall=[]
        L=self.L
        Layer_d_cumsum = np.cumsum(self.Layer_d)
        Layer_d_cumsum = np.hstack((0,Layer_d_cumsum))
        print('Place cells....')
        if type == 'Cylinder':
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]
                # x0 =float(np.random.uniform(low=-210, high=210, size=1))
                # # y=np.sqrt(np.abs(210**2-x**2))*(-1)**random.randint(1, 2)
                # CellPosition.append(np.array([val for val in [x0,float(np.random.uniform(low=-np.sqrt(np.abs(210**2-x0**2)), high=np.sqrt(np.abs(210**2-x0**2)), size=1)),float(np.random.uniform(low=L-np.sum(self.Layer_d[0:l+1]), high=L-np.sum(self.Layer_d[0:l]), size=1))]]))
                module = float(np.random.uniform(low=0, high=1, size=1))
                phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0, y0 = pol2cart(module*self.D/2, phi)
                z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
                CellPosition.append(np.array([x0,y0,z0]))
                # print(CellPosition)
                for nb in range(int(self.Layer_nbCells[l])-1):
                    # x = np.array([np.random.uniform(low=-210, high=210, size=1) for fd in range(20)])
                    # # print(x)
                    #
                    # candidate=np.array([np.array([val for val in [xx[0],float(np.random.uniform(low=-np.sqrt(np.abs(210**2-xx**2)), high=np.sqrt(np.abs(210**2-xx**2)), size=1)),
                    #                            float(np.random.uniform(low=L-np.sum(self.Layer_d[0:l+1]), high=L-np.sum(self.Layer_d[0:l]), size=1))]]) for xx in x])
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))
                        x0, y0 = pol2cart(module*self.D/2, phi)
                        z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)
        elif type == 'Cylinder with curvature':
            CurvatureD = self.C
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]
                # x0 =float(np.random.uniform(low=-210, high=210, size=1))
                # # y=np.sqrt(np.abs(210**2-x**2))*(-1)**random.randint(1, 2)
                # CellPosition.append(np.array([val for val in [x0,float(np.random.uniform(low=-np.sqrt(np.abs(210**2-x0**2)), high=np.sqrt(np.abs(210**2-x0**2)), size=1)),float(np.random.uniform(low=L-np.sum(self.Layer_d[0:l+1]), high=L-np.sum(self.Layer_d[0:l]), size=1))]]))
                module = float(np.random.uniform(low=0, high=1, size=1))
                phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0, y0 = pol2cart(module*self.D/2, phi)

                zL = CurvatureD + L
                c = np.sqrt(x0 * x0 + y0 * y0)
                thetaX= np.arctan(x0/zL)
                thetaY= np.arctan(y0/zL)

                zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
                zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                module = float(np.random.uniform(low=zmin, high=zmax, size=1))
                # phi1 = float(np.random.uniform(low=-thetamin, high=thetamin, size=1))
                # phi2 = float(np.random.uniform(low=-thetamin, high=thetamin, size=1))

                x0, y0, z0 = asCartesian(module, thetaX, float(np.random.uniform(low=0, high=2*np.pi, size=1)))
                z0 = z0 - CurvatureD
                # z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))



                # thetamax = np.arctan(c/zmax)
                #
                # Zmin = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]) * np.cos(thetamin)
                # Zmax = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]) * np.cos(thetamax)
                #
                # z0 = float(np.random.uniform(low=Zmin-CurvatureD, high=Zmax-CurvatureD, size=1))


                # good = True
                # while good:
                #     z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
                #     d = np.linalg.norm(np.array([x0,y0,z0])-np.array([0,0,-CurvatureD]))
                #     print(d, CurvatureD + L - Layer_d_cumsum[l], L - Layer_d_cumsum[l + 1])
                #     if d <= CurvatureD + L - Layer_d_cumsum[l] and d >= L - Layer_d_cumsum[l + 1]:
                #         good = False
                CellPosition.append(np.array([x0,y0,z0]))
                # print(CellPosition)
                for nb in range(int(self.Layer_nbCells[l])-1):
                    # x = np.array([np.random.uniform(low=-210, high=210, size=1) for fd in range(20)])
                    # # print(x)
                    #
                    # candidate=np.array([np.array([val for val in [xx[0],float(np.random.uniform(low=-np.sqrt(np.abs(210**2-xx**2)), high=np.sqrt(np.abs(210**2-xx**2)), size=1)),
                    #                            float(np.random.uniform(low=L-np.sum(self.Layer_d[0:l+1]), high=L-np.sum(self.Layer_d[0:l]), size=1))]]) for xx in x])
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))
                        x0, y0 = pol2cart(module * self.D / 2, phi)

                        zL = CurvatureD + L
                        c = np.sqrt(x0 * x0 + y0 * y0)
                        thetaX = np.arctan(x0 / zL)
                        thetaY = np.arctan(y0 / zL)

                        zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                        zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                        module = float(np.random.uniform(low=zmin, high=zmax, size=1))
                        # phi1 = float(np.random.uniform(low=-thetamin, high=thetamin, size=1))
                        # phi2 = float(np.random.uniform(low=-thetamin, high=thetamin, size=1))

                        x0, y0, z0 = asCartesian(module, thetaX, float(np.random.uniform(low=0, high=2*np.pi, size=1)))
                        z0 = z0 - CurvatureD

                        # good = True
                        #
                        # while good:
                        #     z0 = float(
                        #         np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                        #     d = np.linalg.norm(
                        #         np.array([x0, y0, z0]) - np.array([0, 0, -CurvatureD ]))
                        #     # print(d,CurvatureD + L - Layer_d_cumsum[l],L - Layer_d_cumsum[l + 1])
                        #     if d <= CurvatureD + L - Layer_d_cumsum[l]  and d >= L - Layer_d_cumsum[l + 1]:
                        #         good = False
                        # z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)
        elif type == 'Square':
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]

                # phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                y0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
                CellPosition.append(np.array([x0,y0,z0]))
                for nb in range(int(self.Layer_nbCells[l])-1):
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                        x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        y0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)
        elif type == 'Square with curvature':

            CurvatureD = self.C
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]

                # phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                y0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))

                # zL = CurvatureD + L
                zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
                zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                c = np.sqrt(x0 * x0 + y0 * y0)
                thetamin = np.arctan(c/zmin)
                thetamax = np.arctan(c/zmax)

                Zmin = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]) * np.cos(thetamin)
                Zmax = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]) * np.cos(thetamax)

                z0 = float(np.random.uniform(low=Zmin-CurvatureD, high=Zmax-CurvatureD, size=1))
                CellPosition.append(np.array([x0,y0,z0]))
                for nb in range(int(self.Layer_nbCells[l])-1):
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                        x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        y0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        # zL = CurvatureD + L
                        zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                        zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                        c = np.sqrt(x0 * x0 + y0 * y0)
                        thetamin = np.arctan(c / zmin)
                        thetamax = np.arctan(c / zmax)

                        Zmin = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]) * np.cos(thetamin)
                        Zmax = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]) * np.cos(thetamax)

                        z0 = float(np.random.uniform(low=Zmin - CurvatureD, high=Zmax - CurvatureD, size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)

        elif type == 'Rectange':
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]

                # phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                y0 = float(np.random.uniform(low=-self.L/2, high=self.L/2, size=1))
                z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
                CellPosition.append(np.array([x0,y0,z0]))
                for nb in range(int(self.Layer_nbCells[l])-1):
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                        x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        y0 = float(np.random.uniform(low=-self.L/2, high=self.L/2, size=1))
                        z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)
        elif type == 'Rectange with curvature':

            CurvatureD = self.C
            for l in range(len(self.Layer_nbCells)):
                CellPosition=[]
                PYRCellPosition=[]
                IntCellPosition=[]

                # phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
                x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                y0 = float(np.random.uniform(low=-self.L/2, high=self.L/2, size=1))
                # zL = CurvatureD + L
                zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
                zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                c = np.sqrt(x0 * x0 + y0 * y0)
                thetamin = np.arctan(c/zmin)
                thetamax = np.arctan(c/zmax)

                Zmin = (zmin) * np.cos(thetamin)
                Zmax = (zmax) * np.cos(thetamax)

                z0 = float(np.random.uniform(low=Zmin-CurvatureD, high=Zmax-CurvatureD, size=1))

                CellPosition.append(np.array([x0,y0,z0]))
                for nb in range(int(self.Layer_nbCells[l])-1):
                    candidate=[]
                    for k_i in range(20):
                        module = float(np.random.uniform(low=0, high=1, size=1))
                        phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                        x0 = float(np.random.uniform(low=-self.D/2, high=self.D/2, size=1))
                        y0 = float(np.random.uniform(low=-self.L/2, high=self.L/2, size=1))

                        # zL = CurvatureD + L
                        zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                        zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                        c = np.sqrt(x0 * x0 + y0 * y0)
                        thetamin = np.arctan(c / zmin)
                        thetamax = np.arctan(c / zmax)

                        Zmin = (zmin) * np.cos(thetamin)
                        Zmax = (zmax) * np.cos(thetamax)

                        z0 = float(np.random.uniform(low=Zmin - CurvatureD, high=Zmax - CurvatureD, size=1))
                        candidate.append(np.array([x0, y0, z0]))
                    candidate = np.array(candidate)

                    # print(candidate)
                    CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                    # print(CellDistances)
                    argmin = np.argmin(CellDistances, axis=0)
                    valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                    argmax = np.argmax(valmin)
                    CellPosition.append(candidate[argmax, :])
                    if nb<self.NB_PYR[l]-1:
                        PYRCellPosition.append(candidate[argmax, :])
                    else:
                        IntCellPosition.append(candidate[argmax, :])
                CellPosition=np.array(CellPosition)
                marange = np.arange(CellPosition.shape[0])
                np.random.shuffle(marange)
                CellPosition2 = CellPosition[marange]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(CellPosition[:,0],CellPosition[:,1],CellPosition[:,2],'X')
                # plt.show()
                Cellpositionall.append(CellPosition2)
        self.Cellpos=np.array(Cellpositionall)
        self.Cellposflat=np.vstack((self.Cellpos[0], self.Cellpos[1], self.Cellpos[2], self.Cellpos[3], self.Cellpos[4]))
        return self.Cellpos

    def createFlatCEllpos(self):
        self.Cellposflat=np.vstack((self.Cellpos[0], self.Cellpos[1], self.Cellpos[2], self.Cellpos[3], self.Cellpos[4]))


    def Create_Connectivity_Matrix(self,seed = 0):
        if not seed == 0:
            np.random.seed(seed)
        self.createFlatCEllpos()
        print('Create_Connectivity_Matrices........')
        connectivitymatrix = []
        connectivityweight = []
        connectivityZpos = []
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_AMPA = []
        self.PreSynaptic_Soma_GABA_d = []
        self.PreSynaptic_Soma_GABA_s = []
        self.PreSynaptic_Soma_GABA_a = []
        self.ExternalPreSynaptic_Cell_AMPA_DPYR = []
        self.ExternalPreSynaptic_Cell_AMPA_Th= []
        self.PreSynapticWeight_AMPA=[]
        self.PreSynapticWeight_GABA=[]
        self.PreSynapticPos_AMPA=[]
        self.PreSynapticPos_GABA=[]


        target = Cell_morphology.Neuron(0,1,0)
        Neighbor = Cell_morphology.Neuron(0,1,0)
        nbcells = np.sum(self.Layer_nbCells)
        nbcellscum = np.cumsum(self.Layer_nbCells)
        nbcellscum = np.append(0,nbcellscum)

        PCsubtypes_Per = np.cumsum(self.PCsubtypes_Per,axis=1)
        indexval = -1

        cm = np.zeros(nbcells, dtype=int)  # []
        dist = np.zeros(nbcells, dtype=float)  # []
        Weight = np.zeros(nbcells, dtype=float)  # []
        Zpos = np.zeros(nbcells, dtype=np.int32)  # []

        AMPAcells = np.zeros(nbcells, dtype=int) - 1  # []
        GABAcells = np.zeros(nbcells, dtype=int) - 1  # []

        Layer_nbCells = np.cumsum(np.hstack((0,self.Layer_nbCells)))
        self.List_cellsubtypes = copy.deepcopy(self.List_celltypes)
        for l in range(len(self.Layer_nbCells)):
            for cell in range(int(self.Layer_nbCells[l])):
                if np.mod(cell,20) == 0:
                    print('cell', cell)
                # if indexval ==80:
                #     a=1
                # self.updateCell.something_happened.emit(indexval)
                indexval +=1
                cm = cm*0
                dist = dist *0
                Weight = Weight*0
                Zpos=Zpos*0 # The Z position of the synaptic connexion (Zpos=0 no connection, Zpos=-1 ==> not enough cells connected)

                AMPAcells = AMPAcells*0-1
                AMPAcells_sparse = []
                Weight_AMPA=[]
                Pos_AMPA=[]
                GABAcells=GABAcells*0-1#[]
                GABAcells_sparse = []
                GABASoma=np.zeros(nbcells,dtype=str)#[]
                GABASoma_d_sparse=[]
                GABASoma_s_sparse=[]
                GABASoma_a_sparse=[]
                Weight_GABA = []
                Pos_GABA=[]

                subtype = []

                #if Principal cell check subtypes
                if (self.List_celltypes[l][cell] == 0):  # getsubtype
                    if cell < PCsubtypes_Per[l][0]:
                        subtype = 0  # TPC
                    elif (cell >= PCsubtypes_Per[l][0]) and (cell < PCsubtypes_Per[l][1]):
                        subtype = 1  # UPC
                    elif (cell >= PCsubtypes_Per[l][1]) and (cell < PCsubtypes_Per[l][2]):
                        subtype = 2  # IPC
                    elif (cell >= PCsubtypes_Per[l][2]) and (cell < PCsubtypes_Per[l][3]):
                        subtype = 3  # BPC
                    elif (cell >= PCsubtypes_Per[l][3]) and (cell < PCsubtypes_Per[l][4]):
                        subtype = 4  # SSC
                    self.List_cellsubtypes[l][cell] = subtype

                #if PV check is chandeliers or Basket
                elif (self.List_celltypes[l][cell] == 1):  # PV get subtype
                    if (cell - self.NB_PYR[l]) < self.NB_PV_BC[l]:
                        subtype = 0  # BC
                    else:
                        subtype = 1  # Chandelier
                    self.List_cellsubtypes[l][cell] = subtype
                else:
                    self.List_cellsubtypes[l][cell] = -1

                target.update_type(type=self.List_celltypes[l][cell], layer=l, subtype=subtype)
                index = 0

                d = np.linalg.norm(self.Cellposflat[:,:2]  - self.Cellpos[l][cell][:2], axis=1)
                indflat = -1
                for subl in range(len(self.Layer_nbCells)):
                    for v in range(0, int(self.Layer_nbCells[subl])):
                        indflat += 1
                        subtype = None

                        if self.inputpercent[int(self.List_celltypes[subl][v] + 5 * subl), int(target.type + 5 * l)] == 0:
                            cm[index] = 0
                            Weight[index] = -1
                            AMPAcells[index] = -1
                            GABAcells[index] = -1
                            GABASoma[index] = '0'
                            Zpos[index] = 0


                        else:
                            #remove auto connections except for PV [Deleuze et al. 2019,plos Bio]
                            if ((v == cell) and (subl == l) and (self.List_celltypes[l][cell] != 1)):
                                    cm[index] = 0
                                    Weight[index] = -1
                                    AMPAcells[index] = -1
                                    GABAcells[index] = -1
                                    GABASoma[index] = '0'
                                    Zpos[index] =0

                            else:
                                if (self.List_celltypes[subl][v] == 0):  # getsubtype
                                    if v < PCsubtypes_Per[subl][0]:
                                        subtype = 0  # TPC
                                    elif (v >= PCsubtypes_Per[subl][0]) and (v < PCsubtypes_Per[subl][1]):
                                        subtype = 1  # UPC
                                    elif (v >= PCsubtypes_Per[subl][1]) and (v < PCsubtypes_Per[subl][2]):
                                        subtype = 2  # IPC
                                    elif (v >= PCsubtypes_Per[subl][2]) and (v < PCsubtypes_Per[subl][3]):
                                        subtype = 3  # BPC
                                    elif (v >= PCsubtypes_Per[subl][3]) and (v < PCsubtypes_Per[subl][4]):
                                        subtype = 4  # SSC

                                if (self.List_celltypes[subl][v] == 1):  # PV get subtype
                                    if (v - self.NB_PYR[subl]) < self.NB_PV_BC[subl]:
                                        subtype = 0  # BC
                                    else:
                                        subtype = 1  # Chandelier
                                Neighbor.update_type(type=self.List_celltypes[subl][v], layer=subl,subtype=subtype)

                                isconnected, overlap, ConnPos = self.IsConnected(Neighbor, self.Cellpos[subl][v], target,
                                                                        self.Cellpos[l][cell],d[indflat]) #Neighbor to target)
                                #find at which layer the dendritic connection is
                                if ConnPos==0: #not connected
                                    Layerconn=0
                                else:
                                    Layerconn=np.argwhere(np.sort(np.concatenate((self.Layertop_pos,np.array([ConnPos])),axis=0))==ConnPos) #1--> layer 6,5 layer I
                                    Layerconn =Layerconn[0][0] + 1
                                    # if ConnPos < self.Cellpos[l][cell][2]: #if connection below soma center
                                    #     Layerconn=-1*Layerconn




                                if isconnected == 1:
                                    cm[index] = 1
                                    Weight[index] = overlap
                                    Zpos[index] = Layerconn

                                    ####Fill presynatptic cell
                                    if Neighbor.type==0: #excitatory Input
                                        AMPAcells[index] = v+np.sum(Layer_nbCells[subl])
                                        GABAcells[index] = -1
                                        GABASoma[index] = '0'
                                    else:
                                        #inhibitory input
                                        GABAcells[index] = v+np.sum(Layer_nbCells[subl])
                                        AMPAcells[index] = -1

                                        if target.type in [0]:
                                            if Neighbor.type == 1: #from, PV
                                                if target.subtype == 0: #if from Basket cell
                                                    GABASoma[index] = 's'
                                                else: #if from chandelier cell
                                                    GABASoma[index] = 'a'
                                            else:
                                                GABASoma[index] = 'd'
                                        elif target.type in [1,2,3,4]:
                                            GABASoma[index] = 's'
                                else:
                                    cm[index] = 0
                                    Weight[index] = -1
                                    AMPAcells[index] = -1
                                    GABAcells[index] = -1
                                    GABASoma[index] = '0'
                                    Zpos[index] = 0


                        index += 1
                #####Check afferences:##############
                Afferences=self.inputpercent[:,int(target.type+5*l)]

                #getminweight
                Weight2 = Weight[Weight > 0]
                if Weight2.size == 0:
                    weigthmini = 1
                    print(weigthmini)
                else:
                    weigthmini = np.min(Weight2)


                if np.sum(cm[0:self.NB_RLN[0]]) > Afferences[4]:
                    NBrange = np.array(range(0, self.NB_RLN[0]))
                    th = np.argsort([Weight[j] for j in np.array(NBrange)])[::-1]
                    Weight[NBrange[th[int(Afferences[4]):]]]=0
                    Zpos[NBrange[th[int(Afferences[4]):]]]=0
                    cm[NBrange[th[int(Afferences[4]):]]]=0
                    AMPAcells[NBrange[th[int(Afferences[4]):]]]=-1
                    GABAcells[NBrange[th[int(Afferences[4]):]]]=-1
                    GABASoma[NBrange[th[int(Afferences[4]):]]]='0'

                elif np.sum(cm[0:self.NB_RLN[0]]) < Afferences[4]:
                    nb = Afferences[4] - np.sum(cm[0:self.NB_RLN[0]])
                    indice_ = np.random.randint(0, self.NB_RLN[0], size=nb)
                    for ind in range(nb):
                        pos = indice_[ind]
                        if not pos ==indexval:
                            Weight[pos]=weigthmini
                            cm[pos]=1
                            AMPAcells[pos]=-1
                            GABAcells[pos]=pos
                            GABASoma[pos]='s'
                            Zpos[pos]= 5 #from layer 1


                for ll in range(1,5):
                    for type in range(5):
                        if not int(Afferences[type+5*ll]) == 0:
                            NBrange = []
                            if type == 0:  # PC
                                NBrange = np.array(range(nbcellscum[ll],nbcellscum[ll] + self.NB_PYR[ll]))
                            elif type == 1:  # PV
                                NBrange = np.array(range(nbcellscum[ll] + self.NB_PYR[ll],nbcellscum[ll] + self.NB_PYR[ll] +self.NB_PV[ll]))
                            elif type == 2:  # SST
                                NBrange = np.array(range(nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll],nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll] +self.NB_SST[ll]))
                            elif type == 3:  # VIP
                                NBrange = np.array(
                                    range(nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll] +self.NB_SST[ll],nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll] +self.NB_SST[ll] +self.NB_VIP[ll]))
                            elif type == 4:  # RLN
                                NBrange = np.array(range(
                                    nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll] + self.NB_SST[ll] +self.NB_VIP[ll],nbcellscum[ll] + self.NB_PYR[ll] + self.NB_PV[ll] + self.NB_SST[ll] +
                                    self.NB_VIP[ll] + self.NB_RLN[ll]))

                            if len(NBrange)>0:
                                somme_cm = np.sum(cm[NBrange])
                                if somme_cm > int(Afferences[type+5*ll]): #np.sum([cm[j] for j in NBrange]) > int(Afferences[type+5*ll]):
                                    th=np.argsort( Weight[NBrange])[::-1]
                                    ind_th = NBrange[th[int(Afferences[type+5*ll]):]]
                                    Weight[ind_th]=0
                                    Zpos[ind_th]=0
                                    cm[ind_th]=0
                                    AMPAcells[ind_th]=-1
                                    GABAcells[ind_th]=-1
                                    GABASoma[ind_th]='0'

                                elif somme_cm < int(Afferences[type+5*ll]):
                                    # print('not enough)')
                                    nb = int(Afferences[type+5*ll]) - somme_cm
                                    indice_ = np.random.randint(0, len(NBrange), size=nb)
                                    for ind in range(nb):
                                        pos = NBrange[indice_[ind]]
                                        if not pos == indexval:
                                            cm[pos] = 1
                                            Weight[pos]=weigthmini
                                            Zpos[pos] =5-ll #add the layer of presynaptic cell
                                            if type == 0:
                                                AMPAcells[pos] = pos
                                                GABAcells[pos] = -1
                                                GABASoma[pos] = '0'
                                            else:
                                                GABAcells[pos] = pos
                                                AMPAcells[pos] = -1
                                                if target.type in [0]:
                                                    if type == 1:
                                                        if target.subtype == 0:
                                                            GABASoma[pos] = 's'
                                                        else:
                                                            GABASoma[pos] = 'a'
                                                    else:
                                                        GABASoma[pos] = 'd'
                                                elif target.type in [1, 2, 3, 4]:
                                                    GABASoma[pos] = 's'


                #create sparse arrays
                for i in range(len(AMPAcells)):
                    if AMPAcells[i]!=-1:
                        AMPAcells_sparse.append(AMPAcells[i])
                        Weight_AMPA.append(Weight[i])
                        Pos_AMPA.append(Zpos[i])


                        # AMPA_ProjVect.append(dist[i,:])
                for i in range(len(GABAcells)):
                    if GABAcells[i]!=-1:
                        GABAcells_sparse.append(GABAcells[i])
                        Weight_GABA.append(Weight[i])
                        Pos_GABA.append(Zpos[i])

                        # GABA_ProjVect.append(dist[i,:])
                        if GABASoma[i]=='d':
                            GABASoma_d_sparse.append(1) #1   for dent
                            GABASoma_s_sparse.append(0) #1   for soma
                            GABASoma_a_sparse.append(0) #1   for ais
                        elif GABASoma[i]=='s':
                            GABASoma_d_sparse.append(0) #1   for dent
                            GABASoma_s_sparse.append(1) #1   for soma
                            GABASoma_a_sparse.append(0) #1   for ais
                        elif GABASoma[i]=='a':
                            GABASoma_d_sparse.append(0) #1   for dent
                            GABASoma_s_sparse.append(0) #1   for soma
                            GABASoma_a_sparse.append(1) #1   for ais


                self.PreSynaptic_Cell_AMPA.append(np.asarray(AMPAcells_sparse))
                self.PreSynaptic_Cell_GABA.append(np.asarray(GABAcells_sparse))
                self.PreSynaptic_Soma_AMPA.append(np.ones(len(AMPAcells_sparse),dtype=int))
                self.PreSynaptic_Soma_GABA_d.append(np.asarray(GABASoma_d_sparse))
                self.PreSynaptic_Soma_GABA_s.append(np.asarray(GABASoma_s_sparse))
                self.PreSynaptic_Soma_GABA_a.append(np.asarray(GABASoma_a_sparse))
                self.PreSynapticWeight_AMPA.append(np.asarray(Weight_AMPA))
                self.PreSynapticWeight_GABA.append(np.asarray(Weight_GABA))
                self.PreSynapticPos_AMPA.append(np.asarray(Pos_AMPA))
                self.PreSynapticPos_GABA.append(np.asarray(Pos_GABA))

                #connectivitymatrix.append(np.asarray(cm))
                #connectivityweight.append(np.asarray(Weight))


                connectivitymatrix.append(np.where(cm==1)[0])
                connectivityweight.append(Weight[np.where(cm==1)])
                connectivityZpos.append(Zpos[np.where(cm==1)])
                #create external synaptic input
                nbstim=int(Afferences[26])
                nbth=int(Afferences[25])

                if (nbstim != 0):
                    x0 = [np.random.randint(self.NB_DPYR ) for i in range(np.min((int(nbstim),self.NB_DPYR)))]
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR.append(np.asarray(x0))
                    # connectivitymatrix[-1] = np.hstack((connectivitymatrix[-1], -np.asarray(x0) - 1))
                else:
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR.append(np.asarray([]))

                if (nbth!=0):
                    x1 = [np.random.randint(self.NB_Th) for i in range(np.min((int(nbth),self.NB_Th)))]
                    self.ExternalPreSynaptic_Cell_AMPA_Th.append(np.asarray(x1))
                    # connectivitymatrix[-1] = np.hstack((connectivitymatrix[-1], -np.asarray(x1) - 1))
                else:
                    self.ExternalPreSynaptic_Cell_AMPA_Th.append(np.asarray([]))

        # self.updateCell.something_happened.emit(-1)
        return connectivitymatrix, connectivityweight

    @staticmethod
    @njit
    def find_intersection(r1, r2, d):
        rad1sqr = r1 * r1
        rad2sqr = r2 * r1

        if d == 0:
            r3 = min(r1, r2)
            return math.pi * r3 * r3

        angle1 = (rad1sqr + d * d - rad2sqr) / (2 * r1 * d)
        angle2 = (rad2sqr + d * d - rad1sqr) / (2 * r2 * d)

        if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
            theta1 = math.acos(angle1) * 2
            theta2 = math.acos(angle2) * 2
            area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
            area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))

            return area1 + area2
        elif angle1 < -1 or angle2 < -1:
            r3 = min(r1, r2)
            return math.pi * r3 * r3
        return 0

    @staticmethod
    @njit
    def find_intersection_high(S_z, T_z, r1, r2, d):
        h = np.minimum(S_z[1], T_z[1]) - np.maximum(S_z[0], T_z[0])
        if h > 0:
            overlap = find_intersection(r1, r2, d) * h
            return overlap, 1
        else:
            return 0., 0.

    @staticmethod
    @vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
    def find_intersection_high_vectorize(S_z0,S_z1, T_z0,T_z1,r1,r2,d):
        h = np.minimum(S_z1, T_z1) - np.maximum(S_z0, T_z0)
        if h > 0:
            if r1+r2 < d:
                return 0.
            if d == 0:
                r3 = min(r1, r2)
                return math.pi * r3 * r3
            rad1sqr = r1 * r1
            rad2sqr = r2 * r2
            d2 = d * d
            angle1 = (rad1sqr + d2 - rad2sqr) / (2 * r1 * d)
            angle2 = (rad2sqr + d2 - rad1sqr) / (2 * r2 * d)
            if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
                theta1 = math.acos(angle1) * 2
                theta2 = math.acos(angle2) * 2
                area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
                area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))
                return area1 + area2
            elif angle1 < -1 or angle2 < -1:
                r3 = min(r1, r2)
                return math.pi * r3 * r3
            return 0.
        else:
            return 0.

    @staticmethod
    @vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64, float64)])
    def find_conic_intersection_high_vectorize(S_z0,S_z1, T_z0,T_z1,r1,r2,d, step, cone_sens):
        h = np.minimum(S_z1, T_z1) - np.maximum(S_z0, T_z0)
        if h > 0:
            heith = T_z1 - T_z0
            r = np.arange(step, heith, step) / heith * r2
            h2 = np.arange(step, heith, step)
            if cone_sens == 0:
                d2 = T_z0 + h2
                r = r[np.bitwise_and(d2 >= S_z0, d2 <= S_z1)]
            elif cone_sens ==1:
                d2 = T_z1 - h2
                r = r[np.bitwise_and(d2 >= S_z0, d2 <= S_z1)]
            o = 0.
            for ids in range(len(r)):
                r3 = r[ids]
                if r1 + r3 < d:
                    continue
                if d == 0:
                    r4 = min(r1, r3)
                    o += math.pi * r4 * r4
                    continue
                rad1sqr = r1 * r1
                rad2sqr = r3 * r3
                d2 = d * d
                angle1 = (rad1sqr + d2 - rad2sqr) / (2 * r1 * d)
                angle2 = (rad2sqr + d2 - rad1sqr) / (2 * r3 * d)
                if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
                    theta1 = math.acos(angle1) * 2
                    theta2 = math.acos(angle2) * 2
                    area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
                    area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))
                    o +=  area1 + area2
                    continue
                elif angle1 < -1 or angle2 < -1:
                    r4 = min(r1, r3)
                    o += math.pi * r4 * r4
                    continue
            return o
        else:
            return 0.


    def IsConnected(self,Source,Sourcepos,Target,Targetpos,d):
        step = 10.  # micrometeres
        connected = 0
        MeanPos =0
        overlap = 0.0
        DendConn = 0
        ConnPos = []


        if Source.type in [0, 3, 4]:  # PC/ VIP or RLN
            S_Z0 = Sourcepos[2] + Source.AX_down # lower threshold
            S_Z1 = Sourcepos[2] + Source.AX_up  # upper threshold

            if S_Z0 < 0:#kepp axon in the cortical columnlength
                S_Z0 = 0.
            if S_Z1 > self.L:
                S_Z1 = self.L+0.

            # if PC ---> PC ----> #dendrite targeting
            if Target.type == 0:  # PC
                DendConn = 1
                if Target.subtype in [1,2]:  # IPC or UPC
                    # first cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))




                    #  second cylinder
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))






                elif Target.subtype == 3:  # BPC
                    # cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    o=self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Adend_w, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))

                    # cylinder haut
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, 10, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))

                    # cylinder bas
                    T_Z0_n = Targetpos[2] + Target.c3_down
                    T_Z1_n = Targetpos[2] + Target.c3_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, 10, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))

                elif Target.subtype == 4:  # SSC
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))


                elif Target.subtype == 0: # TPC
                    # first cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))

                    #  second cylinder
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    o= self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))

                    #  cone
                    T_Z0_n = Targetpos[2] + Target.c3_down
                    T_Z1_n = Targetpos[2] + Target.c3_up
                    o= self.find_conic_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2, Target.r3, d, step, 0)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([S_Z0,T_Z0_n]),np.min([S_Z1,T_Z1_n])]))


            elif Target.type in [1, 2, 3, 4]:  # PV, SST, VIP or RLN
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)



        ###interneurons
        ###PV
        elif Source.type == 1:  # PV

            if Source.subtype == 0:  # BC  only inside the layer
                S_Z0 = Sourcepos[2] + Source.AX_down
                S_Z1 = Sourcepos[2] + Source.AX_up
                if S_Z0 < 0:  # kepp axon in the cortical columnlength
                    S_Z0 = 0.
                if S_Z1 > self.L:
                    S_Z1 = self.L + 0.

                if Target.type == 0:  # PV basket -->PC targetssoma
                    if  Target.layer == 2  and  Target.subtype == 4 : #Si layeur IV et stellate
                        T_Z0_n = Targetpos[2] - Target.Bdend_l / 4
                        T_Z1_n = Targetpos[2] + Target.Bdend_l / 4
                        overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2, Target.Bdend_w / 4, d)


                    else:  #
                        T_Z0_n = Targetpos[2] - Target.Bdend_l / 2
                        T_Z1_n = Targetpos[2] + Target.hsoma
                        overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.dsoma, d)


                elif Target.type in [1, 2, 3, 4]:  # PV-->PV/SST/VIP/RLN
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)



            elif Source.subtype == 1:  # PV Chandelier -> PC
                S_Z0 = Sourcepos[2] + Source.AX_down
                S_Z1 = Sourcepos[2] + Source.AX_up

                if Target.type == 0:  # PV Chandelier -->PC targetssoma
                    T_Z0_n = Targetpos[2] - Target.hsoma
                    T_Z1_n = Targetpos[2]
                    overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2, Target.dsoma / 2, d)


                elif Target.type == 1 in [1, 2, 3, 4]:  # PV-->PV/SST/VIP/RLN
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    overlap += self.find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)



        ####
        # SST
        elif Source.type == 2:  # SST
            Sbot_Z0 = Sourcepos[2] + Source.c1_down
            Sbot_Z1 = self.L - self.Layer_d[0]
            if Sbot_Z0 < 0:#kepp axon in the cortical columnlength
                Sbot_Z0 = 0.
            if Sbot_Z1 > self.L:
                Sbot_Z1 = self.L+0.
            Stop_Z0 = Sbot_Z1
            Stop_Z1 = self.L
            if Sbot_Z0 < 0:#kepp axon in the cortical columnlength
                Sbot_Z0 = 0.
            if Sbot_Z1 > self.L:
                Sbot_Z1 = self.L+0.

            if Target.type == 0:  # SST -> PC
                DendConn = 1

                if Target.subtype in [1, 2]:  # IPC or UPC
                    # first cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up

                    o = self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                    overlap +=o
                    if o>0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0,T_Z0_n]),np.min([Sbot_Z1,T_Z1_n])]))
                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r1, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))


                    # cylinder
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    #first axon
                    o = self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    #second axon
                    o = self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))



                elif Target.subtype == 3:  # BPC
                    # cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    #first axon
                    o= self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Adend_w, d)
                    overlap += o
                    if o>0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0,T_Z0_n]),np.min([Sbot_Z1,T_Z1_n])]))

                    #second axon
                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2 , Target.Adend_w, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))

                    # cylinder bas
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    o= self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2 , Target.r2, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))

                    # cylinder haut
                    T_Z0_n = Targetpos[2] + Target.c3_down
                    T_Z1_n = Targetpos[2] + Target.c3_up

                    o= self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2,d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2,d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))



                elif Target.subtype == 4:  # SSC
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up

                    o= self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w,d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.Bdend_w, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))



                elif Target.subtype == 0:  # TPC
                    # first cylinder
                    T_Z0_n = Targetpos[2] + Target.c1_down
                    T_Z1_n = Targetpos[2] + Target.c1_up
                    o = self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o = self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r1, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))



                    # cylinder
                    T_Z0_n = Targetpos[2] + Target.c2_down
                    T_Z1_n = Targetpos[2] + Target.c2_up
                    o= self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o= self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))

                    # cone
                    T_Z0_n = Targetpos[2]  + Target.c3_down
                    T_Z1_n = Targetpos[2] + Target.c3_up
                    o = self.find_conic_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r3, d, step, 0)
                    overlap += o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Sbot_Z0, T_Z0_n]), np.min([Sbot_Z1, T_Z1_n])]))

                    o = self.find_conic_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r3, d, step, 0)
                    overlap +=o
                    if o > 0:
                        ConnPos.append(np.mean([np.max([Stop_Z0, T_Z0_n]), np.min([Stop_Z1, T_Z1_n])]))


            #     # if SST--->PV
            elif Target.type in [1, 2, 3, 4]:  # PV insidelayer
                T_Z0_n = Targetpos[2] +Target.c1_down
                T_Z1_n = Targetpos[2] +Target.c1_up
                overlap += self.find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)
                overlap += self.find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.Bdend_w,d)

        if overlap > 0:
            connected = 1
            if DendConn==1:
                MeanPos=np.mean(ConnPos)
            else:
                MeanPos=0

        ################
        #        print(connected)
        #        print(overlap)
        return connected, overlap,MeanPos


    def Get_max_width_vect(self, Cell, z, Z0):
        d = z - Z0

        if Cell.subtype == 0:  # tufted
            r = np.zeros(d.shape)
            r[d<Cell.d1] = (Cell.d1-d[d<Cell.d1])*self.tanpix40p180
            r[(d>Cell.d1)&(d<(Cell.d1+Cell.d2))] = 10
            r[(d>(Cell.d1+Cell.d2))&(d<(Cell.Adend_l))] = (d[(d>(Cell.d1+Cell.d2))&(d<(Cell.Adend_l))]-Cell.d1-Cell.d2)*self.tanpix50p180
        elif Cell.subtype == 1: #untufted
            r = np.zeros(d.shape)
            r[d<Cell.d1] = (Cell.d1-d[d<Cell.d1])*self.tanpix40p180
            r[(d>Cell.d1)&(d<Cell.Adend_l)] = 10
        elif Cell.subtype == 2: #Inversed
            r = np.ones(d.shape) * 10
            r[(d>Cell.d3+Cell.d2)] = (d[(d>Cell.d3+Cell.d2)]-(Cell.d3+Cell.d2))*self.tanpix40p180
        elif Cell.subtype == 3: #Bipolar
            r = np.ones(d.shape) * 10
            r[(d < Cell.Adend_l/6)&(d > (-Cell.Adend_l/6))] = Cell.Adend_w
        elif Cell.subtype == 4:  #SSC
            r = np.ones(d.shape) * Cell.Adend_w
        return r

    def Get_max_width_vect2(self, Cell, z, Z0):
        d = z - Z0

        if Cell.subtype == 0:  # tufted
            r = np.zeros(d.shape)
            self.Get_max_width_vect_subtype0(d,Cell.d1,Cell.d2,Cell.Adend_l,self.tanpix40p180,self.tanpix50p180, r)

        elif Cell.subtype == 1: #untufted
            r = np.zeros(d.shape)
            self.Get_max_width_vect_subtype1(d, Cell.d1, Cell.Adend_l, self.tanpix40p180, r)
        elif Cell.subtype == 2: #Inversed
            r = np.ones(d.shape) * 10
            self.Get_max_width_vect_subtype2(d, Cell.d2, Cell.d3, self.tanpix40p180, r)
        elif Cell.subtype == 3: #Bipolar
            r = np.ones(d.shape) * 10
            self.Get_max_width_vect_subtype3(d, Cell.Adend_l, Cell.Adend_w, r)
        elif Cell.subtype == 4:  #SSC
            r = np.ones(d.shape) * Cell.Adend_w
        return r


    @staticmethod
    @guvectorize(["float64[:], float64, float64, float64, float64, float64, float64[:]"], '(n),(),(),(),(),()->(n)')
    def Get_max_width_vect_subtype0(d,d1,d2,Adend_l,tanpix40p180,tanpix50p180, r):
        r[d < d1] = (d1 - d[d < d1]) * tanpix40p180
        r[(d > d1) & (d < (d1 + d2))] = 10
        r[(d > (d1 + d2)) & (d < (Adend_l))] = (d[(d > (d1 + d2)) & (d < (Adend_l))] - d1 - d2) * tanpix50p180

    @staticmethod
    @guvectorize(["float64[:], float64, float64, float64, float64[:]"], '(n),(),(),()->(n)')
    def Get_max_width_vect_subtype1(d,d1,Adend_l,tanpix40p180, r):
        r[d < d1] = (d1 - d[d < d1]) * tanpix40p180
        r[(d > d1) & (d < Adend_l)] = 10

    @staticmethod
    @guvectorize(["float64[:], float64, float64, float64, float64[:]"], '(n),(),(),()->(n)')
    def Get_max_width_vect_subtype2(d, d2,d3, tanpix40p180, r):
        r[(d > d3 + d2)] = (d[(d > d3 + d2)] - (d3 + d2)) * tanpix40p180

    @staticmethod
    @guvectorize(["float64[:], float64, float64, float64[:]"], '(n),(),()->(n)')
    def Get_max_width_vect_subtype3(d, Adend_l, Adend_w, r):
        r[(d < Adend_l / 6) & (d > (-Adend_l / 6))] = Adend_w

    def Get_max_width(self,Cell,z,Z0):
        d = z - Z0
        if Cell.subtype==0: #tufted
            if d<Cell.d1:
                return (Cell.d1-d)*math.tan(40*math.pi/180)

            elif (d>Cell.d1)&(d<(Cell.d1+Cell.d2)):
                return 10

            elif (d>(Cell.d1+Cell.d2))&(d<(Cell.Adend_l)):
                return (d-Cell.d1-Cell.d2)*math.tan(50*math.pi/180)

            else:
                return 0


        elif Cell.subtype == 1: #untufted
            if d<Cell.d1:
                return (Cell.d1-d)*math.tan(40*math.pi/180)

            if (d>Cell.d1)&(d<Cell.Adend_l):
                return 10

        elif Cell.subtype == 2: #Inversed
            if (d>Cell.d3+Cell.d2):
                return (d-(Cell.d3+Cell.d2))*math.tan(40*math.pi/180)
            else:
                return 10

        elif Cell.subtype == 3: #Bipolar
            if (d < Cell.Adend_l/6)&(d > (-Cell.Adend_l/6)):
                return Cell.Adend_w
            else:
                return 10

        elif Cell.subtype == 4: #SSC
            return Cell.Adend_w

    def Get_cell_layer(self,CellNb):
        if CellNb<self.Nbcells[0]:
            return 0
        else:
            if CellNb<self.Nbcells[1]:
                return 1
            else:
                if CellNb<self.Nbcells[2]:
                    return 2
                else:
                    if CellNb<self.Nbcells[3]:
                        return 3
                    else:
                        return 4



    def All2layer(self, indexall, NB_Cells, NB_pyr, NB_PV, NB_SST, NB_VIP, NB_RLN,celltype):  # tranform index of a cell in the network into layer and index in the layer
        layer = []
        new_i = []  # index in the layer
        i = []  # index per type
        if indexall < NB_Cells[0]:  # layer 1
            layer = 0
            new_i=indexall
        elif indexall >= NB_Cells[0:1] and indexall < np.sum(NB_Cells[0:2]):  # layer 2/3
            layer = 1
            new_i = indexall - NB_Cells[0]
        elif indexall >= np.sum(NB_Cells[0:2]) and indexall < np.sum(NB_Cells[0:3]):  # Layer 4
            layer = 2
            new_i = indexall - np.sum(NB_Cells[0:2])
        elif indexall >= np.sum(NB_Cells[0:3]) and indexall < np.sum(NB_Cells[0:4]):  # Layer 5
            layer = 3
            new_i = indexall - np.sum(NB_Cells[0:3])
        elif indexall >= np.sum(NB_Cells[0:4]) and indexall < np.sum(NB_Cells[0:5]):  # Layer 6
            layer = 4
            new_i = indexall - np.sum(NB_Cells[0:4])


        type = int(celltype[layer][new_i])


        if type == 0:  # PC
            i = new_i + np.sum(NB_pyr[0:layer])
        if type == 1:  # PV
            i = new_i + np.sum(NB_PV[0:layer]) - np.sum(NB_pyr[layer])
        if type == 2:  # SST
            i = new_i + np.sum(NB_SST[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer])
        if type == 3:  # VIP
            i = new_i + np.sum(NB_VIP[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer]) - np.sum(NB_SST[layer])
        if type == 4:  # RLN
            i = new_i + np.sum(NB_RLN[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer]) - np.sum(NB_SST[layer]) - np.sum(NB_VIP[layer])


        return layer, type, i



    def Generate_Stims(self, I_inj=60, tau=4, stimDur=3, nbstim=5, varstim=12, freq=1.5,StimStart = 0):
        print('computing stim signals')
        # nb = int(1e-3 * self.T*freq) #frequency of fast ripples
        if not self.seed == 0:
            np.random.seed(self.seed)
        # else:
        #     np.random.seed()
        nb = 1
        nb_Stim_Signals = self.NB_DPYR
        nbOfSamplesStim = int(stimDur / self.dt)
        npos = np.round(np.random.beta(2, 2, nb) * self.nbEch / nb) + np.linspace(0, self.nbEch, nb)  # position of HFOs
        # npos=self.nbEch/2
        Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))
        varianceStim = varstim / self.dt
        t = np.arange(self.nbEch) * self.dt
        y = np.zeros(t.shape)
        for tt, tp in enumerate(t):
            if tt <= nbOfSamplesStim:
                y[tt] = (1. - np.exp(-tp / tau)) * I_inj
            if tt > nbOfSamplesStim:
                y[tt] = (np.exp(-(tp - nbOfSamplesStim * self.dt) / tau)) * y[nbOfSamplesStim - 1]
        # print(n)
        # print(n*self.dt)
        if StimStart == 0:
            N0 = int(self.nbEch / 2)
        else:
            N0 = int(StimStart / self.dt)
        for St in range(nb_Stim_Signals):
            yt = np.zeros(self.nbEch)
            for j in range(nb):
                y2 = np.zeros(self.nbEch)
                for i in range(nbstim):
                    # N0=int(npos[j])
                    # N0 = int(self.nbEch / 2)
                    inst = int(np.round((np.random.normal(0, varianceStim))))
                    t0 = N0 + inst
                    # print(inst)
                    if t0 == 0:
                        t0 = 0
                    elif t0 < 0:
                        t0 = N0 + int(varianceStim * inst / abs(2 * inst))
                    elif t0 > self.nbEch - 1:
                        if inst > 0:
                            t0 = N0 + int(varianceStim * inst / abs(2 * inst))
                        else:
                            t0 = N0
                    try:
                        y2[range(t0, len(y2))] = y2[range(t0, len(y2))] + y[range(len(y2) - t0)]
                    except:
                        pass
                yt += y2
            Stim_Signals_out[St, :] = yt

        # for j in range(nb):
        #     N0 = int(npos[j])
        #     self.Stim_Signals[:, int(N0-2*varianceStim):int(N0+2*varianceStim)]=Stim_Signals_out[:, int(N0-2*varianceStim):int(N0+2*varianceStim)]
        #self.Stim_Signals = Stim_Signals_out + self.Stim_InputSignals
        self.Stim_Signals = Stim_Signals_out

        return self.Stim_Signals

    def set_seed(self,seed):
        self.seed=seed

    def Generate_input(self, I_inj=25, tau=4, stimDur=3, nbstim=5, deltamin=14, delta=18):
        if not self.seed == 0:
            np.random.seed(self.seed)
        # else:
        #     np.random.seed()
        nb_Stim_Signals = self.NB_Th
        nbOfSamplesStim = int(stimDur / self.dt)  # nb of samples in one stimulation signal
        Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))  # all signal of stim
        t = np.arange(self.nbEch) * self.dt
        y = np.zeros(t.shape)
        for tt, tp in enumerate(t):
            if tt <= nbOfSamplesStim:
                y[tt] = (1. - np.exp(-tp / tau)) * I_inj
            if tt > nbOfSamplesStim:
                y[tt] = (np.exp(-(tp - nbOfSamplesStim * self.dt) / tau)) * y[nbOfSamplesStim - 1]

        nbsigs = int(self.T / (2 * delta))
        for St in range(nb_Stim_Signals):
            yt = np.zeros(self.nbEch)
            n = np.round(np.random.uniform(delta / self.dt, (self.T - delta) / self.dt, nbsigs))
            d = np.round(np.random.uniform(deltamin, delta, nbsigs)) / self.dt
            for k in range(int(nbsigs)):
                y2 = np.zeros(self.nbEch)
                N = int(n[k])
                D = int(d[k])
                for i in range(nbstim):
                    inst = int(np.round(np.random.normal(0, D)))
                    if inst == 0:
                        continue
                    t0 = N + inst
                    if t0 < 0:
                        t0 = N + D * int(inst / abs(2 * inst))
                    if t0 > self.nbEch - 1:
                        t0 = N + D * int(inst / abs(2 * inst))
                    y2[range(t0, len(y2))] = y2[range(t0, len(y2))] + y[range(len(y2) - t0)]
                yt += y2
            Stim_Signals_out[St, :] = yt
        print('computing stim signals....finished')
        self.Stim_InputSignals = Stim_Signals_out
        return self.Stim_InputSignals

    def create_cells(self):
        self.presynaptic_instance = presynaptic_class(
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Cell_AMPA],
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Cell_GABA],
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Soma_AMPA],
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Soma_GABA_d],
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Soma_GABA_s],
            [np.array(l, dtype=np.int32) for l in self.PreSynaptic_Soma_GABA_a],
            [np.array(l, dtype=np.int32) for l in self.ExternalPreSynaptic_Cell_AMPA_DPYR],
            [np.array(l, dtype=np.int32) for l in self.ExternalPreSynaptic_Cell_AMPA_Th],
            [np.array(l, dtype=np.float64) for l in self.PreSynapticWeight_AMPA],
            [np.array(l, dtype=np.int32) for l in self.PreSynapticPos_AMPA],
            [np.array(l, dtype=np.float64) for l in self.PreSynapticWeight_GABA],
            [np.array(l, dtype=np.int32) for l in self.PreSynapticPos_GABA]
        )

        self.List_DPYR = []
        self.List_Th = []
        self.List_PYR = []
        self.List_PV = []
        self.List_SST = []
        self.List_VIP = []
        self.List_RLN = []
        self.List_Neurone_param = []

        ###DPC Define the PYR cells in the distant cortex in percentage
        layer_ratio = 0.4  # the ratio of PYR cells from layer 2/3 with respect to 5/6.
        if self.NB_DPYR == 0:
            self.List_DPYR = [PC_neo3.pyrCellneo(1)]
        for i in range(int(self.NB_DPYR * layer_ratio)):  # define PYR cells sublayer types
            self.List_DPYR.append(PC_neo3.pyrCellneo(1))  # from layer 2
        for i in range(int(self.NB_DPYR * layer_ratio), self.NB_DPYR + 1):  # from layer 5
            self.List_DPYR.append(PC_neo3.pyrCellneo(3))

        ####TH Cells represent the input from thalamus with input signals

        for i in range(self.NB_Th):
            self.List_Th.append(PC_neo3.pyrCellneo(1))  # TO do adjust parameters

        ###PC
        if np.sum(self.NB_PYR) == 0:
            self.List_PYR = [PC_neo3.pyrCellneo(1)]
        count = 0
        PCsubtypes_Per = np.cumsum(self.PCsubtypes_Per,axis=1)
        for l in range(1, 5):
            for i in range(self.NB_PYR[l]):  # define PYR cells sublayer types
                self.List_PYR.append(PC_neo3.pyrCellneo(l))
                # Check subtypes and load lambda values
                if i < PCsubtypes_Per[l][0]:
                    subtype = 0  # TPC
                elif (i >= PCsubtypes_Per[l][0]) and (i < PCsubtypes_Per[l][1]):
                    subtype = 1  # UPC
                elif (i >= PCsubtypes_Per[l][1]) and (i < PCsubtypes_Per[l][2]):
                    subtype = 2  # IPC
                elif (i >= PCsubtypes_Per[l][2]) and (i < PCsubtypes_Per[l][3]):
                    subtype = 3  # BPC
                elif (i >= PCsubtypes_Per[l][3]) and (i < PCsubtypes_Per[l][4]):
                    subtype = 4  # SSC
                self.List_PYR[count].Lambda_s[2] = self.List_Lambda_s[l-1][subtype]
                self.List_PYR[count].Lambda_a[2] = self.List_Lambda_s[l-1][subtype]
                self.List_PYR[count].Lambda_d[2] = self.List_Lambda_d[l-1][subtype]
        ######PV
        if np.sum(self.NB_PV) == 0:
            self.List_PV = [PV_neo.PVcell()]
        for i in range(np.sum(self.NB_PV)):
            self.List_PV.append(PV_neo.PVcell())
        #######SST
        if np.sum(self.NB_SST) == 0:
            self.List_SST = [SST_neo.SSTcell()]
        for i in range(np.sum(self.NB_SST)):
            self.List_SST.append(SST_neo.SSTcell())
        #######VIP
        if np.sum(self.NB_VIP) == 0:
            self.List_VIP = [VIP_neo.VIPcell()]
        for i in range(np.sum(self.NB_VIP)):
            self.List_VIP.append(VIP_neo.VIPcell())
        #######RLN
        if np.sum(self.NB_RLN) == 0:
            self.List_RLN = [RLN_neo.RLNcell()]
        for i in range(np.sum(self.NB_RLN)):
            self.List_RLN.append(RLN_neo.RLNcell())

        # initialize synaptic ODE
        #########################################################
        count = 0
        for i in range(np.sum(self.NB_RLN[0])):
            # print(self.PreSynaptic_Cell_AMPA)
            self.List_RLN[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(self.ExternalPreSynaptic_Cell_AMPA_Th[count])
            self.List_RLN[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
            self.List_RLN[i].init_vector()
            count += 1

        ###external input from th ad DPYR
        # nbDpyr = self.inputNB * np.array([0.25, 0.06, 0.05,0.01])
        # nbTh = self.inputNB * np.array([0.15,0.11,0.11,0.04])

        for l in range(1, 5):  # was 1 to 5 before to check if I corrected rigth
            for i in range(np.sum(self.NB_PYR[0:l]), self.NB_PYR[l] + np.sum(self.NB_PYR[0:l])):
                self.List_PYR[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_PYR[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_PYR[i].init_vector()
                count += 1
            for i in range(np.sum(self.NB_PV[0:l]), self.NB_PV[l] + np.sum(self.NB_PV[0:l])):
                self.List_PV[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_PV[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_PV[i].init_vector()
                count += 1
            for i in range(np.sum(self.NB_SST[0:l]), self.NB_SST[l] + np.sum(self.NB_SST[0:l])):
                self.List_SST[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_SST[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_SST[i].init_vector()
                count += 1
            for i in range(np.sum(self.NB_VIP[0:l]), self.NB_VIP[l] + np.sum(self.NB_VIP[0:l])):
                self.List_VIP[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_VIP[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_VIP[i].init_vector()
                count += 1

            for i in range(np.sum(self.NB_RLN[0:l]), self.NB_RLN[l] + np.sum(self.NB_RLN[0:l])):
                self.List_RLN[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_RLN[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_RLN[i].init_vector()
                count += 1

        for l in range(0, 5):
            Neurone_param = []
            for i in range(np.sum(self.NB_PYR[0:l]), self.NB_PYR[l] + np.sum(self.NB_PYR[0:l])):
                Neurone_param.append({s: getattr(self.List_PYR[i], s) for s in self.get_PYR_Variables()})
            for i in range(np.sum(self.NB_PV[0:l]), self.NB_PV[l] + np.sum(self.NB_PV[0:l])):
                Neurone_param.append({s: getattr(self.List_PV[i], s) for s in self.get_PV_Variables()})
            for i in range(np.sum(self.NB_SST[0:l]), self.NB_SST[l] + np.sum(self.NB_SST[0:l])):
                Neurone_param.append({s: getattr(self.List_SST[i], s) for s in self.get_SST_Variables()})
            for i in range(np.sum(self.NB_VIP[0:l]), self.NB_VIP[l] + np.sum(self.NB_VIP[0:l])):
                Neurone_param.append({s: getattr(self.List_VIP[i], s) for s in self.get_VIP_Variables()})
            for i in range(np.sum(self.NB_RLN[0:l]), self.NB_RLN[l] + np.sum(self.NB_RLN[0:l])):
                Neurone_param.append({s: getattr(self.List_RLN[i], s) for s in self.get_RLN_Variables()})
            self.List_Neurone_param.append(Neurone_param)
        self.UpdateModel()
        self.ImReady = True

    def Update_param_model(self):
        for l in range(0, 5):
            ind = 0
            for i in range(np.sum(self.NB_PYR[0:l]), self.NB_PYR[l] + np.sum(self.NB_PYR[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_PYR[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.NB_PV[0:l]), self.NB_PV[l] + np.sum(self.NB_PV[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_PV[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.NB_SST[0:l]), self.NB_SST[l] + np.sum(self.NB_SST[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_SST[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.NB_VIP[0:l]), self.NB_VIP[l] + np.sum(self.NB_VIP[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_VIP[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.NB_RLN[0:l]), self.NB_RLN[l] + np.sum(self.NB_RLN[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_RLN[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1



    def Reset_states(self):
        self.UpdateModel()

    def create_Hyper_Cluster(self, radius=10, EGaba=-50,gGaba=25,gAMPA=8,gNMDA=0.15,BASEGaba=-75,center=None):
        HypPyr_Idx = []
        if center is None:
            center = np.array([self.dx / 2, self.dy / 2, self.dz / 2])
        distances = distance.cdist(self.CellPosition[0:self.Nb_all_cells - 1, :], [center], 'euclidean')
        #print(len(distances))
        for i, d in enumerate(distances):
            if d <= radius:
                if self.List_Neurone_PYR[i].Type == 1:
                    self.List_Neurone_PYR[i].E_GABA = EGaba
                    self.List_Neurone_PYR[i].g_GABA = gGaba
                    self.List_Neurone_PYR[i].g_AMPA = gAMPA
                    self.List_Neurone_PYR[i].g_NMDA = gNMDA
                    HypPyr_Idx.append(i)

        distances = distance.cdist(self.CellPosition[self.Nb_all_cells:self.NbInter+self.Nb_all_cells-1, :], [center], 'euclidean')
        #print(len(distances))
        for i, d in enumerate(distances):
            if d <= radius:
                self.List_Neurone_BAS[i].E_GABA =BASEGaba

        #for k in range(self.Nb_all_cells):
            #print('New')
            #print(self.List_Neurone_PYR[k].E_GABA)
        return np.asarray(HypPyr_Idx)

    def Rest_clusters(self):
        for i in range(self.Nb_all_cells):
            if self.List_Neurone_PYR[i].Type == 1:
                self.List_Neurone_PYR[i].E_GABA = -75

    def create_mutiple_Hyper_Clusters(self, nbclusters=2, radius=50,EGaba=[-50,-50],gGaba=[25, 25]):
        HypPyr_Idx = []
        center = np.array([self.dx / 2, self.dy / 2, self.dz / 2])
        Ccenter = []
        Cluster_lists = []
        for i in range(nbclusters):
            choose = True
            while (choose):
                C = np.random.choice(range(self.Nb_of_PYR_Stimulated, self.Nb_PYR_Cells))
                Cpos = self.CellPosition[C, :]
                if np.abs(Cpos[0] - self.dx / 2) < (self.dx / 2 - (radius + 30)):
                    if np.abs(Cpos[1] - self.dy / 2) < (self.dy / 2 - (radius + 30)):
                        if not Ccenter == []:
                            s = np.asarray(Ccenter)
                            dis = np.sort(distance.cdist(s, [Cpos, Cpos], 'euclidean')[0, :])
                            #print(dis)
                            if dis[0] > 2 * radius:
                                choose = False
                        else:
                            choose = False
            Ccenter.append(Cpos)
            cluster = self.create_Hyper_Cluster(radius, EGaba=EGaba[i],gGaba=gGaba[i],center=Cpos)
            Cluster_lists.append(cluster)
        return Cluster_lists
     ##################################################
    def UpdateModel(self):

        for i in range(self.NB_Th):
            self.List_Th[i].init_I_syn()
            self.List_Th[i].init_vector()
            self.List_Th[i].setParameters()

        for i in range(self.NB_DPYR):
            self.List_DPYR[i].init_I_syn()
            self.List_DPYR[i].init_vector()
            self.List_DPYR[i].setParameters()

        for i in range(np.sum(self.NB_PYR)):
            self.List_PYR[i].init_I_syn()
            self.List_PYR[i].init_vector()
            self.List_PYR[i].setParameters()

        for i in range(np.sum(self.NB_PV)):
            self.List_PV[i].init_I_syn()
            self.List_PV[i].init_vector()
            self.List_PV[i].setParameters()

        for i in range(np.sum(self.NB_SST)):
            self.List_SST[i].init_I_syn()
            self.List_SST[i].init_vector()
            self.List_SST[i].setParameters()

        for i in range(np.sum(self.NB_VIP)):
            self.List_VIP[i].init_I_syn()
            self.List_VIP[i].init_vector()
            self.List_VIP[i].setParameters()

        for i in range(np.sum(self.NB_RLN)):
            self.List_RLN[i].init_I_syn()
            self.List_RLN[i].init_vector()
            self.List_RLN[i].setParameters()

    def Generate_EField(self, EField, stimOnOff, Constant = False):

        index = 0
        for i, list_type in enumerate(self.List_celltypes):
            for k, type in enumerate(list_type):
                if type == 0:
                    neuron = self.List_PYR[index]
                    if not Constant:
                        pos = self.Cellpos[i][k]
                        x = np.argmin(np.abs(EField['x'] - pos[0]))
                        y = np.argmin(np.abs(EField['y'] - pos[1]))
                        z = np.argmin(np.abs(EField['z'] - pos[2]))
                        E = EField['Er'][x,y,z,:]
                    else:
                        E =  EField['Er']
                    neuron.projection_E(E,E,E)

                    neuron.update_EField_Stim(stimOnOff)
                    neuron.OnOff = int(stimOnOff)

    def Generate_EField_Stim(self,type,A,F,stimOnOff,Start = None, Length = None ):
        Stim_Signals_out = np.zeros(self.nbEch)
        t = np.arange(self.nbEch) *(self.dt/1000)
        if not stimOnOff:
            pass
        elif type == 'Constant':
            Stim_Signals_out = Stim_Signals_out + A
        elif type == 'Sinusoidal':
            Stim_Signals_out = A * np.sin(2 * np.pi * F * t)
        elif type == 'rectangular':
            Stim_Signals_out = scipy.signal.square(2 * np.pi * F * t, 0.5)
        elif type == 'triangular':
            Stim_Signals_out = scipy.signal.sawtooth(2 * np.pi * F * t, 0.5)
        if not Start is None and not Length is None:
            start_ind = int(Start/self.dt)
            if start_ind < 0:
                start_ind = 0
            elif start_ind > self.nbEch:
                start_ind = self.nbEch-1

            end_ind = int((Start+Length)/self.dt)
            if end_ind > self.nbEch:
                end_ind = self.nbEch-1
            elif end_ind < 0:
                end_ind = 0
            Stim_Signals_out[:start_ind] = 0.
            Stim_Signals_out[end_ind:] = 0.

        self.Stim_EField = Stim_Signals_out

#        plt.plot(self.Stim_EField)
#        plt.show()



    def runSim(self):

        # self.UpdateModel()
        print('computing signals...')
        # self.ApplyParamDict()
        t = np.arange(self.nbEch) * self.dt
        self.DpyrVs = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.DpyrVd = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.ThpyrVs = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.ThpyrVd = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.pyrVd = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrVs = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrVa = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSE_d1 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSE_d23 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSE_d4 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSE_d5 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSE_d6 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_d1 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_d23 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_d4 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_d5 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_d6 = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_s = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_a = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.PV_Vs = np.zeros((np.sum(self.NB_PV), len(t)))
        self.SST_Vs = np.zeros((np.sum(self.NB_SST), len(t)))
        self.VIP_Vs = np.zeros((np.sum(self.NB_VIP), len(t)))
        self.RLN_Vs = np.zeros((np.sum(self.NB_RLN), len(t)))
        self.DPYR_Vs = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.Th_Vs = np.zeros((np.sum(self.NB_Th), len(t)))
        nbcells = np.sum(self.Layer_nbCells)
        layerS = np.zeros(nbcells,dtype=int)
        typeS = np.zeros(nbcells,dtype=int)
        indexS = np.zeros(nbcells,dtype=int)
        for ind in range(nbcells):
            layerS[ind], typeS[ind], indexS[ind] = self.All2layer(ind, self.Layer_nbCells, self.NB_PYR, self.NB_PV, self.NB_SST, self.NB_VIP,
                                               self.NB_RLN, self.List_celltypes)

        if 1:
            t0 = time.time()
            self.t, self.pyrVs, self.pyrVd, self.pyrVa, self.pyrPPSE_d1, self.pyrPPSE_d23, self.pyrPPSE_d4, self.pyrPPSE_d5, self.pyrPPSE_d6, self.pyrPPSI_d1, self.pyrPPSI_d23, self.pyrPPSI_d4, self.pyrPPSI_d5, self.pyrPPSI_d6, self.pyrPPSI_s, self.pyrPPSI_a, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs = Model_compute(np.int(self.nbEch),
                                                                                             np.float(self.dt),
                                                                                             np.float(self.tps_start),
                                                                                             self.Layer_nbCells,
                                                                                             self.NB_PYR,
                                                                                             self.NB_PV,
                                                                                             self.NB_SST,
                                                                                             self.NB_VIP,
                                                                                             self.NB_RLN,
                                                                                             self.NB_DPYR,
                                                                                             self.NB_Th,
                                                                                             np.int(self.inputNB),
                                                                                             self.List_PYR,
                                                                                             self.List_PV,
                                                                                             self.List_SST,
                                                                                             self.List_VIP,
                                                                                             self.List_RLN,
                                                                                             self.List_DPYR,
                                                                                             self.List_Th,
                                                                                             self.Stim_Signals,
                                                                                             self.Stim_InputSignals,
                                                                                             self.Stim_EField,
                                                                                             self.presynaptic_instance,
                                                                                             self.pyrVs,
                                                                                             self.pyrVd,
                                                                                             self.pyrVa,
                                                                                             self.pyrPPSE_d1,
                                                                                             self.pyrPPSE_d23,
                                                                                             self.pyrPPSE_d4,
                                                                                             self.pyrPPSE_d5,
                                                                                             self.pyrPPSE_d6,
                                                                                             self.pyrPPSI_d1,
                                                                                             self.pyrPPSI_d23,
                                                                                             self.pyrPPSI_d4,
                                                                                             self.pyrPPSI_d5,
                                                                                             self.pyrPPSI_d6,
                                                                                             self.pyrPPSI_s,
                                                                                             self.pyrPPSI_a,
                                                                                             self.PV_Vs,
                                                                                             self.SST_Vs,
                                                                                             self.VIP_Vs,
                                                                                             self.RLN_Vs,
                                                                                             self.DPYR_Vs,
                                                                                             self.Th_Vs,
                                                                                             layerS,
                                                                                             typeS,
                                                                                             indexS,
                                                                                             t,
                                                                                             np.int(self.seed))
            print(time.time() - t0)
            #run model


        return (self.t, self.pyrVs, self.pyrVd,self.pyrVa, self.pyrPPSE_d1, self.pyrPPSE_d23, self.pyrPPSE_d4, self.pyrPPSE_d5, self.pyrPPSE_d6, self.pyrPPSI_d1, self.pyrPPSI_d23, self.pyrPPSI_d4, self.pyrPPSI_d5, self.pyrPPSI_d6, self.pyrPPSI_s, self.pyrPPSI_a, self.PV_Vs, self.SST_Vs, self.VIP_Vs,self.RLN_Vs, self.DPYR_Vs, self.Th_Vs)

    def Compute_LFP_fonc(self):
        electrode_pos = np.array([0, self.D/2+20, self.L/2])

        #get principal cells positions
        CellPosition=[]
        for l in range(1,5):
            for i in range(self.NB_PYR[l]):
                CellPosition.append(self.Cellpos[l][i])
        # for i, cellpos in enumerate(CellPosition):
        #     cellpos[0] = cellpos[0] * self.somaSize
        #     cellpos[1] = cellpos[1] * self.somaSize

        Distance_from_electrode = distance.cdist([electrode_pos, electrode_pos], CellPosition, 'euclidean')[0, :]
        # vect direction??
        U = (CellPosition - electrode_pos) / Distance_from_electrode[:, None]

        Ssoma = np.pi * (self.somaSize / 2.) * ((self.somaSize / 2.) + self.somaSize * np.sqrt(5. / 4.))
        Stotal = Ssoma / self.p

        # potentials
        self.LFP = np.zeros(self.pyrVs.shape[1])
        Vs_d = self.pyrVs - self.pyrVd

        for k in range(self.Nb_PYR_Cells):
            i = k + self.Nb_of_PYR_Stimulated
            Vdi = np.zeros((len(Vs_d[i, :]), 3))
            Vdi[:, 2] = Vs_d[i, :]
            Vm = np.sum(Vdi * U[i, :], axis=1)
            Vm = Vm / (4. * self.sigma * 1e-3 * np.pi * Distance_from_electrode[i] * Distance_from_electrode[i])
            Vm = Vm * (self.dendriteSize + self.somaSize) / 2. * self.gc * Stotal
            self.LFP += Vm * 1e3
        return self.LFP


    def get_PYR_Variables(self):
        return PC_neo3.get_Variable_Names()

    def get_PV_Variables(self):
        return PV_neo.get_Variable_Names()

    def get_SST_Variables(self):
        return SST_neo.get_Variable_Names()

    def get_VIP_Variables(self):
        return VIP_neo.get_Variable_Names()

    def get_RLN_Variables(self):
        return RLN_neo.get_Variable_Names()

