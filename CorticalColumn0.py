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
import Connectivity
#import pyscenarios
import csv
#import pyedflib
import datetime
from numba import guvectorize,vectorize, float64, int64
from numba.experimental import jitclass
from numba import jit, njit, types, typeof
import Cell_morphology
import Column_morphology
import ComputeSim
# import circos
#import bigfloat
#bigfloat.exp(5000,bigfloat.precision(100))
from PyQt5.QtCore import *
import matplotlib as plt




class SenderObject(QObject):
    something_happened = pyqtSignal(np.float)

class SenderObjectInt(QObject):
    something_happened = pyqtSignal(np.int)

class CorticalColumn:
    updateTime = SenderObject()
    updateCell = SenderObjectInt()
    def __init__(self):
        #initialize cortical column parameters
        self.C=Column_morphology.Column(1) #0: human, 1:Rat, 0:mice
        self.D = self.C.D  # neocortical column diameter in micrometers
        self.Layer_d = self.C.L_th  # layers' thicknesses (L1-L2/3-L4-L5-L6)
        self.L=sum(self.Layer_d) #column length
        self.Layertop_pos=np.cumsum(self.Layer_d[::-1])
        self.Layer_nbCells=self.C.Layer_nbCells #nb of cells in each layer
        self.Layer_nbcells_pertype=self.C.Layer_nbCells_pertype # nb of each type cell in the layers PC,PV,SST,VIP,RLN
        self.Nbcells = int(np.sum(self.Layer_nbCells)) #total nb of cells
        print('Total cell number:',self.Nbcells)
        self.NB_PYR = self.Layer_nbcells_pertype[0]

        #subtypes of PYR cells
        self.PCsubtypes_Per=self.C.PCsubtypes_Per #TPC,UPC,IPC,BPC,SSC
        self.List_celltypes = self.C.List_celltypes
        self.List_cellsubtypes = self.C.List_cellsubtypes
        #####External afferences
        self.NB_DPYR=int(np.sum(self.NB_PYR)*0.07)
        self.NB_Th=int(np.sum(self.NB_PYR)*0.07)
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

        self.List_Lambda_s=self.C.List_Lambda_s
        self.List_Lambda_d=self.C.List_Lambda_d
        self.Conx={}

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

    def createFlatCEllpos(self):
        self.Cellposflat=np.vstack((self.Cellpos[0], self.Cellpos[1], self.Cellpos[2], self.Cellpos[3], self.Cellpos[4]))

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

        self.PreSynaptic_Cell_AMPA = self.Conx['PreSynaptic_Cell_AMPA']
        self.PreSynaptic_Cell_GABA = self.Conx['PreSynaptic_Cell_GABA']
        self.PreSynaptic_Soma_AMPA = self.Conx['PreSynaptic_Soma_AMPA']
        self.PreSynaptic_Soma_GABA_d = self.Conx['PreSynaptic_Soma_GABA_d']
        self.PreSynaptic_Soma_GABA_s = self.Conx['PreSynaptic_Soma_GABA_s']
        self.PreSynaptic_Soma_GABA_a = self.Conx['PreSynaptic_Soma_GABA_a']
        self.ExternalPreSynaptic_Cell_AMPA_DPYR =self.Conx['ExternalPreSynaptic_Cell_AMPA_DPYR']
        self.ExternalPreSynaptic_Cell_AMPA_Th = self.Conx['ExternalPreSynaptic_Cell_AMPA_Th']
        self.PreSynapticWeight_AMPA = self.Conx['PreSynapticWeight_AMPA']
        self.PreSynapticPos_AMPA = self.Conx['PreSynapticPos_AMPA']
        self.PreSynapticWeight_GABA = self.Conx['PreSynapticWeight_GABA']
        self.PreSynapticPos_GABA = self.Conx['PreSynapticPos_GABA']

        self.presynaptic_instance = ComputeSim.presynaptic_class(
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
            [np.array(l, dtype=np.int32) for l in self.PreSynapticPos_GABA])

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
        for l in range(1, 5):
            for i in range(self.NB_PYR[l]):  # define PYR cells sublayer types
                self.List_PYR.append(PC_neo3.pyrCellneo(l))
                # Check subtypes and load lambda values
                self.List_PYR[count].Lambda_s[2] = self.List_Lambda_s[l-1][self.C.List_cellsubtypes[l][i]]
                self.List_PYR[count].Lambda_a[2] = self.List_Lambda_s[l-1][self.C.List_cellsubtypes[l][i]]
                self.List_PYR[count].Lambda_d[2] = self.List_Lambda_d[l-1][self.C.List_cellsubtypes[l][i]]
        ######PV
        if np.sum(self.C.NB_PV) == 0:
            self.List_PV = [PV_neo.PVcell()]
        for i in range(np.sum(self.C.NB_PV)):
            self.List_PV.append(PV_neo.PVcell())
        #######SST
        if np.sum(self.C.NB_SST) == 0:
            self.List_SST = [SST_neo.SSTcell()]
        for i in range(np.sum(self.C.NB_SST)):
            self.List_SST.append(SST_neo.SSTcell())
        #######VIP
        if np.sum(self.C.NB_VIP) == 0:
            self.List_VIP = [VIP_neo.VIPcell()]
        for i in range(np.sum(self.C.NB_VIP)):
            self.List_VIP.append(VIP_neo.VIPcell())
        #######RLN
        if np.sum(self.C.NB_RLN) == 0:
            self.List_RLN = [RLN_neo.RLNcell()]
        for i in range(np.sum(self.C.NB_RLN)):
            self.List_RLN.append(RLN_neo.RLNcell())

        # initialize synaptic ODE
        #########################################################
        count = 0
        self.PreSynaptic_Cell_AMPA=self.Conx['PreSynaptic_Cell_AMPA']
        self.PreSynaptic_Cell_GABA=self.Conx['PreSynaptic_Cell_GABA']
        self.ExternalPreSynaptic_Cell_AMPA_DPYR=self.Conx['ExternalPreSynaptic_Cell_AMPA_DPYR']
        self.ExternalPreSynaptic_Cell_AMPA_Th=self.Conx['ExternalPreSynaptic_Cell_AMPA_Th']

        for i in range(np.sum(self.C.NB_RLN[0])):
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
            for i in range(np.sum(self.C.NB_PYR[0:l]), self.C.NB_PYR[l] + np.sum(self.C.NB_PYR[0:l])):
                self.List_PYR[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_PYR[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_PYR[i].init_vector()
                count += 1
            for i in range(np.sum(self.C.NB_PV[0:l]), self.C.NB_PV[l] + np.sum(self.C.NB_PV[0:l])):
                self.List_PV[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_PV[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_PV[i].init_vector()
                count += 1
            for i in range(np.sum(self.C.NB_SST[0:l]), self.C.NB_SST[l] + np.sum(self.C.NB_SST[0:l])):
                self.List_SST[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_SST[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_SST[i].init_vector()
                count += 1
            for i in range(np.sum(self.C.NB_VIP[0:l]), self.C.NB_VIP[l] + np.sum(self.C.NB_VIP[0:l])):
                self.List_VIP[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_DPYR[count]) + len(
                    self.ExternalPreSynaptic_Cell_AMPA_Th[count])
                self.List_VIP[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
                self.List_VIP[i].init_vector()
                count += 1

            for i in range(np.sum(self.C.NB_RLN[0:l]), self.C.NB_RLN[l] + np.sum(self.C.NB_RLN[0:l])):
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
            for i in range(np.sum(self.C.NB_PV[0:l]), self.C.NB_PV[l] + np.sum(self.C.NB_PV[0:l])):
                Neurone_param.append({s: getattr(self.List_PV[i], s) for s in self.get_PV_Variables()})
            for i in range(np.sum(self.C.NB_SST[0:l]), self.C.NB_SST[l] + np.sum(self.C.NB_SST[0:l])):
                Neurone_param.append({s: getattr(self.List_SST[i], s) for s in self.get_SST_Variables()})
            for i in range(np.sum(self.C.NB_VIP[0:l]), self.C.NB_VIP[l] + np.sum(self.C.NB_VIP[0:l])):
                Neurone_param.append({s: getattr(self.List_VIP[i], s) for s in self.get_VIP_Variables()})
            for i in range(np.sum(self.C.NB_RLN[0:l]), self.C.NB_RLN[l] + np.sum(self.C.NB_RLN[0:l])):
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
            for i in range(np.sum(self.C.NB_PV[0:l]), self.C.NB_PV[l] + np.sum(self.C.NB_PV[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_PV[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.C.NB_SST[0:l]), self.C.NB_SST[l] + np.sum(self.C.NB_SST[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_SST[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.C.NB_VIP[0:l]), self.C.NB_VIP[l] + np.sum(self.C.NB_VIP[0:l])):
                for key in self.List_Neurone_param[l][ind]:
                    setattr(self.List_VIP[i], key, self.List_Neurone_param[l][ind][key])
                ind += 1
            for i in range(np.sum(self.C.NB_RLN[0:l]), self.C.NB_RLN[l] + np.sum(self.C.NB_RLN[0:l])):
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

        for i in range(np.sum(self.C.NB_PYR)):
            self.List_PYR[i].init_I_syn()
            self.List_PYR[i].init_vector()
            self.List_PYR[i].setParameters()

        for i in range(np.sum(self.C.NB_PV)):
            self.List_PV[i].init_I_syn()
            self.List_PV[i].init_vector()
            self.List_PV[i].setParameters()

        for i in range(np.sum(self.C.NB_SST)):
            self.List_SST[i].init_I_syn()
            self.List_SST[i].init_vector()
            self.List_SST[i].setParameters()

        for i in range(np.sum(self.C.NB_VIP)):
            self.List_VIP[i].init_I_syn()
            self.List_VIP[i].init_vector()
            self.List_VIP[i].setParameters()

        for i in range(np.sum(self.C.NB_RLN)):
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

        self.pyrPPSE = np.zeros((5,np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI = np.zeros((5,np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_s = np.zeros((np.sum(self.NB_PYR), len(t)))
        self.pyrPPSI_a = np.zeros((np.sum(self.NB_PYR), len(t)))

        self.PV_Vs = np.zeros((np.sum(self.C.NB_PV), len(t)))
        self.SST_Vs = np.zeros((np.sum(self.C.NB_SST), len(t)))
        self.VIP_Vs = np.zeros((np.sum(self.C.NB_VIP), len(t)))
        self.RLN_Vs = np.zeros((np.sum(self.C.NB_RLN), len(t)))
        self.DPYR_Vs = np.zeros((np.sum(self.NB_DPYR), len(t)))
        self.Th_Vs = np.zeros((np.sum(self.NB_Th), len(t)))



        nbcells = np.sum(self.Layer_nbCells)
        layerS = np.zeros(nbcells,dtype=int)
        typeS = np.zeros(nbcells,dtype=int)
        indexS = np.zeros(nbcells,dtype=int)
        for ind in range(nbcells):
            layerS[ind], typeS[ind], indexS[ind] = self.All2layer(ind, self.Layer_nbCells, self.NB_PYR, self.C.NB_PV, self.C.NB_SST, self.C.NB_VIP,
                                               self.C.NB_RLN, self.List_celltypes)

        if 1:
            t0 = time.time()
            self.t, self.pyrVs, self.pyrVd, self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs,self.pyrPPSE,self.pyrPPSI,self.pyrPPSI_s,self.pyrPPSI_a = ComputeSim.Model_compute(np.int(self.nbEch),
                                                                                             np.float(self.dt),
                                                                                             np.float(self.tps_start),
                                                                                             self.Layer_nbCells,
                                                                                             self.NB_PYR,
                                                                                             self.C.NB_PV,
                                                                                             self.C.NB_SST,
                                                                                             self.C.NB_VIP,
                                                                                             self.C.NB_RLN,
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
                                                                                             np.int(self.seed),
                                                                                             self.pyrPPSE,
                                                                                             self.pyrPPSI,
                                                                                             self.pyrPPSI_s,
                                                                                             self.pyrPPSI_a)
            print(time.time() - t0)
            #run model


        return (self.t, self.pyrVs, self.pyrVd,self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs,self.RLN_Vs, self.DPYR_Vs, self.Th_Vs,self.pyrPPSE,self.pyrPPSI,self.pyrPPSI_s,self.pyrPPSI_a)

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

