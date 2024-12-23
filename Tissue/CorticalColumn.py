__author__ = 'Mariam, Maxime'

import numpy as np
from scipy.spatial import distance
import time
from Tissue import SST_neo
from Tissue import PV_neo
from Tissue import VIP_neo
from Tissue import RLN_neo
from Tissue import PC_neo3
from Tissue import Column_morphology
from Computation import ComputeSim
from PyQt5.QtCore import *





class SenderObject(QObject):
    something_happened = pyqtSignal(float)

class SenderObjectInt(QObject):
    something_happened = pyqtSignal(int)

class CorticalColumn:
    """ Class corresponding to the modeled cortical column
    """
    updateTime = SenderObject()
    updateCell = SenderObjectInt()
    def __init__(self, type = 1):
        #initialize cortical column parameters
        self.type = type
        self.C=Column_morphology.Column(type) #0: 'human, 1:Rat, 0:mice
        self.D = self.C.D  # neocortical column diameter in micrometers
        self.Layer_d = self.C.L_th  # layers' thicknesses (L1-L2/3-L4-L5-L6)
        self.L=sum(self.Layer_d) #column length
        self.Curvature = self.C.Curvature
        self.Layertop_pos=np.cumsum(self.Layer_d[::-1])
        self.Layertop_pos_mean=self.Layertop_pos-self.Layer_d[::-1]/2
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
            [[1, 1, 1, 1, 0],  # PC -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 0, 0],  # PV -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 1, 1],  # SST -> PC,PV,SST,VIP ,RLN
             [0, 0, 1, 0, 0],   #VIP --> PC,PV,SST,VIP ,RLN
             [1, 1, 1, 1, 1]    #RLN --> PC,PV,SST,VIP ,RLN
             ], dtype=float)


        self.Afferences = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,20,2,0,2,3,12,3,0,2,1,0,2,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,22,28,65,27,0,9,2,0,10,0,20,24,6,22,0,0,0,0,0,0],
        [0,0,0,0,0,7,11,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,15,2,3,0,9,15,1,1,0,0,0,1,1,0,2,0,1,1,0,0,0],
        [0,0,0,0,0,0,0,13,0,0,0,0,3,0,0,0,0,3,0,0,0,0,1,0,0],
        [0,0,0,0,10,5,4,5,5,26,4,0,3,2,0,3,0,2,0,0,0,0,0,0,0],
        [0,0,0,0,0,36,40,13,15,0,19,36,29,20,0,20,19,25,14,0,6,0,0,11,0],
        [0,0,0,0,0,1,2,0,0,0,7,5,0,0,0,1,2,0,0,0,0,0,0,0,0],
        [0,0,0,0,10,1,1,0,0,12,4,4,0,5,0,0,0,0,7,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,3,0,0,0,0,12,0,0,0,0,3,0,0,0,0,3,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,4,0,0,3,0,5,0,0,5,0,20,19,35,14,0,47,30,31,33,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,7,7,0,0,0,0,2,0,0,0],
        [0,0,0,0,5,2,3,0,3,8,1,2,0,2,0,2,2,0,4,0,3,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,13,0,0,0,0,3,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,28,29,35,15,0,5,5,3,4,0,23,48,49,22,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,12,9,0,0,0],
        [0,0,0,0,0,2,3,0,3,2,1,2,0,2,0,2,2,0,4,0,3,3,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,9,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,15,6,2,2,14,10,11,10,7,20,0,11,7,3,18,0,4,5,3,24,0],
        [0,0,0,0,25,10,3,4,18,15,6,6,6,18,0,5,2,2,11,0,1,1,1,10,0]])

        self.update_connections(self.Afferences)
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

        self.Conx={}

    def updateTissue(self, D, L, Curvature, Layer_d):
        """ New tissue dimension
        """
        self.D = D
        self.L = L
        self.Curvature = Curvature
        self.Layer_d = Layer_d
        self.C.update_morphology(L=self.L,
                          D=self.D,
                          Curvature=self.Curvature,
                          L_th=self.Layer_d)

    def update_cellNumber(self,Layer_nbCells,
                                  PYRpercent,
                                  PVpercent ,
                                  SSTpercent ,
                                  VIPpercent ,
                                  RLNpercent ,

                              PCsubtypes_Per=None,
                              NB_PYR=None,
                              NB_PV_BC=None,
                              NB_PV_ChC=None,
                              NB_IN=None,
                              NB_PV=None,
                              NB_SST=None,
                              NB_VIP=None,
                              NB_RLN=None
                              ):

        """ New neuron density in the tissue
        """
        self.C.update_morphology(Layer_nbCells=Layer_nbCells,
                                  PYRpercent=PYRpercent,
                                  PVpercent=PVpercent ,
                                  SSTpercent=SSTpercent ,
                                  VIPpercent=VIPpercent ,
                                  RLNpercent=RLNpercent,
                                 NB_PYR=NB_PYR,
                                 NB_PV_BC=NB_PV_BC,
                                 NB_PV_ChC=NB_PV_ChC,
                                 NB_IN=NB_IN,
                                 NB_PV=NB_PV,
                                 NB_SST=NB_SST,
                                 NB_VIP=NB_VIP,
                                 NB_RLN=NB_RLN
                                 )
        self.Layer_nbCells=self.C.Layer_nbCells #nb of cells in each layer
        self.Layer_nbcells_pertype=self.C.Layer_nbCells_pertype # nb of each type cell in the layers PC,PV,SST,VIP,RLN
        self.Nbcells = int(np.sum(self.Layer_nbCells)) #total nb of cells

        print('Total cell number:',self.Nbcells)
        self.NB_PYR = self.Layer_nbcells_pertype[0]

        #subtypes of PYR cells
        if PCsubtypes_Per is None:
            self.PCsubtypes_Per=self.C.PCsubtypes_Per #TPC,UPC,IPC,BPC,SSC
        self.List_celltypes = self.C.List_celltypes
        self.List_cellsubtypes = self.C.List_cellsubtypes
        #####External afferences
        self.NB_DPYR=int(np.sum(self.NB_PYR)*0.07)
        self.NB_Th=int(np.sum(self.NB_PYR)*0.07)
        self.update_connections()

    def update_inputNB(self, division = 60):
        """ ratio for the density
        """
        self.inputNB = int(np.sum(self.Layer_nbCells) / division)

    def update_connections(self,matrice=None, fixed = False, division = 60):
        """ compute the number of connection according to the connectivity matrix
        """
        self.update_inputNB(division)
        if not matrice is None:
            self.Afferences = matrice
        if fixed:
            self.inputpercent = self.Afferences.astype(int)
        else:
            self.inputpercent=np.ceil(self.Afferences*self.inputNB/100)
            self.inputpercent=self.inputpercent.astype(int)
        self.Allconnexions=[]

    def createFlatCEllpos(self):
        """ Acces to the neurons with a vector
        """
        self.Cellposflat=np.vstack((self.Cellpos[0], self.Cellpos[1], self.Cellpos[2], self.Cellpos[3], self.Cellpos[4]))

    def Generate_Stims(self, I_inj=60, tau=4, stimDur=3, nbstim=5, varstim=12, freq=1.5,StimStart = 0,S_StimStop=1000,type='One shot', periode=200,nbEch=1000):
        """ Compute distant cortex stimulation
        """
        print('computing stim signals')
        if not self.seed == 0:
            np.random.seed(self.seed)
        if type == 'One shot':
            # nb = int(1e-3 * self.T*freq) #frequency of fast ripples
            nb = 1
            nb_Stim_Signals = self.C.NB_DPYR
            nbOfSamplesStim = int(stimDur / self.dt)

            Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))
            varianceStim = varstim / self.dt
            t = np.arange(self.nbEch) * self.dt
            y = np.zeros(t.shape)
            for tt, tp in enumerate(t):
                if tt <= nbOfSamplesStim:
                    y[tt] = (1. - np.exp(-tp / tau)) * I_inj
                if tt > nbOfSamplesStim:
                    y[tt] = (np.exp(-(tp - nbOfSamplesStim * self.dt) / tau)) * y[nbOfSamplesStim - 1]

            if StimStart == 0:
                N0 = int(self.nbEch / 2)
            else:
                N0 = int(StimStart / self.dt)
            S_StimStop =int(S_StimStop / self.dt)
            for St in range(nb_Stim_Signals):
                yt = np.zeros(self.nbEch)
                for j in range(nb):
                    y2 = np.zeros(self.nbEch)
                    for i in range(nbstim):

                        inst = int(np.round((np.random.normal(0, varianceStim))))
                        t0 = N0 + inst

                        if t0 == 0:
                            t0 = 0
                        elif t0 < 0:
                            t0 = N0 + int(varianceStim * inst / abs(2 * inst))
                        elif t0 > self.nbEch - 1 or t0 > S_StimStop:
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

        elif type == 'Periodic':

            nb_Stim_Signals = self.C.NB_DPYR
            nbOfSamplesStim = int(stimDur / self.dt)
            npos = np.arange(int(StimStart / self.dt), nbEch, int(periode / self.dt), dtype=float)
            npos += (np.random.uniform(size=len(npos)) - 0.5) * (int(periode / self.dt) / 2)
            Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))
            varianceStim = varstim / self.dt
            t = np.arange(self.nbEch) * self.dt
            y = np.zeros(t.shape)
            for tt, tp in enumerate(t):
                if tt <= nbOfSamplesStim:
                    y[tt] = (1. - np.exp(-tp / tau)) * I_inj
                if tt > nbOfSamplesStim:
                    y[tt] = (np.exp(-(tp - nbOfSamplesStim * self.dt) / tau)) * y[nbOfSamplesStim - 1]



            for St in range(nb_Stim_Signals):
                yt = np.zeros(self.nbEch)
                for j in range(len(npos)):
                    y2 = np.zeros(self.nbEch)
                    for i in range(nbstim):
                        N0=int(npos[j])
                        inst = int(np.round((np.random.normal(0, varianceStim))))
                        t0 = N0 + inst
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

        self.Stim_Signals = Stim_Signals_out
        print('end computing stim signals')
        return self.Stim_Signals

    def set_seed(self,seed):
        self.seed=seed

    def Generate_input(self, I_inj=25, tau=4, stimDur=3, nbstim=5, deltamin=14, delta=18):
        """ Compute thalamus input stimulation
        """
        print('end computing input ')
        if not self.seed == 0:
            np.random.seed(self.seed)

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
        """ Instanciate neuron models
        """
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
        if self.C.NB_DPYR == 0:
            self.List_DPYR = [PC_neo3.pyrCellneo(1)]
        for i in range(int(self.C.NB_DPYR * layer_ratio)):  # define PYR cells sublayer types
            self.List_DPYR.append(PC_neo3.pyrCellneo(1))  # from layer 2
        for i in range(int(self.C.NB_DPYR * layer_ratio), self.C.NB_DPYR + 1):  # from layer 5
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

        """ Update neurons parameters values
        """
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



     ##################################################
    def UpdateModel(self):
        """ Reset all model parameters
        """
        for i in range(self.NB_Th):
            self.List_Th[i].init_I_syn()
            self.List_Th[i].init_vector()
            self.List_Th[i].setParameters()

        for i in range(self.C.NB_DPYR):
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




    def runSim(self):
        """ Apply the simualation
        """
        print('computing signals...')
        self.t = np.arange(self.nbEch) * self.dt

        self.DpyrVs = np.zeros((np.sum(self.C.NB_DPYR), len(self.t)))
        self.DpyrVd = np.zeros((np.sum(self.C.NB_DPYR), len(self.t)))
        self.ThpyrVs = np.zeros((np.sum(self.C.NB_Th), len(self.t)))
        self.ThpyrVd = np.zeros((np.sum(self.C.NB_Th), len(self.t)))
        self.pyrVd = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrVs = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrVa = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))

        self.pyrPPSE_Dpyr=np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrPPSE_Th=np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrPPSE = np.zeros((5,np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrPPSI = np.zeros((5,np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrPPSI_s = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrPPSI_a = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrI_S = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrI_d = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))
        self.pyrI_A = np.zeros((np.sum(self.C.NB_PYR), len(self.t)))

        self.PV_Vs = np.zeros((np.sum(self.C.NB_PV), len(self.t)))
        self.SST_Vs = np.zeros((np.sum(self.C.NB_SST), len(self.t)))
        self.VIP_Vs = np.zeros((np.sum(self.C.NB_VIP), len(self.t)))
        self.RLN_Vs = np.zeros((np.sum(self.C.NB_RLN), len(self.t)))
        self.DPYR_Vs = np.zeros((np.sum(self.C.NB_DPYR), len(self.t)))
        self.Th_Vs = np.zeros((np.sum(self.C.NB_Th), len(self.t)))



        t0 = time.time()
        self.t, self.pyrVs, self.pyrVd, self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs,self.pyrPPSE,self.pyrPPSI,self.pyrPPSI_s,self.pyrPPSI_a,self.pyrPPSE_Dpyr,self.pyrPPSE_Th,self.pyrI_S,self.pyrI_d, self.pyrI_A = ComputeSim.Model_compute(np.int32(self.nbEch),
                                                                                                                                                                                                                                                                                          np.float64(self.dt),
                                                                                                                                                                                                                                                                                          np.float64(self.tps_start),
                                                                                                                                                                                                                                                                                          self.Layer_nbCells,
                                                                                                                                                                                                                                                                                          self.C.NB_PYR,
                                                                                                                                                                                                                                                                                          self.C.NB_PV,
                                                                                                                                                                                                                                                                                          self.C.NB_SST,
                                                                                                                                                                                                                                                                                          self.C.NB_VIP,
                                                                                                                                                                                                                                                                                          self.C.NB_RLN,
                                                                                                                                                                                                                                                                                          self.C.NB_DPYR,
                                                                                                                                                                                                                                                                                          self.C.NB_Th,
                                                                                                                                                                                                                                                                                          np.int32(self.inputNB),
                                                                                                                                                                                                                                                                                          self.List_PYR,
                                                                                                                                                                                                                                                                                          self.List_PV,
                                                                                                                                                                                                                                                                                          self.List_SST,
                                                                                                                                                                                                                                                                                          self.List_VIP,
                                                                                                                                                                                                                                                                                          self.List_RLN,
                                                                                                                                                                                                                                                                                          self.List_DPYR,
                                                                                                                                                                                                                                                                                          self.List_Th,
                                                                                                                                                                                                                                                                                          self.Stim_Signals,
                                                                                                                                                                                                                                                                                          self.Stim_InputSignals,
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
                                                                                                                                                                                                                                                                                          self.C.layerS,
                                                                                                                                                                                                                                                                                          self.C.typeS,
                                                                                                                                                                                                                                                                                          self.C.indexS,
                                                                                                                                                                                                                                                                                          self.t,
                                                                                                                                                                                                                                                                                          np.int32(self.seed),
                                                                                                                                                                                                                                                                                          self.pyrPPSE,
                                                                                                                                                                                                                                                                                          self.pyrPPSI,
                                                                                                                                                                                                                                                                                          self.pyrPPSI_s,
                                                                                                                                                                                                                                                                                          self.pyrPPSI_a,
                                                                                                                                                                                                                                                                                          self.pyrPPSE_Dpyr,
                                                                                                                                                                                                                                                                                          self.pyrPPSE_Th,
                                                                                                                                                                                                                                                                                          self.pyrI_S,
                                                                                                                                                                                                                                                                                          self.pyrI_d,
                                                                                                                                                                                                                                                                                          self.pyrI_A)
        print(time.time() - t0)


        return (self.t, self.pyrVs, self.pyrVd,self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs,self.RLN_Vs, self.DPYR_Vs, self.Th_Vs,self.pyrPPSE,self.pyrPPSI,self.pyrPPSI_s,self.pyrPPSI_a,self.pyrPPSE_Dpyr,self.pyrPPSE_Th, self.pyrI_S,self.pyrI_d, self.pyrI_A)


    def get_PYR_Variables(self):
        """ Get Pyr model parameter name for the GUI
        """
        return PC_neo3.get_Variable_Names()

    def get_PV_Variables(self):
        """ Get Pv model parameter name for the GUI
        """
        return PV_neo.get_Variable_Names()

    def get_SST_Variables(self):
        """ Get SST model parameter name for the GUI
        """
        return SST_neo.get_Variable_Names()

    def get_VIP_Variables(self):
        """ Get VIP model parameter name for the GUI
        """
        return VIP_neo.get_Variable_Names()

    def get_RLN_Variables(self):
        """ Get RLN model parameter name for the GUI
        """
        return RLN_neo.get_Variable_Names()

