import os
import sys
import math
import numpy as np
from scipy.stats import levy_stable
import scipy
import scipy.io as sio
from scipy.spatial import distance
from scipy import sparse
from scipy import signal
import pickle
import time
import matplotlib

import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import inspect
import copy
import subprocess
import struct
import random
import csv
#import pyedflib
import datetime

import numba as nb
from numba import guvectorize, float64, int64
from numba.experimental import jitclass
from numba import jit, njit, types, typeof

import timeit
import PC_neo3 as PC_neo
import PyrCellCA1
from numba import types, typeof
from numba.experimental import jitclass

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

plt.rcParams.update({'font.size': 8})
from pyqtgraph import PlotWidget
import pyqtgraph as pg

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class LineEdit(QLineEdit):
    KEY = Qt.Key_Return
    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegExpValidator(QRegExp("[+-]?\\d*[\\.]?\\d+"))
        QREV.setLocale(QLocale(QLocale.English))
        self.setValidator(QREV)


class LineEdit_Int(QLineEdit):
    KEY = Qt.Key_Return
    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegExpValidator(QRegExp("[+-]?\\d+"))
        QREV.setLocale(QLocale(QLocale.English))
        self.setValidator(QREV)

class Ui_Main(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_Main, self).__init__()
        self.parent = parent

        self.width150 = 50

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX = QHBoxLayout()

        #param layout
        self.Param_L1 = QVBoxLayout()
        self.Param_L2 = QVBoxLayout()
        #compartment number
        self.Compartment_GB = QGroupBox("Choose the number of compartments")
        layout = QVBoxLayout()
        self.Compartment_GB.setLayout(layout)
        self.Compartment_BG = QButtonGroup()
        self.Compartiment2_RB = QRadioButton('2 compartments')
        self.Compartiment2_RB.setChecked(True)
        self.Compartiment3_RB = QRadioButton('3 compartments')
        self.Compartment_BG.addButton(self.Compartiment2_RB)
        self.Compartment_BG.addButton(self.Compartiment3_RB)
        self.Compartment_BG.setExclusive(True)
        layout.addWidget(self.Compartiment2_RB)
        layout.addWidget(self.Compartiment3_RB)
        # self.checkBox.stateChanged.connect(self.state_changed)
        # self.lfpplot_plottype.buttonClicked['QAbstractButton *'].connect(self.lfpplot_plottype_clicked)

        #Reversal potentials
        self.Reversal_Potential_GB = QGroupBox("Reversal potentials")
        widget = QWidget()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignLeft)
        widget.setLayout(grid)
        self.E_Na = LineEdit('60')
        self.E_K = LineEdit('-80')
        self.E_Ca = LineEdit('140')
        self.E_h = LineEdit('-10')
        self.E_leak = LineEdit('-70')
        name = ['E_Na', 'E_K', 'E_Ca', 'E_h', 'E_leak' ]
        for i, v in enumerate([self.E_Na,self.E_K,self.E_Ca,self.E_h,self.E_leak]):
            v.setFixedWidth(self.width150 )
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.setColumnStretch(2,10)
        self.Reversal_Potential_GB.setLayout(grid)


        #Conductance
        self.Conductances_GB = QGroupBox("Conductances")
        self.Conductances_GB_l = QHBoxLayout()
        self.Conductances_GB.setLayout(self.Conductances_GB_l)
        self.tabs = QTabWidget()
        self.tab_Soma = QWidget()
        self.tab_Dendrite = QWidget()
        self.tab_AIS = QWidget()
        self.tabs.addTab(self.tab_Soma,"Soma")
        self.tabs.addTab(self.tab_Dendrite,"Dendrite")
        self.tabs.addTab(self.tab_AIS,"IS")
        self.Conductances_GB_l.addWidget(self.tabs)


        #Soma
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        self.tab_Soma.setLayout(grid)
        self.S_g_Na = LineEdit('70')
        self.S_g_KDR = LineEdit('6')
        self.S_g_CaL = LineEdit('0.5')
        self.S_g_AHP = LineEdit('0.1')
        self.S_g_BK = LineEdit('2')
        self.S_g_m = LineEdit('3.1')
        self.S_g_KNa = LineEdit('0')
        self.S_g_Leak_s = LineEdit('0.18')

        name = ['g_Na','g_KDR','g_CaL','g_AHP','g_BK','g_m','g_KNa','g_Leak']
        for i, v in enumerate([self.S_g_Na,self.S_g_KDR,self.S_g_CaL,self.S_g_AHP,self.S_g_BK,self.S_g_m,self.S_g_KNa,self.S_g_Leak_s]):
            v.setFixedWidth(self.width150)
            # name = f'{self.E_Na=}'.split('=')[0].replace('self.', '')
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.setColumnStretch(2,10)

        #Dendrite
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        self.tab_Dendrite.setLayout(grid)
        self.D_g_Na = LineEdit('14')
        self.D_g_KDR = LineEdit('2')
        self.D_g_CaT = LineEdit('1')
        self.D_g_CaR = LineEdit('3')
        self.D_g_AHP = LineEdit('10')
        self.D_g_BK = LineEdit('1')
        self.D_g_m = LineEdit('0.2')
        self.D_g_h = LineEdit('0.6')
        self.D_g_KA = LineEdit('55')
        self.D_g_Leak_d = LineEdit('0.18')
        name = ['g_Na','g_KDR','g_CaT','g_CaR','g_AHP','g_BK','g_m','g_h','g_KA','g_Leak']
        for i, v in enumerate([self.D_g_Na,self.D_g_KDR,self.D_g_CaT,self.D_g_CaR,self.D_g_AHP,self.D_g_BK,self.D_g_m,self.D_g_h,self.D_g_KA,self.D_g_Leak_d]):
            v.setFixedWidth(self.width150 )
            # name = f'{self.E_Na=}'.split('=')[0].replace('self.', '')
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.setColumnStretch(2,10)

        #AIS
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        self.tab_AIS.setLayout(grid)
        self.A_g_Na = LineEdit('200')
        self.A_g_KDR = LineEdit('200')
        self.A_g_m = LineEdit('0')
        self.A_g_Leak_a = LineEdit('0.18')
        name = ['g_Na','g_KDR','g_m','g_Leak']
        for i, v in enumerate([self.A_g_Na,self.A_g_KDR,self.A_g_m,self.A_g_Leak_a]):
            v.setFixedWidth(self.width150 )
            # name = f'{self.E_Na=}'.split('=')[0].replace('self.', '')
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.setColumnStretch(2,10)


        #General
        self.General_GB = QGroupBox("General")
        widget = QWidget()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        widget.setLayout(grid)
        self.gc_SD = LineEdit('1')
        self.gc_SA = LineEdit('2')
        self.p_SD = LineEdit('0.15')
        self.p_SA= LineEdit('0.95')
        self.Cm_s = LineEdit('1')
        self.Cm_d = LineEdit('2')
        self.Cm_a = LineEdit('1')
        self.Noise = LineEdit('0')
        name = ['gc_SD','gc_SA','p_SD','p_SA','Cm_s','Cm_d','Cm_a','Noise']
        for i, v in enumerate([self.gc_SD,self.gc_SA,self.p_SD,self.p_SA,self.Cm_s,self.Cm_d,self.Cm_a,self.Noise]):
            v.setFixedWidth(self.width150 )
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.setColumnStretch(2, 10)
        self.General_GB.setLayout(grid)

        #Stimulation parameters
        self.Stimulation_GB = QGroupBox("Stimulation Parameters")
        widget = QWidget()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        widget.setLayout(grid)
        self.Fs = LineEdit('40')
        self.T = LineEdit('1000')
        self.stim_T = LineEdit('400')
        self.I_soma= LineEdit('20')
        self.CB_inj=QComboBox()
        self.CB_PYR_l=QComboBox()
        self.CB_inj.addItem('uA/cm2')
        self.CB_inj.addItem('pA')
        self.CB_PYR_l.addItem('2/3')
        self.CB_PYR_l.addItem('4')
        self.CB_PYR_l.addItem('5')
        self.CB_PYR_l.addItem('6')




        name = ['Fs (Hz)', 'T (ms)', 'Stim time (ms)', 'I_Soma']
        for i, v in enumerate([self.Fs,self.T,self.stim_T,self.I_soma]):
            v.setFixedWidth(self.width150 )
            grid.addWidget(QLabel(name[i]), i, 0)
            grid.addWidget(v, i, 1)
            grid.addItem(QSpacerItem(1, 0), i, 2)
        grid.addWidget(self.CB_inj)
        grid.addWidget(self.CB_PYR_l,4,1)

        self.IstimSimu_s = LineEdit('0')
        self.IstimSimu_e = LineEdit('10')
        self.IstimSimu_step = LineEdit('1')
        name = ['start', 'end', 'step']
        for i, v in enumerate([self.IstimSimu_s, self.IstimSimu_e, self.IstimSimu_step]):
            v.setFixedWidth(self.width150)
            grid.addWidget(QLabel(name[i]), i+5, 0)
            grid.addWidget(v, i+5, 1)
            # grid.addItem(QSpacerItem(1, 0), i+1, 2)

        self.IstimSimu_RUN = QPushButton('RUN multi stim')
        grid.addWidget(self.IstimSimu_RUN, 8, 0,1,2)
        grid.setColumnStretch(8, 10)
        self.IstimSimu_RUN.clicked.connect(self.clickmemultiIstim)
        self.Stimulation_GB.setLayout(grid)

        # self.tabs.resize(300, 200)
        self.Param_L1.addWidget(self.Compartment_GB)
        self.Param_L1.addWidget(self.Reversal_Potential_GB)
        self.Param_L1.addWidget(self.Conductances_GB)

        self.Param_L2.addWidget(self.General_GB)
        self.Param_L2.addWidget(self.Stimulation_GB)

        # Actions
        self.Action_L = QVBoxLayout()
        self.Action_L.setAlignment(Qt.AlignTop)
        self.Load = QPushButton('Load')
        self.RUN = QPushButton('RUN')
        self.ClearFigure = QPushButton('Clear Figure')

        self.Load.clicked.connect(self.loadparam)
        self.RUN.clicked.connect(self.clickme)
        self.ClearFigure.clicked.connect(self.cleargraph)

        self.CB_Dend = QCheckBox('Display V_d')
        self.CB_Dend.setStyleSheet('QCheckBox {background-color: green;}')#{background-color(163, 37, 11, 255)}')
        self.CB_ais = QCheckBox('Display V_AIS')
        self.CB_ais.setStyleSheet('QCheckBox {background-color: red;}')
        self.CB_Dend.stateChanged.connect(self.CB_Dend_action)
        self.CB_ais.stateChanged.connect(self.CB_ais_action)

        # Stimulation parameters
        self.RangeSimu_GB = QGroupBox("RangeSimu")
        widget = QWidget()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        widget.setLayout(grid)
        self.RangeSimu_CB = QComboBox()
        grid.addWidget(self.RangeSimu_CB, 0, 0,1,2)
        self.RangeSimu2_CB = QComboBox()
        grid.addWidget(self.RangeSimu2_CB, 0, 2,1,2)

        # grid.addItem(QSpacerItem(1, 0), 0, 2)
        self.RangeSimu_s = LineEdit('0')
        self.RangeSimu2_s = LineEdit('0')
        self.RangeSimu_e = LineEdit('10')
        self.RangeSimu2_e = LineEdit('10')
        self.RangeSimu_step = LineEdit('1')
        self.RangeSimu2_step = LineEdit('1')
        name = ['start', 'end', 'step']
        for i, v in enumerate([self.RangeSimu_s, self.RangeSimu_e, self.RangeSimu_step]):
            v.setFixedWidth(self.width150)
            grid.addWidget(QLabel(name[i]), i+1, 0)
            grid.addWidget(v, i+1, 1)
            # grid.addItem(QSpacerItem(1, 0), i+1, 2)
        for i, v in enumerate([self.RangeSimu2_s, self.RangeSimu2_e, self.RangeSimu2_step]):
            v.setFixedWidth(self.width150)
            grid.addWidget(QLabel(name[i]), i+1, 2)
            grid.addWidget(v, i+1, 3)
            grid.addItem(QSpacerItem(1, 0), i+1, 4)
        self.RangeSimu_RUN = QPushButton('RUN multi')
        self.RangeSimugrid_RUN = QPushButton('RUN grid')
        grid.addWidget(self.RangeSimu_RUN, 4, 0,1,2)
        grid.addWidget(self.RangeSimugrid_RUN, 4, 2,1,2)
        grid.setColumnStretch(4, 10)
        self.RangeSimu_GB.setLayout(grid)
        self.RangeSimu_RUN.clicked.connect(self.clickmemulti)
        self.RangeSimugrid_RUN.clicked.connect(self.clickmegrid)

        #Detection choice
        self.Detection_GB = QGroupBox("Detection choice")
        widget = QWidget()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        widget.setLayout(grid)
        self.Detection_BG = QButtonGroup()
        self.Detection_BG.setExclusive(True)
        self.NB_PA_RB = QRadioButton('Number of PA')
        self.NB_PA_RB.setChecked(True)
        self.Detection_BG.addButton(self.NB_PA_RB)
        self.D_Threshold = LineEdit('0')
        self.Fs_PA_RB = QRadioButton('Frequence of PA')
        self.Detection_BG.addButton(self.Fs_PA_RB)
        self.Do_Detection_PB = QPushButton('Apply')
        self.Do_Detection_PB.clicked.connect(self.Do_Detection)

        grid.addWidget(self.NB_PA_RB, 0, 0,1,2)
        grid.addWidget(QLabel('Threshold'), 1, 0)
        grid.addWidget(self.D_Threshold, 1, 0)
        grid.addWidget(self.Fs_PA_RB, 2, 0,1,2)
        grid.addWidget(self.Do_Detection_PB, 3, 0,1,2)

        # grid.addItem(QSpacerItem(1, 0), i, 2)

        grid.setColumnStretch(4, 10)
        self.Detection_GB.setLayout(grid)



        # self.Action_L.addWidget(self.Load)
        self.Action_L.addWidget(QLabel(''))
        self.Action_L.addWidget(self.RUN)
        self.Action_L.addWidget(QLabel(''))
        self.Action_L.addWidget(self.ClearFigure)
        self.Action_L.addWidget(QLabel(''))
        self.Action_L.addWidget(self.CB_Dend)
        self.Action_L.addWidget(self.CB_ais)
        self.Action_L.addWidget(QLabel(''))
        self.Action_L.addWidget(self.RangeSimu_GB)
        self.Action_L.addWidget(self.Detection_GB)


        ###graph
        # self.graphwidget = PlotWidget()
        # # self.graphwidget.setGeometry(.QRect(500, 150, 650, 500))
        # self.graphwidget.setObjectName("graphwidget")
        # self.graphwidget.setBackground('w')
        # styles = {'color': 'k', 'font-size': '20px'}
        # self.graphwidget.setLabel('left', 'Vs (mV)', **styles)
        # self.graphwidget.setLabel('bottom', 'Time (ms)', **styles)

        self.graphwidget =  Viewer()
        self.graphwidget.setMinimumWidth(600)
        self.graphwidget.setMinimumHeight(600)

        param1 =  np.arange(10, 20, 1)
        param2 =      np.arange(1, 20, 0.5)
        self.xv, self.yv = np.meshgrid(param2, param1)
        self.listcolor_lfp_states_Sig = [QColor(Qt.darkGreen).name(),
                                         QColor(Qt.darkMagenta).name(),
                                         QColor(0, 150, 210).name(),
                                         QColor(Qt.red).name(),
                                         QColor(Qt.blue).name(),
                                         QColor(Qt.darkGray).name(),
                                         QColor(Qt.black).name()
                                         ]
        self.mascenegrid = gridViewer(self)

        self.graphsplitter = QSplitter(Qt.Vertical)
        self.graphsplitter.addWidget(self.graphwidget)
        self.graphsplitter.addWidget(self.mascenegrid)
        self.graphsplitter.addWidget(QLabel(''))
        ###
        Param_L1_w = QWidget()
        Param_L1_w.setLayout(self.Param_L1)
        # Param_L1_w.setFixedWidth(200)
        Param_L2_w = QWidget()
        Param_L2_w.setLayout(self.Param_L2)
        # Param_L2_w.setFixedWidth(150)
        Action_L_w = QWidget()
        Action_L_w.setLayout(self.Action_L)
        # Param_L2_w.setFixedWidth(150)

        self.leftpanel_l = QHBoxLayout()
        self.leftpanel_l.addWidget(Param_L1_w)
        self.leftpanel_l.addWidget(Param_L2_w)
        self.leftpanel_l.addWidget(Action_L_w)
        leftpanel_w = QWidget()
        leftpanel_w.setLayout(self.leftpanel_l)
        leftpanel_w.setFixedWidth(600)

        self.mainsplitter = QSplitter(Qt.Horizontal)
        self.mainsplitter.addWidget(leftpanel_w)
        self.mainsplitter.addWidget(self.graphsplitter)

        self.mainHBOX.addWidget(self.mainsplitter)

        self.centralWidget.setLayout(self.mainHBOX)


        self.Lines = []
        self.cleargraph()

        self.grid_range = (0, 0)
    # def CB_Dend_action(self):
    #     if self.CB_Dend.isChecked():
    #         pen = pg.mkPen(color=(0, 255, 0), width=1, style=Qt.SolidLine)
    #         self.graphwidget.plot(self.t, self.pyrVd[0, :], pen=pen)
    #     else:
    #         pen = pg.mkPen(color=(255, 255, 255), width=1, style=Qt.SolidLine)
    #         self.graphwidget.plot(self.t, self.pyrVd[0, :], pen=pen)
    #
    #
    # def CB_ais_action(self):
    #     if self.CB_ais.isChecked():
    #         pen = pg.mkPen(color=(255, 0, 0), width=1, style=Qt.SolidLine)
    #         self.graphwidget.plot(self.t, self.pyrVa[0, :], pen=pen)
    #     else:
    #         pen = pg.mkPen(color=(255, 255, 255), width=1, style=Qt.SolidLine)
    #         self.graphwidget.plot(self.t, self.pyrVa[0, :], pen=pen)

    def CB_Dend_action(self):
        if self.CB_Dend.isChecked():
            self.graphwidget.plot(self.pyrVd[0, :], self.t, Colors=(0, 1, 0),Names = 'Dent')
        # else:
        #     pen = pg.mkPen(color=(255, 255, 255), width=1, style=Qt.SolidLine)
        #     self.graphwidget.plot(self.t, self.pyrVd[0, :], pen=pen)


    def CB_ais_action(self):
        if self.CB_ais.isChecked():
            # pen = pg.mkPen(color=(255, 0, 0), width=1, style=Qt.SolidLine)
            self.graphwidget.plot(self.pyrVa[0, :], self.t,Colors=(1, 0, 0), Names='ais')
        # else:
        #     pen = pg.mkPen(color=(255, 255, 255), width=1, style=Qt.SolidLine)
        #     self.graphwidget.plot(self.t, self.pyrVa[0, :], pen=pen)

    # def cleargraph(self):
    #
    #     for line in self.Lines:
    #         line.clear()
    #
    #     self.Lines = []
    #     self.graphwidget.clear()

    def cleargraph(self):
        self.graphwidget.clear()

    def loadparam(self):
        Fs = int(self.Fs.text())  # Khz
        T = int(self.T.text())
        self.dt = 1 / Fs
        nbEch = int(T / self.dt)
        self.t = np.arange(nbEch) * self.dt
        self.tps_start = 0
        self.pyrVs = np.zeros((1, len(self.t)))
        self.pyrVd = np.zeros((1, len(self.t)))
        self.pyrVa = np.zeros((1, len(self.t)))
        self.Stim_Signal = np.zeros((1, len(self.t)))
        # stim
        I_inj = float64(self.I_soma.text())
        stimDur = int(self.stim_T.text())
        t0 = 100
        nbOfSamplesStim = int(stimDur / self.dt)
        Soma_A=np.array([1.8,2.365,3.91,1.881]) #surface areas of somas layers 2/3, 4, 5 and 6
        if self.CB_inj.currentText()=='pA':
            self.Stim_Signal[0, int(t0 / self.dt):int(t0 / self.dt) + nbOfSamplesStim - 1] = I_inj*Soma_A[self.CB_PYR_l.currentIndex()]

        else:
            self.Stim_Signal[0, int(t0 / self.dt):int(t0 / self.dt) + nbOfSamplesStim - 1] = I_inj

        ###
        ##get characteristics
        if self.Compartiment2_RB.isChecked():
            self.Neurone= PyrCellCA1.pyrCellCA1(1)
            self.Neurone.p = float(self.p_SD.text())
            self.Neurone.Cm = float(self.Cm_s.text())
            self.Neurone.g_c_SD = float(self.gc_SD.text())
            self.Neurone.g_leak = float(self.A_g_Leak_s.text())



        else:
            self.Neurone = PC_neo.pyrCellneo(1)
            self.Neurone.g_Na_a = float(self.A_g_Na.text())
            self.Neurone.g_KDR_a = float(self.A_g_KDR.text())
            self.Neurone.p_SA = float(self.p_SA.text())
            self.Neurone.p_SD = float(self.p_SD.text())
            self.Neurone.Cm = float(self.Cm_s.text())
            self.Neurone.Cm_d = float(self.Cm_d.text())
            self.Neurone.g_c_SA = float(self.gc_SA.text())
            self.Neurone.g_c_SD = float(self.gc_SD.text())
            self.Neurone.g_leak_a = float(self.A_g_Leak_a.text())
            self.Neurone.g_leak_s = float(self.S_g_Leak_s.text())
            self.Neurone.g_leak_d = float(self.D_g_Leak_d.text())

        self.Neurone.dt = self.dt
        self.Neurone.E_Na = float(self.E_Na.text())
        self.Neurone.E_K = float(self.E_K.text())
        self.Neurone.E_Ca = float(self.E_Ca.text())
        self.Neurone.E_h = float(self.E_h.text())
        self.Neurone.E_leak = float(self.E_leak.text())
        self.Neurone.g_Na_s = float(self.S_g_Na.text())
        self.Neurone.g_KDR_s = float(self.S_g_KDR.text())
        self.Neurone.g_CaL_s = float(self.S_g_CaL.text())
        self.Neurone.g_AHP_s = float(self.S_g_AHP.text())
        self.Neurone.g_AHP_BK_s = float(self.S_g_BK.text())
        self.Neurone.g_m_s = float(self.S_g_m.text())
        self.Neurone.g_KNa = float(self.S_g_KNa.text())


        self.Neurone.g_Na_d = float(self.D_g_Na.text())
        self.Neurone.g_KDR_d = float(self.D_g_KDR.text())
        self.Neurone.g_CaR_d = float(self.D_g_CaR.text())
        self.Neurone.g_CaT_d = float(self.D_g_CaT.text())
        self.Neurone.g_h_d = float(self.D_g_h.text())
        self.Neurone.g_m_d = float(self.D_g_m.text())
        self.Neurone.g_AHP_d = float(self.D_g_AHP.text())
        self.Neurone.g_AHP_BK_d = float(self.D_g_BK.text())
        self.Neurone.PERCENT = float(self.Noise.text())


        text = self.RangeSimu_CB.currentText()
        text2 = self.RangeSimu2_CB.currentText()
        self.RangeSimu_CB.clear()
        if self.Compartiment2_RB.isChecked():
            self.RangeSimu_CB.addItems(PyrCellCA1.get_Variable_Names())
            self.RangeSimu2_CB.addItems(PyrCellCA1.get_Variable_Names())

        else:
            self.RangeSimu_CB.addItems(PC_neo.get_Variable_Names())
            self.RangeSimu2_CB.addItems(PC_neo.get_Variable_Names())

        if text in [self.RangeSimu_CB.itemText(i) for i in range(self.RangeSimu_CB.count())]:
            self.RangeSimu_CB.setCurrentIndex(self.RangeSimu_CB.findText(text))
        if text2 in [self.RangeSimu2_CB.itemText(i) for i in range(self.RangeSimu2_CB.count())]:
            self.RangeSimu2_CB.setCurrentIndex(self.RangeSimu2_CB.findText(text2))


    def clickme(self,dum = None, color=(0,0,0),name='',plot = True, reload = True ):
        if reload :
            self.loadparam()
        for tt, tp in enumerate(self.t):
            self.tps_start += self.dt
            self.Neurone.init_I_syn()
            self.Neurone.updateParameters()
            self.Neurone.I_soma_stim = self.Stim_Signal[0, tt]
            self.Neurone.rk4()
            self.pyrVs[0, tt] = self.Neurone.y[0]
            self.pyrVd[0, tt] = self.Neurone.y[1]
            if self.Compartiment3_RB.isChecked():
                self.pyrVa[0, tt] = self.Neurone.y[28]

        # pen = pg.mkPen(color=color, width=2, style=Qt.SolidLine)
        # line = self.graphwidget.plot(self.t, self.pyrVs[0, :],pen=pen,name=name)
        # self.Lines.append(line)
        if plot:
            self.graphwidget.plot(self.pyrVs[0, :],self.t, Names=name, Colors=color)

        # return line

    def updateplot(self,x,y):
        self.cleargraph()
        name = self.Param1str + ' : ' + str(self.yv[int(self.coordx), 0]) + '  ' + self.Param2str + ' : ' + str(self.xv[0, int(self.coordy)])
        self.graphwidget.plot(self.lfp[x, y, :], self.t, Names=name, Colors=(0,0,0))

    def clickmemulti(self):
        self.loadparam()
        name = self.RangeSimu_CB.currentText()
        rangeval = np.arange(float(self.RangeSimu_s.text()),
                             float(self.RangeSimu_e.text())+float(self.RangeSimu_step.text()),
                             float(self.RangeSimu_step.text()))
        lines = []
        for val in rangeval:
            color = tuple( random.uniform(0.0, 1.0)   for i in range(3))
            setattr(self.Neurone,name, val)
            self.clickme(color=color,name=name + ' : ' + str(val), reload=False)
            # lines.append(line)
            # self.parent.processEvents()
        self.graphwidget.addLegend()

    def clickmemultiIstim(self):
        self.loadparam()
        rangeval = np.arange(float(self.IstimSimu_s.text()),
                             float(self.IstimSimu_e.text())+float(self.IstimSimu_step.text()),
                             float(self.IstimSimu_step.text()))
        lines = []

        self.lfp = np.zeros((len(rangeval), self.t.shape[0]))

        for i, val in enumerate(rangeval):
            color = tuple( random.uniform(0.0, 1.0)   for i in range(3))

            stimDur = int(self.stim_T.text())
            t0 = 100
            nbOfSamplesStim = int(stimDur / self.dt)
            self.Stim_Signal[0, int(t0 / self.dt):int(t0 / self.dt) + nbOfSamplesStim - 1] = val

            self.clickme(color=color,name='Istim' + ' : ' + str(val), reload=False)
            self.lfp[i,:] = self.pyrVs[0, :]
            # lines.append(line)
            # self.parent.processEvents()

        self.graphwidget.addLegend()
        threshold = float(self.D_Threshold.text())
        res = np.zeros(len(rangeval))
        for i, val in enumerate(rangeval):
            lfp = self.lfp[i, :] * 1.
            lfp2 = self.lfp[i, :] * 1.
            lfp2[lfp > threshold] = 1
            lfp2[lfp <= threshold] = 0
            # res[i, j] = np.sum(np.diff(lfp2)==1)
            PA_instant = np.where(np.diff(lfp2) == 1)
            if len(PA_instant[0]) <=1:
                res[i] = 0
            else:
                PA_Periode = np.diff(np.array(PA_instant) * self.dt)
                res[i] = 1000 / np.mean(PA_Periode)
        self.graphwidget.plotAxis2(rangeval, res)




    def clickmegrid(self):
        self.loadparam()
        self.stop = False

        parametre1 = self.RangeSimu_CB.currentText()
        self.Param1str = parametre1
        param1 = np.arange(float(self.RangeSimu_s.text()),
                             float(self.RangeSimu_e.text())+float(self.RangeSimu_step.text()),
                             float(self.RangeSimu_step.text()))

        parametre2 = self.RangeSimu2_CB.currentText()
        self.Param2str = parametre2
        param2 = np.arange(float(self.RangeSimu2_s.text()),
                             float(self.RangeSimu2_e.text())+float(self.RangeSimu2_step.text()),
                             float(self.RangeSimu2_step.text()))


        parametre1_o = getattr(self.Neurone, parametre1)
        parametre2_o = getattr(self.Neurone, parametre2)



        self.xv, self.yv = np.meshgrid(param2, param1)
        self.lfp = np.zeros((len(param1), len(param2), self.t.shape[0]))
        self.lfp_states = [[self.listcolor_lfp_states_Sig[6]] * len(param2) for i in range(len(param1))]
        self.mascenegrid.updategrid(legende='spike')

        xlen, ylen = self.xv.shape
        start_time = time.time()


        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.gridupdatetimerfunc)
        self.timer.start(1000)
        if 0:
            for i in range(xlen):
                for j in range(ylen):
                    setattr(self.Neurone, parametre1, self.yv[i, j])
                    setattr(self.Neurone, parametre2, self.xv[i, j])
                    self.clickme(plot = False)

                    self.lfp[i, j, :] = self.pyrVs[0, :]
                    self.lfp_states[i][j] = self.listcolor_lfp_states_Sig[4]
                    self.parent.processEvents()

            self.timer.stop()
            setattr(self.Neurone, parametre1, parametre1_o)
            setattr(self.Neurone, parametre2, parametre2_o)
            # self.gridupdatetimerfunc()
            self.Spike_detection()
        else:
            list_neurons = []
            for i in range(xlen):
                neurons = []
                for j in range(ylen):
                    self.loadparam()
                    setattr(self.Neurone, parametre1, self.yv[i, j])
                    setattr(self.Neurone, parametre2, self.xv[i, j])
                    neurons.append(self.Neurone)
                list_neurons.append(neurons)
            print('end list neurons')

            # for tt, tp in enumerate(self.t):
            #     self.tps_start += self.dt
            #     for i in range(xlen):
            #         for j in range(ylen):
            #             n= list_neurons[i][j]
            #             n.init_I_syn()
            #             n.updateParameters()
            #             n.I_soma_stim = self.Stim_Signal[0, tt]
            #             n.Euler()
            #             self.lfp[i, j, tt] = n.y[0]
            self.compute(self.t, xlen, ylen, nb.typed.List(list_neurons), self.Stim_Signal, self.lfp)

            print('end computing')
            self.timer.stop()
            setattr(self.Neurone, parametre1, parametre1_o)
            setattr(self.Neurone, parametre2, parametre2_o)
            # self.gridupdatetimerfunc()
            self.Spike_detection()


    @staticmethod
    @njit
    def compute(t,xlen,ylen,list_neurons,Stim_Signal,lfp):
        for tt, tp in enumerate(t):
            for i in range(xlen):
                for j in range(ylen):
                    n = list_neurons[i][j]
                    n.init_I_syn()
                    n.updateParameters()
                    n.I_soma_stim = Stim_Signal[0, tt]
                    n.Euler()
                    lfp[i, j, tt] = n.y[0]

    def Do_Detection(self):
        threshold = float(self.D_Threshold.text())

        if self.NB_PA_RB.isChecked():
            xlen, ylen = self.xv.shape

            res = np.zeros((xlen, ylen))
            for i in range(xlen):
                for j in range(ylen):

                    lfp = self.lfp[i, j, :]*1.
                    lfp2 = self.lfp[i, j, :] * 1.
                    lfp2[lfp>threshold] = 1
                    lfp2[lfp<=threshold] = 0
                    res[i, j] = np.sum(np.diff(lfp2)==1)

            maxi = res.max()
            mini = res.min()

            cmap = cm.get_cmap('viridis')
            Colors = np.uint8(cmap((res-mini)/(maxi-mini)) * 255)[:, :, :3]
            for i  in range(xlen):
                for j in range(ylen):
                    self.lfp_states[i][j] = '#%02x%02x%02x' % tuple(Colors[i,j,:])
            self.grid_range = (res.min(), res.max())
            self.gridupdatetimerfunc()
        elif self.Fs_PA_RB.isChecked():

            xlen, ylen = self.xv.shape

            res = np.zeros((xlen, ylen))
            for i in range(xlen):
                for j in range(ylen):

                    lfp = self.lfp[i, j, :]*1.
                    lfp2 = self.lfp[i, j, :] * 1.
                    lfp2[lfp>threshold] = 1
                    lfp2[lfp<=threshold] = 0
                    # res[i, j] = np.sum(np.diff(lfp2)==1)
                    PA_instant = np.where(np.diff(lfp2) == 1)
                    if len(PA_instant[0]) == []:
                        res[i, j] = 0
                    else:
                        PA_Periode = np.diff(np.array(PA_instant)*self.dt)
                        res[i, j] = 1000 / np.mean(PA_Periode)

            maxi = res.max()
            mini = res.min()

            cmap = cm.get_cmap('viridis')
            Colors = np.uint8(cmap((res - mini) / (maxi - mini)) * 255)[:, :, :3]
            for i in range(xlen):
                for j in range(ylen):
                    self.lfp_states[i][j] = '#%02x%02x%02x' % tuple(Colors[i, j, :])
            self.grid_range = (np.round(res.min()*100)/100, np.round(res.max()*100)/100)
            self.gridupdatetimerfunc()

    def Spike_detection(self):
        xlen, ylen = self.xv.shape

        res = np.zeros((xlen, ylen))
        for i in range(xlen):
            for j in range(ylen):

                lfp = self.lfp[i, j, :]*1.
                lfp[lfp>0] = 1
                lfp[lfp<=0] = 0
                res[i, j] = np.sum(np.diff(lfp)==1)

        maxi = res.max()
        mini = res.min()

        cmap = cm.get_cmap('viridis')
        Colors = np.uint8(cmap((res-mini)/(maxi-mini)) * 255)[:, :, :3]
        for i  in range(xlen):
            for j in range(ylen):
                self.lfp_states[i][j] = '#%02x%02x%02x' % tuple(Colors[i,j,:])
        self.grid_range = (res.min(), res.max())
        self.gridupdatetimerfunc()



    def gridupdatetimerfunc(self):
        self.mascenegrid.updategridcolors()

class Viewer(QGraphicsView):
    def __init__(self, parent=None):
        super(Viewer, self).__init__(parent)

        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')  # Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.clear()
        # self.axes = self.figure.add_subplot(111)
        # self.axes.set_xlabel("Time (s)")
        # self.axes.set_ylabel("Vs (mV) (s)")

        self.canvas.setGeometry(0, 0, 1500, 500)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        self.figure.clear()
        # self.figure.subplots_adjust(left=0.03, bottom=0.02, right=0.99, top=0.95, wspace=0.0, hspace=0.0)
        self.axes = self.figure.add_subplot(111)
        self.gs = gridspec.GridSpec(1, 2)
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Vs (mV) (s)")
        self.canvas.draw_idle()

    def addLegend(self):
        self.axes.legend()
        self.canvas.draw_idle()

    def plot(self, x=None, t=None,  Names=None, Colors=None):
        # Fs = int(1 / (tp[1] - tp[0]))
        self.axes.plot(t, x, c=Colors, label = Names)
        # self.axes.autoscale_view()
        print('finished')
        self.canvas.draw_idle()

    def plotAxis2(self,x,y):
        self.axes.set_position(self.gs[0].get_position(self.figure))
        self.axes.set_subplotspec(self.gs[0])

        self.axes2 = self.figure.add_subplot(self.gs[1])
        self.axes2.set_ylabel('Hz' )  # we already handled the x-label with ax1
        self.axes2.plot(x, y )
        self.canvas.draw_idle()


class gridViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(gridViewer, self).__init__(parent)
        self.parent=parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # self._transformEnable =  True
        # self.setTransformationAnchor(QGraphicsView.NoAnchor)
        # self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(Qt.white))
        xlen , ylen = self.parent.xv.shape
        xsize=0.95*10
        ysize=0.95*10
        xv, yv = np.meshgrid(range(ylen), range(xlen))
        xv=xv*10
        yv=yv*10
        for i in range(xlen):
            for j in range(ylen):
                #print(xv[i,j],yv[i,j],xsize,ysize)
                self.scene.addRect(xv[i,j],yv[i,j],xsize,ysize )
        #print(yv)
        # for i in range(ylen):
        self.font=QFont("helvetica",10)
        text = self.scene.addText(str(self.parent.xv[0,0]),self.font)
        text.setPos(xv[0,0]-10, -20)
        # text.translate(xv[0,0]-5, -7)
        text = self.scene.addText(str(self.parent.xv[0,-1]),self.font)
        text.setPos(xv[0,-1]-10, -20)
        text = self.scene.addText('A',self.font)
        text.setPos(xv[0,int(ylen/2)]-10, -20)
        text = self.scene.addText(str(self.parent.yv[0,0]),self.font)
        text.setPos(-30, yv[0,0]-10)
        text = self.scene.addText(str(self.parent.yv[-1,0]),self.font)
        text.setPos(-30, yv[-1,0]-10)
        text = self.scene.addText('B',self.font)
        text.setPos(-30,yv[int(xlen/2),0]-10 )
            # newtext = QGraphicsTextItem(str(self.parent.xv[0,i]))
            # font=QFont()
            # font.setPixelSize(1)
            # newtext.setFont(font)
            # newtext.setPos(xv[0,i], -10)
            # print((xv[0,i]))
            # self.scene.addItem(newtext)

        # self.scale(xv.max(),yv.max())
        self.scene.update()
        self.selectedplot=[]
        self.selectedplot_x=[]
        self.selectedplot_y=[]

        return

    def updategridcolors(self):
        # try:
            listobj = self.scene.items()
            rectangle = QGraphicsRectItem( )
            listRect = [l for l in listobj if type(l) == type(rectangle)]


            for rect in listRect:
                j=int(round(rect.boundingRect().x())/10)
                i=int(round(rect.boundingRect().y())/10)
                if j >=0:
                    rect.setBrush(QColor(self.parent.lfp_states[i][j]))

            self.text_high.setPlainText(str(self.parent.grid_range[1]))
            self.text_low.setPlainText(str(self.parent.grid_range[0]))


            cmap = cm.get_cmap('viridis')
            Colors = np.uint8(cmap(np.linspace(0, 1, 8)) * 255)[:, :3]
            list_c = []
            for c in Colors:
                list_c.append('#%02x%02x%02x' % tuple(c))
            for i, r in enumerate(self.list_r):
                rect.setBrush(QColor(list_c[i]))
    # except:
        #     pass



    def updategrid(self,legende = 'spike'):



        for item in self.scene.items():
            self.scene.removeItem(item)
        xlen , ylen = self.parent.xv.shape
        xsize=0.95*10
        ysize=0.95*10
        xv, yv = np.meshgrid(range(ylen), range(xlen))
        xv=xv*10
        yv=yv*10

        Textpos = -100
        if legende == 'freq':
            self.scene.addRect(-100,1*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.black))
            text = self.scene.addText('Not computed',self.font)
            text.setPos(Textpos - text.boundingRect().width(),1*xsize-8 )
            self.scene.addRect(-100,3*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.darkGreen))
            text = self.scene.addText('SWO',self.font)
            text.setPos(Textpos - text.boundingRect().width(),3*xsize-8 )
            self.scene.addRect(-100,5*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.darkMagenta))
            text = self.scene.addText('Delta',self.font)
            text.setPos(Textpos - text.boundingRect().width(),5*xsize-8 )
            self.scene.addRect(-100,7*xsize,xsize,ysize,QPen(Qt.black),QBrush(QColor(0, 150, 210)))
            text = self.scene.addText('Teta',self.font)
            text.setPos(Textpos - text.boundingRect().width(),7*xsize-8 )
            self.scene.addRect(-100,9*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.red))
            text = self.scene.addText('Alpha',self.font)
            text.setPos(Textpos - text.boundingRect().width(),9*xsize-8 )
            self.scene.addRect(-100,11*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.blue))
            text = self.scene.addText('Beta',self.font)
            text.setPos(Textpos - text.boundingRect().width(),11*xsize-8)
            self.scene.addRect(-100, 13 * xsize, xsize, ysize, QPen(Qt.black), QBrush(Qt.darkGray))
            text = self.scene.addText('Gamma', self.font)
            text.setPos(Textpos - text.boundingRect().width(), 13 * xsize - 8)
            self.scene.addRect(-100,15*xsize,xsize,ysize,QPen(Qt.black),QBrush(Qt.darkYellow))
            text = self.scene.addText('selected for',self.font)
            text.setPos(Textpos - text.boundingRect().width(),15*xsize-8)
            text = self.scene.addText('curve plot',self.font)
            text.setPos(Textpos - text.boundingRect().width(),17*xsize-8)
        elif legende == 'spike':
            cmap = cm.get_cmap('viridis')
            Colors = np.uint8(cmap(np.linspace(0, 1, 8)) * 255)[:, :3]
            list_c = []
            for c in Colors:
                list_c.append('#%02x%02x%02x' % tuple(c))
            rect1=self.scene.addRect(-100, 1 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[7]))
            self.text_high = self.scene.addText(str(self.parent.grid_range[1]), self.font)
            self.text_high.setPos(Textpos - self.text_high.boundingRect().width(), 1 * xsize - 8)
            rect2=self.scene.addRect(-100, 2 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[6]))
            # text = self.scene.addText('Background', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 3 * xsize - 8)
            rect3=self.scene.addRect(-100, 3 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[5]))
            # text = self.scene.addText('Spike', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 5 * xsize - 8)
            rect4=self.scene.addRect(-100, 4 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[4]))
            # text = self.scene.addText('Rythmic', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 7 * xsize - 8)
            rect5=self.scene.addRect(-100, 5 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[3]))
            # text = self.scene.addText('Alpha', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 9 * xsize - 8))
            rect6=self.scene.addRect(-100, 6 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[2]))
            # text = self.scene.addText('Beta', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 11 * xsize - 8)
            rect7=self.scene.addRect(-100, 7 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[1]))
            # text = self.scene.addText('Gamma', self.font)
            # text.setPos(Textpos - text.boundingRect().width(), 13 * xsize - 8)
            rect8=self.scene.addRect(-100, 8 * xsize, xsize, ysize, QPen(Qt.black), QColor(list_c[0]))
            self.text_low = self.scene.addText(str(self.parent.grid_range[0]), self.font)
            self.text_low.setPos(Textpos - self.text_low.boundingRect().width(), 8 * xsize - 8)

            self.list_r = [rect8,rect7,rect6,rect5,rect4,rect3,rect2,rect1]
        for i in range(xlen):
            for j in range(ylen):

                if np.sum(self.parent.lfp[i,j,:])==0:
                    self.scene.addRect(xv[i,j],yv[i,j],xsize,ysize,QPen(Qt.black),QBrush(Qt.black))
                else:
                    self.scene.addRect(xv[i, j], yv[i, j], xsize, ysize, QPen(Qt.black), QBrush(QColor(self.parent.lfp_states[i][j])))



        # for i in range(ylen):
        text = self.scene.addText(str(self.parent.xv[0,0]),self.font)
        text.setPos(xv[0,0]-10, -20)
        # text.translate(xv[0,0]-5, -7)
        text = self.scene.addText(str(self.parent.xv[0,-1]),self.font)
        text.setPos(xv[0,-1]-10, -20)
        text = self.scene.addText(self.parent.Param2str ,self.font)
        text.setPos(xv[0,int(ylen/2)]-10, -20)
        text = self.scene.addText(str(self.parent.yv[0,0]),self.font)
        text.setPos(-30, yv[0,0]-10)
        text = self.scene.addText(str(self.parent.yv[-1,0]),self.font)
        text.setPos(-30, yv[-1,0]-10)
        text = self.scene.addText(self.parent.Param1str,self.font)
        text.setPos(-30,yv[int(xlen/2),0]-10 )

        self.setSceneRect(-100,-100,  ylen*10+100, xlen*10+100)
        self.scene.update()
        # self.selectedplot= []
        if not self.selectedplot==[]:
            self.selectedplot=self.scene.addRect(self.selectedplot_y*10,self.selectedplot_x*10,10,10,QPen(Qt.black),QBrush(Qt.darkYellow))
        return

    def keyPressEvent(self, pe):
        qpt =  self.selectedplot.rect().center()
        if pe.key() == Qt.Key_Down:
            qpt.setY(qpt.y()+10)
        elif pe.key() == Qt.Key_Up:
            qpt.setY(qpt.y()-10)
        elif pe.key() == Qt.Key_Right:
            qpt.setX(qpt.x()+10)
        elif pe.key() == Qt.Key_Left:
            qpt.setX(qpt.x()-10)
        self.click_move_update(pe, pos=qpt)


    def mousePressEvent(self, event):
        self.click_move_update(event)

    def mouseMoveEvent(self, event):
        self.click_move_update(event)

    def click_move_update(self, event, pos=None):
        if not pos==None:
            position = pos
            for item in self.scene.items():
                if item.type() == QGraphicsRectItem().type():
                    if item.contains(position):
                        rectangle = item.boundingRect()
                        self.parent.coordx = round(rectangle.top() / 10.)
                        self.parent.coordy = round(rectangle.left() / 10.)
                        if self.selectedplot == []:
                            self.selectedplot = self.scene.addRect(self.parent.coordy * 10, self.parent.coordx * 10, 10, 10, QPen(Qt.black), QBrush(Qt.darkYellow))
                            self.selectedplot_x = self.parent.coordx
                            self.selectedplot_y = self.parent.coordy

                        else:
                            self.scene.removeItem(self.selectedplot)
                            self.selectedplot = self.scene.addRect(self.parent.coordy * 10, self.parent.coordx * 10, 10, 10, QPen(Qt.black), QBrush(Qt.darkYellow))
                            self.selectedplot_x = self.parent.coordx
                            self.selectedplot_y = self.parent.coordy
                        self.parent.updateplot(self.parent.coordx,self.parent.coordy)

        elif bool(event.buttons() & Qt.LeftButton):
            position = QPointF(event.pos())
            position = self.mapToScene(position.x(),position.y())
            for item in self.scene.items():
                if item.type() == QGraphicsRectItem().type():
                    if item.contains(position):
                        rectangle=item.boundingRect()
                        self.parent.coordx = round(rectangle.top()/10.)
                        self.parent.coordy = round(rectangle.left()/10.)
                        if self.selectedplot==[]:
                            self.selectedplot=self.scene.addRect(self.parent.coordy*10,self.parent.coordx*10,10,10,QPen(Qt.black),QBrush(Qt.darkYellow))
                            self.selectedplot_x=self.parent.coordx
                            self.selectedplot_y=self.parent.coordy

                        else:
                            self.scene.removeItem(self.selectedplot)
                            self.selectedplot=self.scene.addRect(self.parent.coordy*10,self.parent.coordx*10,10,10,QPen(Qt.black),QBrush(Qt.darkYellow))
                            self.selectedplot_x=self.parent.coordx
                            self.selectedplot_y=self.parent.coordy
                        self.parent.updateplot(self.parent.coordx,self.parent.coordy)


    def wheelEvent(self, event):
        if event.buttons() == Qt.RightButton:
            position = QPointF(event.pos())
            position = self.mapToScene(position.x(),position.y())
            rect = self.mapToScene(self.viewport().geometry()).boundingRect()
            w=(rect.right()-rect.left())/2.
            h=(rect.bottom()-rect.top())/2.
            rect.setLeft(position.x()-w)
            rect.setTop(position.y()-h)
            rect.setRight(w*2.)
            rect.setBottom(h*2.)
            self.setSceneRect(rect.left(),rect.top(),rect.right(),rect.bottom())
        if event.angleDelta().y()  > 0:
            factor = 1.25

        else:
            factor = 0.8
        #
        self.scale(factor,factor)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = Ui_Main(app)
    ui.setWindowTitle('Large Scale GUI')
    ui.show()
    sys.exit(app.exec_())