__author__ = 'Maxime'
# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import sys
import matplotlib
import numpy as np
from scipy.spatial import distance
from scipy import sparse
from scipy.fft import fftshift
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import inspect
import copy
import subprocess
import struct
import random
import csv

import Column_morphology

matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import MultipleLocator
from scipy import signal
import scipy.io
# import pyedflib
import datetime
import scipy.io as sio
from EEGViewer import lfpViewer_EEG
from Modify_Each_NMM_param import Modify_1_NMM
# from Modify_X_NMM_at_once import Modify_X_NMM
from Modify_X_NMM_VTK import Modify_X_NMM
from numba import guvectorize, float64, int64, njit
from Graph_viewer3D_VTK5 import Graph_viewer3D_VTK
from Graph_EField_VTK import Graph_EField_VTK
from scipy.optimize import curve_fit
import RecordedPotential
import Electrode
import platform
import Cell_morphology
import Connectivity
import CreateColumn


class Spoiler(QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = animationDuration
        self.toggleAnimation = QParallelAnimationGroup()
        self.contentArea = QScrollArea()
        self.headerLine = QFrame()
        self.toggleButton = QToolButton()
        self.mainLayout = QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)

        headerLine = self.headerLine
        headerLine.setFrameShape(QFrame.HLine)
        headerLine.setFrameShadow(QFrame.Sunken)
        headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = Qt.DownArrow if checked else Qt.RightArrow
            direction = QAbstractAnimation.Forward if checked else QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)
        toggleButton.setChecked(True)
        start_animation(True)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount() - 1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)
        
if platform.system() == "Windows":
    pass
elif platform.system() == "Darwin":
    matplotlib.rcParams.update({'font.size': 6})
elif platform.system() == "Linux":
    pass


def peakdet(v, delta, x=None):
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

@guvectorize(["float64[:,:],int64, int64, float64, float64, int64, float64, int64, int64, float64, float64[:,:]"], '(n,m),(),(),(),(),(),(),(),(),()->(n,m)')
def Generate_Stim_Signals(Stim_Signals_in, seed, nbOfSamplesStim, i_inj, tau, nbStim, varianceStim, nb_Stim_Signals, nbEch, dt,Stim_Signals_out):
    n = int(1. * nbEch / 2.)
    t = np.arange(nbEch) * dt

    if not seed == 0:
        np.random.seed(seed)
    # else:
    #     np.random.seed()
    for St in range(nb_Stim_Signals):
        y2 = np.zeros(nbEch)
        for i in range(nbStim):

            y = np.zeros(t.shape)

            intervalle_dt = int(np.round((np.random.normal(0, varianceStim))))

            if ((n + intervalle_dt) < 0):
                intervalle_dt = -n
            if ((n + intervalle_dt + nbOfSamplesStim) > nbEch):
                intervalle_dt = nbEch - n - nbOfSamplesStim - 1

            for tt, tp in enumerate(t):
                if (tt > (n + intervalle_dt)):
                    y[tt] = (1. - np.exp(-(tp - (n + intervalle_dt) * dt) / tau)) * i_inj
                if (tt > (n + intervalle_dt + nbOfSamplesStim)):
                    y[tt] = (np.exp(-(tp - ((n + intervalle_dt) + nbOfSamplesStim) * dt) / tau)) * y[n + intervalle_dt + nbOfSamplesStim - 1]

                y2[tt] += y[tt]

        Stim_Signals_out[St,:]=y2
    # return Stim_Signals

@guvectorize(["int64[:], int64, int64, int64, int64[:]"], '(n),(),(),()->(n)')
def pickcell_simple_fun(size, loop,low, high,pickedcell):
    while loop > 0:
        # cell = np.random.randint(low, high, size=1)
        cell = np.random.randint(low, high)
        if cell in pickedcell:
            pass
        else:
            pickedcell[loop-1] = cell
            loop -= 1

@guvectorize(["int64[:], int64, float64, float64, float64[:], int64[:], int64[:]"], '(n),(),(),(),(m),(k)->(n)')
def pickcell_gauss_fun(size, loop, mean, var, CellDistances, sortedIndex, pickedcell):
    while loop>0:
        l = abs(np.random.normal(mean, var))
        cellInd = np.where(CellDistances[sortedIndex]>=l)[0]
        if cellInd.size == 0:
            continue
        else:
            cellInd = cellInd[0]
        cell = sortedIndex[cellInd]
        if not cell in pickedcell :
            pickedcell[loop-1] = cell
            sortedIndex= np.delete(sortedIndex, cellInd)
            loop -= 1


@guvectorize(["float64[:], int64, int64[:]"], '(n),()->(n)')
def argsort_list_fun(CellDistances, plus, sortedIndex):
    sortedIndex = np.argsort(CellDistances) + plus


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


def set_QPushButton_background_color(button=None, color=None):
    if color==None or button==None :
        return
    else :
        button.setAutoFillBackground(True)
        values = "{r}, {g}, {b} ".format(r = color.red(),
                                     g = color.green(),
                                     b = color.blue())
        button.setStyleSheet("QPushButton { background-color: rgb("+values+"); }")

def label_color_clicked(event,button):
    color = QColor(button.palette().button().color())
    colordial = QColorDialog(color)
    colordial.exec_()
    selectedcolor = colordial.currentColor()
    colordial.close()
    set_QPushButton_background_color(button,selectedcolor)
    pass

def Layout_grid_Label_Edit(label = ['None'],edit =['None']):
    widget = QWidget()
    layout_range = QVBoxLayout()
    # layout_range.setContentsMargins(5,5,5,5)

    grid = QGridLayout()
    grid.setContentsMargins(5,5,5,5)
    widget.setLayout(grid)
    layout_range.addLayout(grid)
    Edit_List =[]
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Edit = LineEdit(edit[idx])
        grid.addWidget(Label, idx, 0)
        grid.addWidget(Edit, idx, 1)
        Edit_List.append(Edit)
    return widget, Edit_List




class Colonne_cortical_Thread(QThread):
    finished = pyqtSignal()
    updateTime = pyqtSignal(float)

    def __init__(self,CortexClass):
        QThread.__init__(self )
        self.CC = CortexClass()

        self.percent = 0.
        self.T = 0
        self.dt  = 0
        self.Stim_Signals = []

    @pyqtSlot(float)
    def updatePercent(self, pourcent):
        self.percent = pourcent
        self.updateTime.emit(self.percent)

    def __del__(self):
        self.wait()

    def get_percentage(self):
        return self.percent


    def arret(self):
        self.C.Stop = False


class ModelMicro_GUI(QMainWindow):
    def __init__(self, parent=None):
        super(ModelMicro_GUI, self).__init__()
        self.parent = parent

        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        elif __file__:
            self.application_path = os.path.dirname(__file__)
        sys.path.append(self.application_path)

        from CorticalColumn0 import CorticalColumn
        self.Colonne = CorticalColumn
        self.Colonne_cortical_Thread = Colonne_cortical_Thread(self.Colonne)
        self.CC = self.Colonne_cortical_Thread.CC
        self.CC.updateTime.something_happened.connect(self.updateTime)


        self.dt = 1/25
        self.T = 200
        self.nbEch = int(self.T / self.dt)
        self.List_PYR_Stimulated = []

        self.x = 0.
        self.y = 0.
        self.z = 0.

        self.electrode_pos = [0.,0.,2082.]

        self.createCells = True

        # some variable for widget's length
        self.Qbox_H = 30
        self.Qbox_W = 60
        self.Qedit_H = 30
        self.Qedit_W = 60

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        #### global layout
        self.mainHBOX = QHBoxLayout()
        self.setmarginandspacing(self.mainHBOX)
        ###

        wid_Param_VBOX = QWidget()
        self.Param_VBOX = QVBoxLayout()
        self.Param_VBOX.setAlignment(Qt.AlignTop)
        wid_Param_VBOX.setLayout(self.Param_VBOX)
        # wid_Param_VBOX.setMaximumWidth(300)
        self.setmarginandspacing(self.Param_VBOX)

        # call column layout
        self.set_Param_VBOX()

        self.Vsplitter_middle = QSplitter(Qt.Vertical)
        self.mascene_EEGViewer = lfpViewer_EEG()
        self.mascene_LFPViewer = LFPViewer(self)
        self.Vsplitter_middle.addWidget(self.mascene_EEGViewer)
        self.Vsplitter_middle.addWidget(self.mascene_LFPViewer)
        self.Vsplitter_middle.setStretchFactor(4, 0)
        self.Vsplitter_middle.setStretchFactor(1, 0)
        self.Vsplitter_middle.setSizes([1500,int(1500/4)])

        self.Vsplitter = QSplitter(Qt.Vertical)

        self.GraphWidget = QWidget()
        self.GraphLayout = QVBoxLayout()
        self.Graph_viewer = Graph_viewer3D_VTK(self)
        self.GraphinfoLayout = QHBoxLayout()
        labelr, r_e = Layout_grid_Label_Edit(label=['r'], edit=['50'])
        labell, l_e = Layout_grid_Label_Edit(label=['line'], edit=['5'])
        labels, s_e = Layout_grid_Label_Edit(label=['scale'], edit=['50'])
        self.r_e = r_e[0]
        self.l_e = l_e[0]
        self.s_e = s_e[0]
        self.RedrawVTK_PB = QPushButton('Redraw')
        self.GraphinfoLayout.addWidget(labelr)
        self.GraphinfoLayout.addWidget(self.r_e)
        self.GraphinfoLayout.addWidget(labell)
        self.GraphinfoLayout.addWidget(self.l_e)
        self.GraphinfoLayout.addWidget(labels)
        self.GraphinfoLayout.addWidget(self.s_e)
        self.GraphinfoLayout.addWidget(self.RedrawVTK_PB)
        self.GraphLayout.addLayout(self.GraphinfoLayout)
        self.GraphLayout.addWidget(self.Graph_viewer)
        self.GraphWidget.setLayout(self.GraphLayout)
        self.r_e.editingFinished.connect(self.updateVTKinfo)
        self.l_e.editingFinished.connect(self.updateVTKinfo)
        self.s_e.editingFinished.connect(self.updateVTKinfo)
        self.RedrawVTK_PB.clicked.connect(self.Graph_viewer.draw_Graph)

        self.masceneCM = CMViewer(self)
        self.VStimWidget = QWidget()
        self.VStimLayout = QVBoxLayout()
        self.masceneStim = StimViewer(self)
        self.DisplayStim_PB = QPushButton('DisplayStim')
        self.DisplayStim_PB.clicked.connect(self.DisplayStim_func)
        self.VStimLayout.addWidget(self.DisplayStim_PB)
        self.VStimLayout.addWidget(self.masceneStim)
        self.VStimWidget.setLayout(self.VStimLayout)
        self.Vsplitter.addWidget(self.GraphWidget)
        self.Vsplitter.addWidget(self.masceneCM)
        self.Vsplitter.addWidget(self.VStimWidget)
        self.Vsplitter.setStretchFactor(0, 0)
        self.Vsplitter.setStretchFactor(1, 0)
        self.Vsplitter.setStretchFactor(2, 0)
        self.Vsplitter.setStretchFactor(3, 0)
        self.Vsplitter.setSizes([1500,1500,1500,1500])

        ### add  vectical global splitters
        self.mainsplitter = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(wid_Param_VBOX)
        wid_Param_VBOX.setFixedWidth(400)
        scroll.setWidget(widget)
        # scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400 + 24)
        scroll.setAlignment(Qt.AlignTop)
        scroll.setWidgetResizable(True)
        self.mainsplitter.addWidget(scroll)

        self.mainsplitter.addWidget(self.Vsplitter_middle)
        self.mainsplitter.addWidget(self.Vsplitter)
        self.mainsplitter.setStretchFactor(0, 1)
        self.mainsplitter.setStretchFactor(1, 3)
        self.mainsplitter.setStretchFactor(2, 1)
        self.mainsplitter.setSizes([400, 1500 * 3, 1500])

        self.mainHBOX.addWidget(self.mainsplitter)

        self.centralWidget.setLayout(self.mainHBOX)

        # Menu
        # set actions
        extractLoad_Model = QAction("Load Model", self)
        extractLoad_Model.triggered.connect(self.LoadModel)

        extractLoad_Simul = QAction("Load Simulation", self)
        extractSave_Simul = QAction("Save Simulation", self)
        extractLoad_Simul.triggered.connect(self.LoadSimul)
        extractSave_Simul.triggered.connect(self.SaveSimul)

        extractLoad_Res = QAction("Load Results", self)
        extractSave_Res = QAction("Save Results", self)
        # extractLoad_Res.triggered.connect(self.LoadRes)
        extractSave_Res.triggered.connect(self.SaveRes)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileLoad = menubar.addMenu('&Load/Save')
        fileLoad.addAction(extractLoad_Model)
        fileLoad.addSeparator()
        fileLoad.addAction(extractLoad_Simul)
        fileLoad.addAction(extractSave_Simul)
        fileLoad.addSeparator()
        # fileLoad.addAction(extractLoad_Res)
        fileLoad.addAction(extractSave_Res)

        self.NewModify1NMM = None
        self.NewModifyXNMM = None
        self.selected_cells = []

        # self.update_cellNnumber()

    def updateVTKinfo(self):
        self.Graph_viewer.LinewidthfromGUI = float(self.l_e.text())
        self.Graph_viewer.radiuswidthfromGUI = float(self.r_e.text())
        self.Graph_viewer.setScales(float(self.s_e.text()))

    @pyqtSlot(float)
    def updateTime(self, pourcent):
        ts = time.time() - self.t0
        tf = ts / (pourcent + 0.00000001)
        tr = tf - ts
        self.msg.setText("Computation in progress\nPlease wait.\nTime spend : " + str(datetime.timedelta(seconds=int(ts))) + "\nTime remaining : " + str(datetime.timedelta(seconds=int(tr))))
        self.parent.processEvents()

    def DisplayStim_func(self):
        # self.Generate_Stim_Signals()
        self.masceneStim.update(self.CC.Stim_Signals,self.CC.Stim_InputSignals)

    def set_Param_VBOX(self):
        #tissue size
        self.tissue_size_GB = Spoiler(title=r'tissue information')
        labelD, D_e = Layout_grid_Label_Edit(label=['cylinder diameter / Square XY length / Rectangle X length (µm)'], edit=[str(self.CC.D)])
        labelL, L_e = Layout_grid_Label_Edit(label=['cylinder length / /Rectangle Y length (µm)'], edit=[str(self.CC.L)])
        labelCourbure, Courbure_e = Layout_grid_Label_Edit(label=['Distance for curvature'],
                                             edit=[str(2000)])
        self.D_e = D_e[0]
        self.L_e = L_e[0]
        self.C_e = Courbure_e[0]

        self.Layer_d1_l = LineEdit_Int(str(self.CC.Layer_d[0]))
        self.Layer_d23_l = LineEdit_Int(str(self.CC.Layer_d[1]))
        self.Layer_d4_l = LineEdit_Int(str(self.CC.Layer_d[2]))
        self.Layer_d5_l = LineEdit_Int(str(self.CC.Layer_d[3]))
        self.Layer_d6_l = LineEdit_Int(str(self.CC.Layer_d[4]))

        self.Apply_tissue_PB = QPushButton('Apply Tissue update')
        # grid = QGridLayout()

        self.Density_e = LineEdit('3954334')
        self.nbcell_e = LineEdit_Int('1000')

        self.get_XYZ_PB = QPushButton('Get XYZ')
        self.get_Density_PB = QPushButton('Get Density')
        self.get_nbcell_PB = QPushButton('Get Nb cells')
        line = 0
        grid = QGridLayout()
        grid.addWidget(labelD, line, 0, 1, 4)
        grid.addWidget(self.D_e, line, 4)
        line += 1
        grid.addWidget(labelL, line, 0, 1, 4)
        grid.addWidget(self.L_e, line, 4)
        line += 1
        grid.addWidget(labelCourbure, line, 0, 1, 4)
        grid.addWidget(self.C_e, line, 4)

        line += 1
        grid.addWidget(QLabel('thick 1'), line, 0)
        grid.addWidget(QLabel('thick 2/3'), line, 1)
        grid.addWidget(QLabel('thick 4'), line, 2)
        grid.addWidget(QLabel('thick 5'), line, 3)
        grid.addWidget(QLabel('thick 6'), line, 4)

        line += 1
        grid.addWidget(self.Layer_d1_l, line, 0)
        grid.addWidget(self.Layer_d23_l, line, 1)
        grid.addWidget(self.Layer_d4_l, line, 2)
        grid.addWidget(self.Layer_d5_l, line, 3)
        grid.addWidget(self.Layer_d6_l, line, 4)

        line += 1
        grid.addWidget(self.Apply_tissue_PB, line, 1, 1, 3)

        self.Apply_tissue_PB.clicked.connect(self.set_tissue_func)

        self.tissue_size_GB.setContentLayout(grid)
        self.tissue_size_GB.toggleButton.click()

        # %tage
        self.pourcentageCell_GB = Spoiler(title=r'% de cell')
        labelnb1 = QLabel('nb cells 1')
        labelnb23 = QLabel('nb cells 2/3')
        labelnb4 = QLabel('nb cells 4')
        labelnb5 = QLabel('nb cells 5')
        labelnb6 = QLabel('nb cells 6')
        labelnbtotal = QLabel('nb total')
        self.nbcellsnb1 = LineEdit_Int(str(int(self.CC.Layer_nbCells[0])))
        self.nbcellsnb23 = LineEdit_Int(str(int(self.CC.Layer_nbCells[1])))
        self.nbcellsnb4 = LineEdit_Int(str(int(self.CC.Layer_nbCells[2])))
        self.nbcellsnb5 = LineEdit_Int(str(int(self.CC.Layer_nbCells[3])))
        self.nbcellsnb6 = LineEdit_Int(str(int(self.CC.Layer_nbCells[4])))
        self.nbcellsnbtotal = LineEdit_Int(str(int(np.sum(self.CC.Layer_nbCells))))

        self.nbcellsnb1.returnPressed.connect(lambda s='1': self.Nb_Cell_Changed(s))
        self.nbcellsnb23.returnPressed.connect(lambda s='23': self.Nb_Cell_Changed(s))
        self.nbcellsnb4.returnPressed.connect(lambda s='4': self.Nb_Cell_Changed(s))
        self.nbcellsnb5.returnPressed.connect(lambda s='5': self.Nb_Cell_Changed(s))
        self.nbcellsnb6.returnPressed.connect(lambda s='6': self.Nb_Cell_Changed(s))
        self.nbcellsnbtotal.returnPressed.connect(lambda s='total': self.Nb_Cell_Changed(s))

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(labelnb1, 0, 0)
        grid.addWidget(labelnb23, 0, 1)
        grid.addWidget(labelnb4, 0, 2)
        grid.addWidget(labelnb5, 0, 3)
        grid.addWidget(labelnb6, 0, 4)
        grid.addWidget(labelnbtotal, 0, 5)
        grid.addWidget(self.nbcellsnb1, 1, 0)
        grid.addWidget(self.nbcellsnb23, 1, 1)
        grid.addWidget(self.nbcellsnb4, 1, 2)
        grid.addWidget(self.nbcellsnb5, 1, 3)
        grid.addWidget(self.nbcellsnb6, 1, 4)
        grid.addWidget(self.nbcellsnbtotal, 1, 5)

        # nbconnexion
        label_source = QLabel('Layer')
        label_PYR = QLabel('PYR')
        label_PV = QLabel('PV')
        label_SST = QLabel('SST')
        label_VIP = QLabel('VIP')
        label_RLN = QLabel('RLN')
        label_1 = QLabel('1')
        label_23 = QLabel('2/3')
        label_4 = QLabel('4')
        label_5 = QLabel('5')
        label_6 = QLabel('6')

        grid.addWidget(label_source, 4, 1, 1, 5, Qt.AlignHCenter)
        grid.addWidget(label_PYR, 6, 0)
        grid.addWidget(label_PV, 7, 0)
        grid.addWidget(label_SST, 8, 0)
        grid.addWidget(label_VIP, 9, 0)
        grid.addWidget(label_RLN, 10, 0)
        grid.addWidget(label_1, 5, 1)
        grid.addWidget(label_23, 5, 2)
        grid.addWidget(label_4, 5, 3)
        grid.addWidget(label_5, 5, 4)
        grid.addWidget(label_6, 5, 5)
        self.C = Column_morphology.Column(0)
        self.List_PYRpercent = []
        for l in range(len(self.CC.C.PYRpercent)):
            edit = LineEdit(str(self.CC.C.PYRpercent[l]))
            self.List_PYRpercent.append(edit)
            grid.addWidget(edit, 6, l + 1)

        self.List_PVpercent = []
        for l in range(len(self.CC.C.PVpercent)):
            edit = LineEdit(str(self.CC.C.PVpercent[l]))
            self.List_PVpercent.append(edit)
            grid.addWidget(edit, 7, l + 1)

        self.List_SSTpercent = []
        for l in range(len(self.CC.C.SSTpercent)):
            edit = LineEdit(str(self.CC.C.SSTpercent[l]))
            self.List_SSTpercent.append(edit)
            grid.addWidget(edit, 8, l + 1)

        self.List_VIPpercent = []
        for l in range(len(self.CC.C.VIPpercent)):
            edit = LineEdit(str(self.CC.C.VIPpercent[l]))
            self.List_VIPpercent.append(edit)
            grid.addWidget(edit, 9, l + 1)

        self.List_RLNpercent = []
        for l in range(len(self.CC.C.RLNpercent)):
            edit = LineEdit(str(self.CC.C.RLNpercent[l]))
            self.List_RLNpercent.append(edit)
            grid.addWidget(edit, 10, l + 1)


        # Compute cell number
        self.Apply_percentage_PB = QPushButton('Apply Percentage')
        grid.addWidget(self.Apply_percentage_PB, 11, 1, 1, 3)
        self.Apply_percentage_PB.clicked.connect(self.update_cellNnumber)

        self.pourcentageCell_GB.setContentLayout(grid)
        self.pourcentageCell_GB.toggleButton.click()

        # Connection matrixcc
        self.Afferences_GB = Spoiler(title=r'afference matrix')
        Afferences_GB_l = QVBoxLayout()
        self.Afferences_PB = QPushButton('Get Afference matrix')
        self.Afference_group = QButtonGroup(self)
        self.r0 = QRadioButton("Use percentage of the total number of cell")
        self.r1 = QRadioButton("Use fixed number of connection")
        self.Afference_group.addButton(self.r0)
        self.Afference_group.addButton(self.r1)
        self.r0.setChecked(True)
        Afferences_choice_l = QHBoxLayout()
        Afferences_choice_l.addWidget(self.r0)
        Afferences_choice_l.addWidget(self.r1)
        self.Connection_PB = QPushButton('See Connection number matrix')
        Afferences_GB_l.addLayout(Afferences_choice_l)
        Afferences_GB_l.addWidget(self.Afferences_PB)
        Afferences_GB_l.addWidget(self.Connection_PB)
        self.Afferences_PB.clicked.connect(self.update_connections)
        self.Connection_PB.clicked.connect(self.See_connections)
        self.r0.toggled.connect(self.update_connections_per_fixed)
        self.r1.toggled.connect(self.update_connections_per_fixed)
        self.Afferences_GB.setContentLayout(Afferences_GB_l)
        self.Afferences_GB.toggleButton.click()

        # cell placement
        self.cell_placement_GB = Spoiler(title=r"cell placement")
        self.cell_placement_CB = QComboBox()
        list = ['Cylinder','Square', 'Rectange','Cylinder with curvature','Square with curvature', 'Rectange with curvature']
        self.cell_placement_CB.addItems(list)
        self.cell_placement_PB = QPushButton('Place cells')
        self.cell_connectivity_PB = QPushButton('Compute connectivity')
        self.cell_keep_model_param_CB = QCheckBox('Keep model parameters')
        self.cell_keep_model_param_CB.setChecked(True)
        layoutseed = QHBoxLayout()
        self.seed_place = LineEdit_Int('0')
        layoutseed.addWidget(QLabel('Place Seed'))
        layoutseed.addWidget(self.seed_place)
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(self.cell_placement_CB, 0, 0, 1, 1)
        grid.addWidget(self.cell_placement_PB, 0, 1, 1, 2 )
        grid.addWidget(self.cell_connectivity_PB, 1, 1, 1, 2 )
        grid.addLayout(layoutseed, 1, 0, 1, 1)
        self.cell_placement_GB.setContentLayout(grid)
        self.cell_placement_PB.clicked.connect(self.PlaceCell_func)
        self.cell_connectivity_PB.clicked.connect(self.connectivityCell_func)

        self.EField_parameters_GB = Spoiler(title=r"E-Field parameters")
        self.EField_Conv_PB = QPushButton('Convert a txt E-Field')
        self.EField_file_PB = QPushButton('...')
        self.EField_file_TE = QLineEdit('')

        self.EField_TranslationX_LE = LineEdit('0')
        self.EField_TranslationY_LE = LineEdit('0')
        self.EField_TranslationZ_LE = LineEdit('0')
        lay_Trans = QHBoxLayout()
        lay_Trans.addWidget(QLabel('Translation'))
        lay_Trans.addWidget(QLabel('x'))
        lay_Trans.addWidget(self.EField_TranslationX_LE)
        lay_Trans.addWidget(QLabel('y'))
        lay_Trans.addWidget(self.EField_TranslationY_LE)
        lay_Trans.addWidget(QLabel('z'))
        lay_Trans.addWidget(self.EField_TranslationZ_LE)

        self.EField_RotationXY_LE = LineEdit('0')

        lay_Rot = QHBoxLayout()
        lay_Rot.addWidget(QLabel('Rotation'))
        lay_Rot.addWidget(QLabel('xy'))
        lay_Rot.addWidget(self.EField_RotationXY_LE)
        lay_Rot.addWidget(QLabel('Not Implemented yet'))

        self.EField_OnOff_CB = QCheckBox('On-Off')
        # self.EField_OnOff_CB.setEnabled(False)
        self.EField_Start_LE = LineEdit('0')
        self.EField_Length_LE = LineEdit('1000')

        self.EField_group = QButtonGroup(self)
        self.EField_File_RB = QRadioButton("Use File EField")
        self.EField_Const_RB = QRadioButton("Use Constant EField")
        self.EField_group.addButton(self.EField_File_RB)
        self.EField_group.addButton(self.EField_Const_RB)
        self.EField_Const_RB.setChecked(True)

        self.EField_Display_PB = QPushButton('Display')

        layu = QHBoxLayout()
        self.EField_Const_Ex = LineEdit('0')
        self.EField_Const_Ey = LineEdit('0')
        self.EField_Const_Ez = LineEdit('7')
        layu.addWidget(QLabel('Constant EField (mV/m)'))
        layu.addWidget(QLabel('Ex'))
        layu.addWidget(self.EField_Const_Ex)
        layu.addWidget(QLabel('Ey'))
        layu.addWidget(self.EField_Const_Ey)
        layu.addWidget(QLabel('Ez'))
        layu.addWidget(self.EField_Const_Ez)

        lay_stim = QHBoxLayout()
        self.EField_StimSig_CB = QComboBox()
        self.EField_StimSig_CB.addItems(['Constant','Sinusoidal', 'rectangular', 'triangular'])
        self.EField_StimSig_A_LE = LineEdit('1')
        self.EField_StimSig_F_LE = LineEdit('1')
        lay_stim.addWidget(self.EField_StimSig_CB)
        lay_stim.addWidget(QLabel('A'))
        lay_stim.addWidget(self.EField_StimSig_A_LE)
        lay_stim.addWidget(QLabel('F'))
        lay_stim.addWidget(self.EField_StimSig_F_LE)

        grid = QGridLayout()
        line = 0
        grid.addWidget(self.EField_Conv_PB, line, 1, 1, 3)
        line += 1
        grid.addWidget(QLabel('E-Field File (.mat)'), line, 0, 1, 2)
        grid.addWidget(self.EField_file_PB, line, 2, 1, 1)
        grid.addWidget(self.EField_file_TE, line, 3, 1, 2)
        line += 1

        grid.addWidget(self.EField_File_RB, line, 0, 1, 2)
        grid.addWidget(self.EField_Const_RB, line, 2, 1, 3)
        grid.addWidget(self.EField_Display_PB, line, 5, 1, 1)
        line += 1
        grid.addLayout(layu, line, 0, 1, 6)
        line += 1
        grid.addWidget(self.EField_OnOff_CB, line, 0, 1, 1)
        grid.addWidget(QLabel('Start'), line, 1,1,1)
        grid.addWidget(self.EField_Start_LE, line, 2,1,1)
        grid.addWidget(QLabel('Length'), line, 3,1,1)
        grid.addWidget(self.EField_Length_LE, line, 4,1,1)
        line += 1
        grid.addLayout(lay_stim, line, 0, 1, 4)

        self.EField_parameters_GB.setContentLayout(grid)
        self.EField_parameters_GB.toggleButton.click()

        self.EField_Conv_PB.clicked.connect(self.EField_Conv_Fun)
        self.EField_file_PB.clicked.connect(self.get_Efield_path)
        self.EField_Display_PB.clicked.connect(self.EField_Display_Fun)

        # Stim param I_inj=60, tau=4, stimDur=3, nbstim=5, varstim=12
        self.Stimulation_parameters_GB = QGroupBox(r"Stimulation parameters")
        label1 = QLabel('Stim duration (ms)')
        label2 = QLabel(r"Injected current density (µA/cm<sup>2</sup>)")
        label3 = QLabel("Time constant RC (ms)")
        label4 = QLabel("Stim number per cell")
        label5 = QLabel("Jitter variance (ms)")
        label6 = QLabel("seed")
        self.StimDuration_e = LineEdit('3')
        self.i_inj_e = LineEdit('60')
        self.tau_e = LineEdit('4')
        self.nbStim_e = LineEdit_Int('5')
        self.varianceStim_e = LineEdit('12')
        self.seed_e = LineEdit('0')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0)
        grid.addWidget(self.StimDuration_e, 0, 1)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(self.i_inj_e, 1, 1)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(self.tau_e, 2, 1)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(self.nbStim_e, 3, 1)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(self.varianceStim_e, 4, 1)
        grid.addWidget(label6, 5, 0)
        grid.addWidget(self.seed_e, 5, 1)
        self.Stimulation_parameters_GB.setLayout(grid)

        # InputTh param I_inj=25, tau=4, stimDur=3, nbstim=5, deltamin=14, delta=18)
        self.InputTh_parameters_GB = Spoiler(title=r"InputTh parameters")
        label1 = QLabel('Stim duration (ms)')
        label2 = QLabel(r"Injected current density (µA/cm<sup>2</sup>)")
        label3 = QLabel("Time constant RC (ms)")
        label4 = QLabel("Stim number per cell")
        label5 = QLabel("deltamin")
        label6 = QLabel("delta")
        self.TH_StimDuration_e = LineEdit('3')
        self.TH_i_inj_e = LineEdit('25')
        self.TH_tau_e = LineEdit('4')
        self.TH_nbStim_e = LineEdit_Int('5')
        self.TH_deltamin_e = LineEdit('14')
        self.TH_delta_e = LineEdit('18')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0)
        grid.addWidget(self.TH_StimDuration_e, 0, 1)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(self.TH_i_inj_e, 1, 1)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(self.TH_tau_e, 2, 1)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(self.TH_nbStim_e, 3, 1)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(self.TH_deltamin_e, 4, 1)
        grid.addWidget(label6, 5, 0)
        grid.addWidget(self.TH_delta_e, 5, 1)
        self.InputTh_parameters_GB.setContentLayout(grid)

        # Simulation parameters
        self.Simulation_parameters_GB = Spoiler(title=r"Simulation parameters")
        label1 = QLabel('Simulation duration (ms)')
        label2 = QLabel(r"Sampling frequency (kHz)")
        self.SimDuration_e = LineEdit('100')
        self.Fs_e = LineEdit('25')
        self.StimStart_e = LineEdit('50')
        self.UpdateModel_PB = QPushButton('Update Model')
        self.ModifyModel_PB = QPushButton('All Model Param')
        self.ModifyXModel_PB = QPushButton('Modify X Models')
        self.Reset_states_PB = QPushButton('Reset states')
        self.Run_PB = QPushButton('Run')
        self.displaycurves_CB = QCheckBox('Display Curves')
        self.displayVTK_CB = QCheckBox('Display VTK')
        self.displaycurve_per_e = LineEdit('100')
        displaycurve_per_l = QLabel(r'% to plot')
        self.displaycurve_per_e = LineEdit('100')

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0, 1, 2)
        grid.addWidget(self.SimDuration_e, 0, 2)
        grid.addWidget(label2, 1, 0, 1, 2)
        grid.addWidget(self.Fs_e, 1, 2)
        grid.addWidget(self.UpdateModel_PB, 0, 3)
        grid.addWidget(self.ModifyModel_PB, 1, 3)
        grid.addWidget(self.ModifyXModel_PB, 2, 3)
        grid.addWidget(self.Run_PB, 3, 3)
        grid.addWidget(self.Reset_states_PB, 3, 2)
        grid.addWidget(self.displaycurves_CB, 3, 0)
        grid.addWidget(self.displayVTK_CB, 2, 0)
        label = QLabel('Stim Start')
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(label, 2, 1)
        grid.addWidget(self.StimStart_e, 2, 2)
        grid.addWidget(displaycurve_per_l, 4, 0)
        grid.addWidget(self.displaycurve_per_e, 4, 1)
        self.Simulation_parameters_GB.setContentLayout(grid)
        self.Run_PB.clicked.connect(self.simulate)
        self.UpdateModel_PB.clicked.connect(self.update_Model)
        self.ModifyModel_PB.clicked.connect(self.modify_Model)
        self.ModifyXModel_PB.clicked.connect(self.ModXNMMclicked)
        self.Reset_states_PB.clicked.connect(self.Reset_states_clicked)
        self.displaycurves_CB.stateChanged.connect(self.displaycurves_CB_fonc)
        self.displaycurves_CB.setChecked(True)
        self.displayVTK_CB.stateChanged.connect(self.displayVTK_CB_fonc)
        self.displayVTK_CB.setChecked(True)

        # electrode placement
        self.electrode_placement_GB = Spoiler(title=r"electrode position")
        label_x = QLabel('x y z')
        label_y = QLabel('y')
        label_z = QLabel('z')
        self.electrode_x_e = LineEdit(str(self.electrode_pos[0]))
        self.electrode_y_e = LineEdit(str(self.electrode_pos[1]))
        self.electrode_z_e = LineEdit(str(self.electrode_pos[2]))
        self.Compute_LFP_PB = QPushButton('Point LFP')
        self.Compute_LFP2_PB = QPushButton('Point LFP2')
        self.electrod_disk_CB = QCheckBox('Disk')
        self.electrode_radius_e = LineEdit('50')
        self.electrode_angle1_e = LineEdit('0')
        self.electrode_angle2_e = LineEdit('0')
        self.Compute_LFPDisk_PB = QPushButton('Disk LFP')
        self.Compute_LFPDiskCoated_PB = QPushButton('Coated Disk LFP')
        self.Compute_LFPDiskCoated_type_CB = QComboBox()
        listitem = ['carbon_non-coated_mean , radius = 200 µm',
                    'gold_non-coated_mean , radius = 62.5 µm',
                    'stainless_steel_non-coated_mean, radius = 62.5 µm',
                    'carbon_coated_5s, radius = 200 µm',
                    'carbon_coated_10s, radius = 200 µm',
                    'carbon_coated_50s, radius = 200 µm',
                    'gold_coated_5s, radius = 62.5 µm',
                    'gold_coated_10s, radius = 62.5 µm',
                    'gold_coated_50s, radius = 62.5 µm']
        self.Compute_LFPDiskCoated_type_CB.addItems(listitem)
        self.Temporal_PSD_CB = QCheckBox('Temporal/PSD')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label_x, 0, 0)
        grid.addWidget(self.electrode_x_e, 0, 1)
        grid.addWidget(self.electrode_y_e, 0, 2)
        grid.addWidget(self.electrode_z_e, 0, 3)
        grid.addWidget(self.Compute_LFP_PB, 0, 4)
        grid.addWidget(self.Compute_LFP2_PB, 0, 5)

        grid.addWidget(self.electrod_disk_CB, 1, 0)
        grid.addWidget(QLabel('r \u03B81 \u03B82'), 1, 1)
        grid.addWidget(self.electrode_radius_e, 1, 2)
        grid.addWidget(self.electrode_angle1_e, 1, 3)
        grid.addWidget(self.electrode_angle2_e, 1, 4)
        grid.addWidget(self.Compute_LFPDisk_PB, 1, 5)
        # grid.addWidget(QLabel('material'), 2, 0)
        grid.addWidget(self.Compute_LFPDiskCoated_type_CB, 2, 0, 1, 4)
        grid.addWidget(self.Compute_LFPDiskCoated_PB, 2, 4, 1, 2)
        grid.addWidget(self.Temporal_PSD_CB, 3, 4, 1, 2)

        self.electrode_placement_GB.setContentLayout(grid)
        [x.editingFinished.connect(self.electrode_placement_func) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        [x.editingFinished.connect(self.electrode_placement_func) for x in [self.electrode_radius_e, self.electrode_angle1_e, self.electrode_angle2_e]]
        self.electrod_disk_CB.stateChanged.connect(self.electrode_placement_func)
        self.Compute_LFP_PB.clicked.connect(lambda state, meth=0: self.Compute_LFP_fonc(meth) )
        self.Compute_LFP2_PB.clicked.connect(lambda state, meth=1: self.Compute_LFP_fonc(meth) )
        self.Compute_LFPDisk_PB.clicked.connect(self.Compute_LFPDisk_fonc)
        self.Compute_LFPDiskCoated_PB.clicked.connect(self.Compute_LFPDiskCoated_fonc)
        self.Temporal_PSD_CB.stateChanged.connect(self.UpdateLFP)

        self.somaSize = 15.0 * 10e-3  # 15 micrometres = 15 * 10e-3 mm
        self.dendriteSize = 300.0 * 10e-3  # 800  micrometres = 400 * 10e-3 mm
        self.sigma = 33.0 * 10e-5
        self.gc = 10.e-5
        self.p = 0.15

        # synchro
        self.synchro_GB = Spoiler(title=r"synchronisation algorithm")
        self.synchro_l = QVBoxLayout()
        self.synchro_method_CB = QComboBox()
        list = ['Thresholding', 'AutoCorrelation', 'Nearest delay','ISI-distance','van Rossum distance','Victor Purpura distance' ]
        self.synchro_method_CB.addItems(list)

        self.synchro_l.addWidget(self.synchro_method_CB)
        self.synchro_widget = QWidget()
        self.synchro_l.addWidget(self.synchro_widget)
        self.update_synchro_method()

        self.synchro_GB.setContentLayout(self.synchro_l)
        self.synchro_GB.toggleButton.click()
        self.synchro_method_CB.currentTextChanged.connect(self.update_synchro_method)


        self.Param_VBOX.addWidget(self.tissue_size_GB)
        self.Param_VBOX.addWidget(self.pourcentageCell_GB)
        self.Param_VBOX.addWidget(self.Afferences_GB)
        self.Param_VBOX.addWidget(self.cell_placement_GB)
        self.Param_VBOX.addWidget(self.EField_parameters_GB)
        self.Param_VBOX.addWidget(self.Stimulation_parameters_GB)
        self.Param_VBOX.addWidget(self.InputTh_parameters_GB)
        self.Param_VBOX.addWidget(self.Simulation_parameters_GB)
        self.Param_VBOX.addWidget(self.electrode_placement_GB)
        self.Param_VBOX.addWidget(self.synchro_GB)
        self.Param_VBOX.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding))
        # self.Param_VBOX.addLayout(QVBoxLayout( ))

        self.NewCreate_EField_view = []


    def Nb_Cell_Changed(self,s):
        if s in ["1","23","4","5","6"]:
            somme = 0
            somme += int(self.nbcellsnb1.text())
            somme += int(self.nbcellsnb23.text())
            somme += int(self.nbcellsnb4.text())
            somme += int(self.nbcellsnb5.text())
            somme += int(self.nbcellsnb6.text())
            self.nbcellsnbtotal.setText(str(int(somme)))
            self.update_cellNnumber()
        elif s == 'total':
            Nbcells = self.CC.Nbcells
            ratio = np.array([int(self.nbcellsnb1.text()),int(self.nbcellsnb23.text()),int(self.nbcellsnb4.text()),int(self.nbcellsnb5.text()),int(self.nbcellsnb6.text())]).astype(float)
            ratio /= float(Nbcells)
            new_Nbcells = float(self.nbcellsnbtotal.text())
            self.nbcellsnb1.setText(str(int(new_Nbcells*ratio[0])))
            self.nbcellsnb23.setText(str(int(new_Nbcells*ratio[1])))
            self.nbcellsnb4.setText(str(int(new_Nbcells*ratio[2])))
            self.nbcellsnb5.setText(str(int(new_Nbcells*ratio[3])))
            self.nbcellsnb6.setText(str(int(new_Nbcells*ratio[4])))
            self.update_cellNnumber()

    def update_synchro_method(self):
        self.synchro_l.removeWidget(self.synchro_widget)
        self.synchro_widget.deleteLater()
        self.synchro_widget = None

        if self.synchro_method_CB.currentText() == 'Thresholding':
            self.synchro_widget = QWidget()
            label_Thresholding = QLabel('Threshold (mV)')
            self.Synchro_Thresholding_Threshold_e = LineEdit('0')
            label_Thresholding_perc = QLabel('Area over (%)')
            self.Synchro_Thresholding_Threshold_perc_e = LineEdit('10')
            self.Synchro_Thresholding_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_Thresholding, 0, 0)
            grid.addWidget(self.Synchro_Thresholding_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_Thresholding_PB, 0, 2)
            grid.addWidget(label_Thresholding_perc, 0, 3)
            grid.addWidget(self.Synchro_Thresholding_Threshold_perc_e, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_Thresholding_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'AutoCorrelation':
            self.synchro_widget = QWidget()
            label_AutoCorrelation = QLabel('Threshold (mV)')
            self.Synchro_AutoCorrelation_Threshold_e = LineEdit('0')
            label_AutoCorrelation_perc = QLabel('Area over (%)')
            self.Synchro_AutoCorrelation_Threshold_perc_e = LineEdit('10')
            self.Synchro_AutoCorrelation_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_AutoCorrelation, 0, 0)
            grid.addWidget(self.Synchro_AutoCorrelation_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_AutoCorrelation_PB, 0, 2)
            grid.addWidget(label_AutoCorrelation_perc, 0, 3)
            grid.addWidget(self.Synchro_AutoCorrelation_Threshold_perc_e, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_AutoCorrelation_PB.clicked.connect(self.Synchro_Method_PB_Apply)

        elif self.synchro_method_CB.currentText() == 'Nearest delay':
            self.synchro_widget = QWidget()
            label_NearestDelay = QLabel('Delta')
            self.Synchro_NearestDelay_Threshold_e = LineEdit('50')
            self.Synchro_NearestDelay_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_NearestDelay, 0, 0)
            grid.addWidget(self.Synchro_NearestDelay_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_NearestDelay_PB, 0, 2)
            self.synchro_widget.setLayout(grid)
            self.Synchro_NearestDelay_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'ISI-distance':
            self.synchro_widget = QWidget()
            label_ISI = QLabel('Delta')
            self.Synchro_ISI_Threshold_e = LineEdit('50')
            self.Synchro_ISI_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_ISI, 0, 0)
            grid.addWidget(self.Synchro_ISI_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_ISI_PB, 0, 2)
            self.synchro_widget.setLayout(grid)
            self.Synchro_ISI_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'van Rossum distance':
            self.synchro_widget = QWidget()
            label_vanRossum_Tau = QLabel('Tau (s)')
            label_vanRossum_delta = QLabel('delta (mV)')
            self.Synchro_vanRossum_Tau_e = LineEdit('1')
            self.Synchro_vanRossum_delta_e = LineEdit('50')
            self.Synchro_vanRossum_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_vanRossum_delta, 0, 0)
            grid.addWidget(self.Synchro_vanRossum_delta_e, 0, 1)
            grid.addWidget(label_vanRossum_Tau, 0, 2)
            grid.addWidget(self.Synchro_vanRossum_Tau_e, 0, 3)
            grid.addWidget(self.Synchro_vanRossum_PB, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_vanRossum_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'Victor Purpura distance':
            self.synchro_widget = QWidget()
            label_VictorPurpura_q = QLabel('q (Hz)')
            label_VictorPurpura_delta = QLabel('delta (mV)')
            self.Synchro_VictorPurpura_q_e = LineEdit('1')
            self.Synchro_VictorPurpura_delta_e = LineEdit('50')
            self.Synchro_VictorPurpura_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_VictorPurpura_delta, 0, 0)
            grid.addWidget(self.Synchro_VictorPurpura_delta_e, 0, 1)
            grid.addWidget(label_VictorPurpura_q, 0, 2)
            grid.addWidget(self.Synchro_VictorPurpura_q_e, 0, 3)
            grid.addWidget(self.Synchro_VictorPurpura_PB, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_VictorPurpura_PB.clicked.connect(self.Synchro_Method_PB_Apply)

        self.synchro_l.insertWidget(1, self.synchro_widget)

    def Synchro_Method_PB_Apply(self):
        try:
            self.Sigs_dict
        except:
            return
        if self.synchro_method_CB.currentText() == 'Thresholding':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            threshold = float(self.Synchro_Thresholding_Threshold_e.text())
            threshold_perc = float(self.Synchro_Thresholding_Threshold_perc_e.text()) / 100
            sigbin = array_Sigs > threshold

            Aps = []
            for i in range(sigbin.shape[0]):
                Ap = 0
                for j in range(sigbin.shape[1] - 1):
                    if sigbin[i, j] == False and sigbin[i, j + 1] == True:
                        Ap += 1
                Aps.append(Ap)
            Aps = int(np.sum(np.array(Aps)))

            synchro_sig = np.sum(sigbin, axis=0)
            synchro_sig = synchro_sig / array_Sigs.shape[0]

            sigbin_perc = np.zeros(synchro_sig.shape) * np.nan
            threshold_perc_sig = synchro_sig > threshold_perc
            selection = np.ones(len(threshold_perc_sig), dtype=bool)
            selection[1:] = threshold_perc_sig[1:] != threshold_perc_sig[:-1]
            selection[0] = False
            selection &= threshold_perc_sig != 0
            threshold_perc_NB = np.sum(threshold_perc_sig[selection])

            sigbin_perc[synchro_sig > threshold_perc] = synchro_sig[synchro_sig > threshold_perc]

            sigbin_perc_sum = np.nansum(sigbin_perc - threshold_perc) * (t[1] - t[0])

            chaine = 'Total number of APs on Pyr = ' + str(Aps) + \
                     ', Nb passage over' + self.Synchro_Thresholding_Threshold_perc_e.text() + '% = ' + str(
                threshold_perc_NB) + \
                     ', Area over' + self.Synchro_Thresholding_Threshold_perc_e.text() + '% = ' + str(sigbin_perc_sum)
            self.mascene_LFPViewer.update_synchro_thresholding(synchro_sig, sigbin_perc,
                                                               shiftT=self.CC.tps_start - self.T, titre=chaine)
        elif self.synchro_method_CB.currentText() == 'AutoCorrelation':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            threshold = float(self.Synchro_AutoCorrelation_Threshold_e.text())
            threshold_perc = float(self.Synchro_AutoCorrelation_Threshold_perc_e.text()) / 100
            sigbin = array_Sigs > threshold

            Aps = []
            for i in range(sigbin.shape[0]):
                Ap = 0
                for j in range(sigbin.shape[1] - 1):
                    if sigbin[i, j] == False and sigbin[i, j + 1] == True:
                        Ap += 1
                Aps.append(Ap)
            Aps = int(np.sum(np.array(Aps)))

            synchro_sig = np.sum(sigbin, axis=0)
            synchro_sig = synchro_sig / array_Sigs.shape[0]

            sigbin_perc_autocorrel = signal.correlate(synchro_sig, synchro_sig, mode='full')

            chaine = 'Total number of APs on Pyr = ' + str(Aps)

            self.mascene_LFPViewer.update_synchro_AutoCorrelation(synchro_sig, sigbin_perc_autocorrel, shiftT=self.CC.tps_start - self.T, titre=chaine)

        elif self.synchro_method_CB.currentText() == 'Nearest delay':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            delta = float(self.Synchro_NearestDelay_Threshold_e.text())

            # sigbin = array_Sigs > threshold
            # Aps_centers = []
            # for i in range(sigbin.shape[0]):
            #     Aps = []
            #     newone = False
            #     currentAp = []
            #     for j in range(sigbin.shape[1] - 1):
            #         if sigbin[i, j] == True and sigbin[i, j + 1] == True:
            #             currentAp.append(t[j])
            #             newone = True
            #         else:
            #             if newone == False:
            #                 pass
            #             elif newone == True:
            #                 Aps.append(np.mean(currentAp))
            #                 currentAp = []
            #                 newone = False
            #     Aps_centers.append(Aps)
            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                    # Aps_centers.append(maxi[:, 0].astype(int))
                else:
                    Aps_centers.append(np.array([]))

            # self.mascene_LFPViewer.update_synchro_NearestDelay_scatter(Aps_centers, shiftT=self.CC.tps_start - self.T, titre='')

            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]

            Aps_delays = []
            meanDelay = []
            for i in range(len(Aps_centers)):
                if not Aps_centers[i] == []:
                    meanDelay.append(np.diff(np.array(Aps_centers[i])))

                    delaysline = []
                    for j in range(len(Aps_centers[i])):
                        delays = []
                        for k in range(len(Aps_centers)):
                            if not k == i:
                                if not len(Aps_centers[k]) == 0:
                                    found_delay = find_nearest(Aps_centers[k], Aps_centers[i][j])
                                    delays.append(abs(Aps_centers[i][j] - found_delay))
                        delaysline.append(delays)
                    Aps_delays.append(delaysline)
                else:
                    Aps_delays.append([])

            meanDelay = np.mean(np.concatenate(meanDelay).ravel())
            # self.mascene_LFPViewer.update_synchro_NearestDelay_boxplot(Aps_delays, shiftT=self.CC.tps_start - self.T, titre='')

            Aps_delays_flatten = []
            for i in range(len(Aps_delays)):
                if not Aps_delays[i] == []:
                    Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
                else:
                    Aps_delays_flatten.append([])
            meanDiffDelay = np.mean(np.concatenate(Aps_delays_flatten).ravel())
            # self.mascene_LFPViewer.update_synchro_NearestDelay(Aps_centers,Aps_delays, shiftT=self.CC.tps_start - self.T, titre='Delay Mean = ' + str(meanDelay) + ', Delay Diff Mean = ' + str(meanDiffDelay))

            Aps_gauss_fit = []

            def gaus(x, a, sigma):
                return a * np.exp(-(x) ** 2 / (2 * sigma ** 2))

            # for i in range(len(Aps_delays)):
            #     gauss_fit = []
            #     for j in range(len(Aps_delays[i])):
            #         monAP = Aps_delays[i][j]
            #         maxi = np.max(np.abs(monAP))
            #         x = np.linspace(-maxi, maxi, 1000)
            #         y = np.zeros(x.shape)
            #         for val in np.abs(monAP):
            #             y[np.abs(x) < val] += np.ones(len(y[np.abs(x) < val]))
            #         popt, pcov = curve_fit(gaus, x, y, p0=[1, 1])
            #         gauss_fit.append(popt[0])
            #     Aps_gauss_fit.append(gauss_fit)
            Aps_gauss_fit = []
            # plt.figure()
            for i in range(len(Aps_delays_flatten)):
                if len(Aps_delays_flatten[i]) >0:
                    h = np.histogram(Aps_delays_flatten[i], bins=int(np.max(Aps_delays_flatten[i])), range=(0, int(np.max(Aps_delays_flatten[i]))))
                    x = np.hstack((-h[1][-2:0:-1], h[1][:-1]))
                    y = np.hstack((+h[0][-1:0:-1], h[0][:]))
                    popt, pcov = curve_fit(gaus, x, y, p0=[1, 1])

                    Aps_gauss_fit.append(popt[1])
                    # plt.subplot(10, 10, i + 1)
                    # plt.plot(x, y)
                    # plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
                    # plt.axis([-200, 200, 0, 100])

            meangauss_fit = np.mean(Aps_gauss_fit)
            # meangauss_fit = np.mean(np.concatenate(Aps_gauss_fit).ravel())
            self.mascene_LFPViewer.update_synchro_NearestDelay2(Aps_centers,Aps_delays, Aps_gauss_fit, shiftT=self.CC.tps_start - self.T,
                                                               titre='Delay Mean = ' + str(meanDelay) + ', Delay Diff Mean = ' + str(meanDiffDelay) + ',  gauss fit sigma = ' + str(meangauss_fit))



            #
            # plt.figure()
            # for i in range(54):
            #     monAP = Aps_delays[0][i]
            #     maxi = np.max(np.abs(monAP))
            #     x = np.linspace(-maxi, maxi, 1000)
            #     y = np.zeros(x.shape)
            #     for val in np.abs(monAP):
            #         y[np.abs(x) < val] += np.ones(len(y[np.abs(x) < val]))
            #
            #     def gaus(x, a, x0, sigma):
            #         return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            #
            #     popt, pcov = curve_fit(gaus, x, y, p0=[1, 0, 1])
            #
            #     plt.subplot(8, 6, i + 1)
            #     plt.plot(x, y)
            #     plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
            #     plt.axis([-200, 200, 0, 100])

        elif self.synchro_method_CB.currentText() == 'ISI-distance':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            dt = t[1] - t[0]
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            delta = float(self.Synchro_ISI_Threshold_e.text())

            Aps = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    # Aps.append(t[maxi[:, 0].astype(int)])
                    Aps.append(maxi[:, 0].astype(int))
                else:
                    Aps.append(np.array([]))

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # for i in range(len(Aps)):
            #     plt.plot(t,array_Sigs[i,:]+i*50)
            #     plt.scatter(Aps[i],array_Sigs[i, [int(a/dt) for a in Aps[i]]]+i*50)

            # ISI distance
            array_ISI = np.zeros(array_Sigs.shape) * np.NAN
            for i in range(len(Aps)):
                for j in range(len(Aps[i]) - 1):
                    array_ISI[i, Aps[i][j]:Aps[i][j + 1]] = t[Aps[i][j + 1]] - t[Aps[i][j]]

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # for i in range(len(Aps)):
            #     plt.plot(t,array_ISI[i,:]+i*10)

            array_ratio = np.zeros((array_ISI.shape[0], array_ISI.shape[0], array_ISI.shape[1]))
            array_ratio = self.ISI_ratio(array_ISI, array_ratio)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # for i in range(len(Aps)):
            #     plt.plot(t,array_ratio[i,0,:]+i*10)

            IntegralI = np.zeros((array_ISI.shape[0], array_ISI.shape[0])) * np.NAN
            for i in range(array_ISI.shape[0]):  # xisi
                for j in range(array_ISI.shape[0]):  # y
                    if not j == i:
                        IntegralI[i,j] = np.nansum(np.abs(array_ratio[i,j,:])) / np.sum(np.isnan(array_ratio[i,j,:]))
                        if IntegralI[i,j]==0.:
                            IntegralI[i,j] = np.NAN

            self.mascene_LFPViewer.update_synchro_ISI_distance(IntegralI, titre='ISI_distance')

        elif self.synchro_method_CB.currentText() == 'van Rossum distance':
            delta = float(self.Synchro_vanRossum_delta_e.text())
            Tau = float(self.Synchro_vanRossum_Tau_e.text())

            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)

            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                    # Aps_centers.append(maxi[:, 0].astype(int))
                else:
                    Aps_centers.append(np.array([]))

            if Tau <= 0:
                spike_counts = [len(st) for st in Aps_centers]
                vr_dist = np.sqrt(spike_counts + np.atleast_2d(spike_counts).T)
            elif Tau == np.inf:
                spike_counts = [len(st) for st in Aps_centers]
                vr_dist = np.absolute(spike_counts - np.atleast_2d(spike_counts).T)
            else:
                k_dist = self._summed_dist_matrix(Aps_centers, Tau, True)
                vr_dist = np.empty_like(k_dist)
                for i, j in np.ndindex(k_dist.shape):
                    vr_dist[i, j] = (
                            k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i])
                vr_dist = np.sqrt(vr_dist)

            # f = plt.figure()
            # ax = f.add_subplot(111)
            # current_cmap = matplotlib.cm.get_cmap('viridis')
            # current_cmap.set_bad(color='black')
            # im = ax.imshow(vr_dist, cmap=current_cmap)
            # f.colorbar(im)
            self.mascene_LFPViewer.update_synchro_vanRossum_distance(vr_dist, titre='vanRossum_distance')

        elif self.synchro_method_CB.currentText() == 'Victor Purpura distance':
            delta = float(self.Synchro_VictorPurpura_delta_e.text())
            q = float(self.Synchro_VictorPurpura_q_e.text())

            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)

            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                    # Aps_centers.append(maxi[:, 0].astype(int))
                else:
                    Aps_centers.append(np.array([]))

            if q <= 0:
                num_spikes = np.atleast_2d([len(st) for st in Aps_centers])
                vp_dist = np.absolute(num_spikes.T - num_spikes)
            elif q == np.inf:
                num_spikes = np.atleast_2d([len(st) for st in Aps_centers])
                vp_dist = num_spikes.T + num_spikes
            else:
                shape = (len(Aps_centers), len(Aps_centers))
                vp_dist = np.empty(shape)
                for i in range(shape[0]):
                    for j in range(i, shape[1]):
                        # if len(Aps_centers[i]) == 0 or len(Aps_centers[j]) == 0:
                        #     vp_dist[i, j] = vp_dist[j, i] = np.NAN
                        # else:
                        vp_dist[i, j] = vp_dist[j, i] = self._victor_purpura_dist_for_st_pair_intuitive(Aps_centers[i],
                                                                                                        Aps_centers[j],
                                                                                                        q=q)
            # f = plt.figure()
            # ax = f.add_subplot(111)
            # current_cmap = matplotlib.cm.get_cmap('viridis')
            # current_cmap.set_bad(color='black')
            # im = ax.imshow(vr_dist, cmap=current_cmap)
            # f.colorbar(im)
            self.mascene_LFPViewer.update_synchro_VictorPurpura_distance(vp_dist, titre='VictorPurpura_distance')

    @staticmethod
    def _victor_purpura_dist_for_st_pair_intuitive(train_a, train_b, q=1.0):
        nspk_a = len(train_a)
        nspk_b = len(train_b)
        scr = np.zeros((nspk_a + 1, nspk_b + 1))
        scr[:, 0] = range(0, nspk_a + 1)
        scr[0, :] = range(0, nspk_b + 1)

        if nspk_a > 0 and nspk_b > 0:
            for i in range(1, nspk_a + 1):
                for j in range(1, nspk_b + 1):
                    scr[i, j] = min(scr[i - 1, j] + 1, scr[i, j - 1] + 1)
                    scr[i, j] = min(scr[i, j], scr[i - 1, j - 1]
                                    + np.float64((q * abs(train_a[i - 1] - train_b[j - 1]))))
        return scr[nspk_a, nspk_b]

    def _summed_dist_matrix(self, spiketrains, tau, presorted=False):
        # The algorithm underlying this implementation is described in
        # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
        # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
        # 48-58. We would like to remark that in this paper in formula (9) the
        # left side of the equation should be divided by two.
        #
        # Given N spiketrains with n entries on average the run-time complexity is
        # O(N^2 * n). O(N^2 + N * n) memory will be needed.

        if len(spiketrains) <= 0:
            return np.zeros((0, 0))

        if not presorted:
            spiketrains = [v.copy() for v in spiketrains]
            for v in spiketrains:
                v.sort()

        sizes = np.asarray([v.size for v in spiketrains])
        values = np.empty((len(spiketrains), max(1, sizes.max())))
        values.fill(np.nan)
        for i, v in enumerate(spiketrains):
            if v.size > 0:
                values[i, :v.size] = \
                    (v / tau)

        exp_diffs = np.exp(values[:, :-1] - values[:, 1:])
        markage = np.zeros(values.shape)
        for u in range(len(spiketrains)):
            markage[u, 0] = 0
            for i in range(sizes[u] - 1):
                markage[u, i + 1] = (markage[u, i] + 1.0) * exp_diffs[u, i]

        # Same spiketrain terms
        D = np.empty((len(spiketrains), len(spiketrains)))
        D[np.diag_indices_from(D)] = sizes + 2.0 * np.sum(markage, axis=1)

        # Cross spiketrain terms
        for u in range(D.shape[0]):
            all_ks = np.searchsorted(values[u], values, 'left') - 1
            for v in range(u):
                js = np.searchsorted(values[v], values[u], 'right') - 1
                ks = all_ks[v]
                slice_j = np.s_[np.searchsorted(js, 0):sizes[u]]
                slice_k = np.s_[np.searchsorted(ks, 0):sizes[v]]
                D[u, v] = np.sum(
                    np.exp(values[v][js[slice_j]] - values[u][slice_j]) *
                    (1.0 + markage[v][js[slice_j]]))
                D[u, v] += np.sum(
                    np.exp(values[u][ks[slice_k]] - values[v][slice_k]) *
                    (1.0 + markage[u][ks[slice_k]]))
                D[v, u] = D[u, v]

        return D

    @staticmethod
    @njit
    def ISI_ratio(array_ISI, array_ratio):
        for i in range(array_ISI.shape[0]):  # xisi
            for j in range(array_ISI.shape[0]):  # y
                if not j > i:
                    for tt in range(array_ISI.shape[1]):
                        if array_ISI[i, tt] <= array_ISI[j, tt]:
                            array_ratio[i, j, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                            array_ratio[j, i, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                        else:
                            array_ratio[i, j, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
                            array_ratio[j, i, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
        return array_ratio

    def get_new_List_C(self):
        for l in range(len(self.List_C)):
            for c in range(len(self.List_C[0])):
                self.List_C[l, c] = float(self.List_C_e[l * len(self.List_C[0]) + c].text())
                # if c == 4:
                #     self.List_C[l, c] = float(self.List_C_e[l*len(self.List_C[0])+c].text() )
                # else:
                #     self.List_C[l, c] = int(float(self.List_C_e[l*len(self.List_C[0])+c].text() ))

    def get_new_List_Var(self):
        for l in range(len(self.List_Var)):
            for c in range(len(self.List_Var)):
                self.List_Var[l, c] = float(self.List_Var_e[l * len(self.List_Var) + c].text())

    pyqtSlot(int)
    def PlaceCell_msg(self, cellnb):
        if cellnb == -2:
            self.msg = msg_wait("Cell Placement in progress\n"+'0/'+str(np.sum(self.CC.Layer_nbCells))+"\nPlease wait.")
            self.msg.setStandardButtons(QMessageBox.Cancel)
            self.PlaceCell_msg_cnacel = False
            self.parent.processEvents()
        elif cellnb == -1:
            self.PlaceCell_msg_cnacel = False
        else:
            self.msg.setText("Cell Placement in progress\n"+str(cellnb)+'/'+str(np.sum(self.CC.Layer_nbCells))+"\nPlease wait.")
            self.parent.processEvents()

    def connectivityCell_func(self):
        seed = int(self.seed_place.text())
        self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th,
                                                       self.CellPosition)
        self.update_model_with_same_param()
        self.masceneCM.update()

    def EField_Conv_Fun(self):
        fileName = QFileDialog.getOpenFileName(self, "E-Field text file", "", "txt (*.txt)")
        if fileName[0] == '':
            return
        if fileName[1] == "txt (*.txt)":
            try:
                n = np.loadtxt(fileName[0], comments='%')

                x = n[:, 0] * 1000
                xmin = np.round(np.min(x) * 1000000) / 1000000
                xmax = np.round(np.max(x) * 1000000) / 1000000
                xsteps = len(set(x))
                xrange = abs(xmax - xmin)
                y = n[:, 1] * 1000
                ymin = np.round(np.min(y) * 1000000) / 1000000
                ymax = np.round(np.max(y) * 1000000) / 1000000
                ysteps = len(set(y))
                yrange = abs(ymax - ymin)
                z = n[:, 2] * 1000
                zmin = np.round(np.min(z) * 1000000) / 1000000
                zmax = np.round(np.max(z) * 1000000) / 1000000
                zsteps = len(set(z))
                zrange = abs(zmax - zmin)

                E = n[:, 4:] * 1000
                Er = np.zeros((xsteps, ysteps, zsteps, 3), dtype=np.float32)
                i = 0
                for kx in range(xsteps):
                    for ky in range(ysteps):
                        for kz in range(zsteps):
                            Er[kx, ky, kz, :] = E[i, :]
                            i += 1

                mdic = {"Er": Er,
                        "xmin": xmin,
                        "xmax": xmax,
                        "xsteps": xsteps,
                        "xrange": xrange,
                        "ymin": ymin,
                        "ymax": ymax,
                        "ysteps": ysteps,
                        "yrange": yrange,
                        "zmin": zmin,
                        "zmax": zmax,
                        "zsteps": zsteps,
                        "zrange": zrange, }

                fileName = QFileDialog.getSaveFileName(caption='Save parameters',
                                                       directory=os.path.splitext(fileName[0])[0] + '.mat',
                                                       filter="mat (*.mat)")
                if (fileName[0] == ''):
                    return
                if os.path.splitext(fileName[0])[1] == '':
                    fileName = (fileName[0] + '.mat', fileName[1])
                if fileName[1] == "mat (*.mat)":
                    scipy.io.savemat(fileName[0], mdic)
            except:
                msg_cri(s='Something went wrong')

    def get_Efield_path(self, ):
        extension = "mat"
        fileName = QFileDialog.getOpenFileName(caption='Load mesh file', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            try:
                self.EField_file_TE.setText(fileName[0])
                self.EField_Load_file()
            except:
                self.EField_file_TE.setText('')
                msg_cri('VTK file cannot be loaded.\nPlease, load another one.')

    def EField_Load_file(self):
        mat = scipy.io.loadmat(self.EField_file_TE.text())
        self.EFieldFile = {}
        for key in [key for key in mat.keys() if not '__' in key]:
            self.EFieldFile[key] = mat[key]  # [0][0]
            if self.EFieldFile[key].shape == (1, 1):
                self.EFieldFile[key] = self.EFieldFile[key][0][0]

        self.EField_XYZ_comp()
        self.EField_Rot_XY_comp()
        self.EFieldFile['Er'] = self.EFieldFile['Er'].astype(np.float64)

        self.EField_OnOff_CB.setEnabled(True)

    def EField_XYZ_comp(self):
        xT = float(self.EField_TranslationX_LE.text())
        yT = float(self.EField_TranslationY_LE.text())
        zT = float(self.EField_TranslationZ_LE.text())

        self.EFieldFile['x'] = np.linspace(-self.EFieldFile['xrange']/2+xT,self.EFieldFile['xrange']/2+xT, self.EFieldFile['xsteps'])
        self.EFieldFile['y'] = np.linspace(-self.EFieldFile['yrange']/2+yT,self.EFieldFile['yrange']/2+yT, self.EFieldFile['ysteps'])
        self.EFieldFile['z'] = np.linspace(0+zT,self.EFieldFile['zrange']+zT, self.EFieldFile['zsteps'])

    def EField_Rot_XY_comp(self):
        self.EField_theta = float(self.EField_RotationXY_LE.text())
        # sintheta = np.sin(theta)
        # costheta = np.cos(theta)
        # for ix in range(self.EFieldFile['Er'].shape[0]):
        #     x = self.EFieldFile['x'][ix]
        #     for iy in range(self.EFieldFile['Er'].shape[1]):
        #         y = self.EFieldFile['y'][iy]
        #         for iz in range(self.EFieldFile['Er'].shape[2]):
        #             x2 = self.EFieldFile['Er'][ix,iy,iz,0] + x
        #             y2 = self.EFieldFile['Er'][ix,iy,iz,1] + y
        #             xp = x2 * costheta - y2 * sintheta
        #             yp = x2 * sintheta + y2 * costheta
        #             self.EFieldFile['Er'][ix,iy,iz,0] = xp
        #             self.EFieldFile['Er'][ix,iy,iz,1] = yp
        # x = self.EFieldFile['x']
        # b = np.repeat(x[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # b2 = np.repeat(b[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        # y = self.EFieldFile['y']
        # c = np.repeat(y[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # c2 = np.repeat(c[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        #
        # x2 = self.EFieldFile['Er'][:,:,:,0] + b2
        # y2 = self.EFieldFile['Er'][:,:,:,1] + c2
        # xp = x2 * costheta - y2 * sintheta
        # yp = x2 * sintheta + y2 * costheta
        #
        # self.EFieldFile['Er'][:,:,:,0] = xp
        # self.EFieldFile['Er'][:,:,:,1] = yp

        # x = self.EFieldFile['x']
        # y = self.EFieldFile['y']
        # xp = x * np.cos(theta) - 0 * np.sin(theta)
        # yp = 0 * np.sin(theta) + y * np.cos(theta)
        # self.EFieldFile['x'] = xp
        # self.EFieldFile['y'] = yp


        # x = self.EFieldFile['x']
        # b = np.repeat(x[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # b2 = np.repeat(b[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        # y = self.EFieldFile['y']
        # c = np.repeat(y[:, np.newaxis], self.EFieldFile['Er'].shape[1], axis=1)
        # c2 = np.repeat(c[:, :, np.newaxis], self.EFieldFile['Er'].shape[2], axis=2)
        #
        # self.EFieldFile['Er'][:,:,:,0] -= b2
        # self.EFieldFile['Er'][:,:,:,1] -= c2

    def EField_Const_fun(self):
        Er = np.array([float(self.EField_Const_Ex.text()) ,float(self.EField_Const_Ey.text()) ,float(self.EField_Const_Ez.text())])
        self.EField_Const = {"Er": Er}

    def EField_Display_Fun(self):
            if self.EField_Const_RB.isChecked():
                self.EField_Const_fun()
                self.EField = self.EField_Const
                self.NewCreate_EField_view.append(Graph_EField_VTK(self,fromfile = False))
                self.NewCreate_EField_view[-1].show()

            elif self.EField_File_RB.isChecked():
                self.EField_Load_file()
                self.EField = self.EFieldFile
                self.NewCreate_EField_view.append(Graph_EField_VTK(self))
                self.NewCreate_EField_view[-1].show()
        # except:
        #     msg_cri("Display not available")

    def PlaceCell_func(self):
        seed = int(self.seed_place.text())
        placement = self.cell_placement_CB.currentText()
        print('placement')
        self.CC.updateCell.something_happened.connect(self.PlaceCell_msg)
        self.CellPosition = CreateColumn.PlaceCell_func(self.CC.L,self.CC.Layer_d,self.CC.D,self.CC.Layer_nbCells,placement,seed = seed)
        self.CellPosition = np.array(self.CellPosition)
        print('Create_Connectivity_Matrices')
        t0 = time.time()
        self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th,
                                                       self.CellPosition)
        print(time.time() - t0)
        self.CC.updateCell.something_happened.disconnect(self.PlaceCell_msg)


        self.createCells = True

        # # self.Create_Connectivity_Matrices()
        # # print(self.ConnectivityMatrix)
        # print('List_Names')
        self.List_Names = []
        self.List_Colors = []
        self.List_Neurone_type = []

        colors_PYR = ['#000000', '#9370db', '#9400d3', '#8b008b', '#4b0082']
        for i in range(5):
            s = ''
            if i == 0:
                s = s + ("1_")
            elif i == 1:
                s = s + ("23_")
            elif i == 2:
                s = s + ("4_")
            elif i == 3:
                s = s + ("5_")
            elif i == 4:
                s = s + ("6_")
            Names = []
            Colors = []
            Neurone_type = []
            for j in range(self.CC.C.NB_PYR[i]):
                Names.append(s + 'PYR_' + str(j))
                Colors.append(colors_PYR[i])
                Neurone_type.append(1)

            for j in range(self.CC.C.NB_PV[i]):
                Names.append(s + 'PV_' + str(j))
                Colors.append('#228B22')
                Neurone_type.append(2)

            for j in range(self.CC.C.NB_SST[i]):
                Names.append(s + 'SST_' + str(j))
                Colors.append('#0000cd')
                Neurone_type.append(3)

            for j in range(self.CC.C.NB_VIP[i]):
                Names.append(s + 'VIP_' + str(j))
                Colors.append('#cd5c5c')
                Neurone_type.append(4)

            for j in range(self.CC.C.NB_RLN[i]):
                Names.append(s + 'RLN_' + str(j))
                Colors.append('#FFA500')
                Neurone_type.append(5)

            self.List_Colors.append(Colors)
            self.List_Names.append(Names)
            self.List_Neurone_type.append(Neurone_type)

        self.update_Model()
        self.electrode_placement_func()
        self.masceneCM.update()
        self.update_graph()

    def scalesize_func(self):
        exPopup = Rescalesize(self, x=self.x_e.text(), y=self.y_e.text(), z=self.z_e.text())
        # exPopup.show()
        if exPopup.exec_() == QDialog.Accepted:
            # exPopup.editfromlabel()
            # self.parent.Graph_Items[cellId] = exPopup.item
            print('Accepted')
            xs = float(exPopup.xs_e.text())
            ys = float(exPopup.ys_e.text())
            zs = float(exPopup.zs_e.text())
            self.x_e.setText(str(float(self.x_e.text()) * xs))
            self.y_e.setText(str(float(self.y_e.text()) * ys))
            self.z_e.setText(str(float(self.z_e.text()) * zs))

            self.CellPosition = self.CellPosition * np.array([xs, ys, zs])

            self.update_graph()


        else:
            print('Cancelled')
        exPopup.deleteLater()

    def update_graph(self):
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()
            self.Graph_viewer.set_center()
            self.parent.processEvents()

    def electrode_placement_func(self):
        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        self.electrode_disk = [self.electrod_disk_CB.isChecked(),
                               -float(self.electrode_radius_e.text()),
                               -float(self.electrode_angle1_e.text()),
                               float(self.electrode_angle2_e.text())]
        self.update_graph()

    def bruitGaussien(self, s, m):
        return np.random.normal(m, s)

    def Compute_synaptic_connections_sparse(self):
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_Dend_AMPA = []
        self.PreSynaptic_Soma_Dend_AMPA_not = []
        self.PreSynaptic_Soma_Dend_GABA = []
        self.PreSynaptic_Soma_Dend_GABA_not = []
        for i, c in enumerate(self.List_Neurone_type):
            # convect = self.ConnectivityMatrix[i, :]
            # convect = self.ConnectivityMatrix.getrow(i).todense()
            # convect = np.where(convect == 1)[0]
            convect = self.ConnectivityMatrix[i]
            convect_AMPA = []
            convect_GABA = []
            convect_Soma_Dend_AMPA = []
            convect_Soma_Dend_GABA = []
            for k in convect:
                if self.List_Neurone_type[k] in [1, 2]:  # si from CA1 ou CA3
                    convect_AMPA.append(k)
                    if c in [3, 4, 5]:  # interneurones
                        convect_Soma_Dend_AMPA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # si from CA1, CA3, SOM, ou BIS
                            convect_Soma_Dend_AMPA.append(0)
                        else:  # si from BAS
                            convect_Soma_Dend_AMPA.append(1)

                else:  # from interneurone
                    convect_GABA.append(k)
                    if c in [3, 4, 5]:  # interneurones
                        convect_Soma_Dend_GABA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # si from CA1, CA3, SOM, ou BIS
                            convect_Soma_Dend_GABA.append(0)
                        else:  # si from BAS
                            convect_Soma_Dend_GABA.append(1)

            self.PreSynaptic_Cell_AMPA.append(convect_AMPA)
            self.PreSynaptic_Cell_GABA.append(convect_GABA)
            self.PreSynaptic_Soma_Dend_AMPA.append(np.array(convect_Soma_Dend_AMPA))
            self.PreSynaptic_Soma_Dend_AMPA_not.append(np.abs(np.array(convect_Soma_Dend_AMPA) - 1))
            self.PreSynaptic_Soma_Dend_GABA.append(np.array(convect_Soma_Dend_GABA))
            self.PreSynaptic_Soma_Dend_GABA_not.append(np.abs(np.array(convect_Soma_Dend_GABA) - 1))

    def update_cellNnumber(self):

        Layer_nbCells = np.array( [int(self.nbcellsnb1.text()), int(self.nbcellsnb23.text()),  int(self.nbcellsnb4.text()),  int(self.nbcellsnb5.text()), int(self.nbcellsnb6.text())])

        PYRpercent = []
        for e in self.List_PYRpercent:
            PYRpercent.append(float(e.text()))
        PYRpercent = np.array(PYRpercent)

        PVpercent = []
        for e in self.List_PVpercent:
            PVpercent.append(float(e.text()))
        PVpercent = np.array(PVpercent)

        SSTpercent = []
        for e in self.List_SSTpercent:
            SSTpercent.append(float(e.text()))
        SSTpercent = np.array(SSTpercent)

        VIPpercent = []
        for e in self.List_VIPpercent:
            VIPpercent.append(float(e.text()))
        VIPpercent = np.array(VIPpercent)

        RLNpercent = []
        for e in self.List_RLNpercent:
            RLNpercent.append(float(e.text()))
        RLNpercent = np.array(RLNpercent)

        self.CC.update_cellNumber(Layer_nbCells,
                                 PYRpercent,
                                 PVpercent,
                                 SSTpercent,
                                 VIPpercent,
                                 RLNpercent)
        self.CC.update_inputNB()
        self.nbcellsnbtotal.setText(str(int(np.sum(self.CC.Layer_nbCells))))

    def update_connections(self):
        Norm = Afferences_ManagmentTable(self)
        if Norm.exec_():
            self.CC.update_connections(self.CC.Afferences, fixed = not self.r0.isChecked())
            pass

    def See_connections(self):
        Norm = Connection_ManagmentTable(self)
        if Norm.exec_():
            pass

    def update_connections_per_fixed(self):
        self.CC.update_connections(self.CC.Afferences, fixed = not self.r0.isChecked())

    def take_cell_number(self):
        if not int(self.Nb_of_PYR_l.text()) == self.Nb_of_PYR:
            self.createCells = True
        self.Nb_of_PYR = int(self.Nb_of_PYR_l.text())
        if not int(self.Nb_of_BAS_l.text()) == self.Nb_of_BAS:
            self.createCells = True
        self.Nb_of_BAS = int(self.Nb_of_BAS_l.text())
        if not int(self.Nb_of_SOM_l.text()) == self.Nb_of_SOM:
            self.createCells = True
        self.Nb_of_SOM = int(self.Nb_of_SOM_l.text())
        if not int(self.Nb_of_BIS_l.text()) == self.Nb_of_BIS:
            self.createCells = True
        self.Nb_of_BIS = int(self.Nb_of_BIS_l.text())
        if not int(self.Nb_of_PYR_Stimulated_l.text()) == self.Nb_of_PYR_Stimulated:
            self.createCells = True
        self.Nb_of_PYR_Stimulated = int(self.Nb_of_PYR_Stimulated_l.text())

        self.Nb_of_PYR_BAS_SOM_BIS_sum = [self.Nb_of_PYR,
                                          self.Nb_of_PYR + self.Nb_of_BAS,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM + self.Nb_of_BIS]

        self.List_Neurone_type = [2] *  self.Nb_of_PYR_Stimulated + [1] * (self.Nb_of_PYR-self.Nb_of_PYR_Stimulated)  + [3] * self.Nb_of_BAS + [4] * self.Nb_of_SOM + [5] * self.Nb_of_BIS
        self.List_PYR_Stimulated = [i for i in range(self.Nb_of_PYR_Stimulated)]

    def update_percent_sum(self):
        sum = float(self.Per_PYR.text())+float(self.Per_BAS.text())+float(self.Per_SOM.text())+float(self.Per_BIS.text())
        self.labelSUM2.setText(str(sum))
        if sum == 100:
            self.labelSUM2.setStyleSheet("QLabel { background-color : none}")
        else:
            self.labelSUM2.setStyleSheet("QLabel { background-color : red}")

    def set_tissue_func(self):

        D = float(self.D_e.text())
        L = float(self.L_e.text())
        C = float(self.C_e.text())

        L_d1 = float(self.Layer_d1_l.text())
        L_d23 = float(self.Layer_d23_l.text())
        L_d4 = float(self.Layer_d4_l.text())
        L_d5 = float(self.Layer_d5_l.text())
        L_d6 = float(self.Layer_d6_l.text())

        self.CC.updateTissue(D, L, C, np.array([L_d1, L_d23, L_d4, L_d5, L_d6]))

    def Generate_Stim_Signals(self):

        self.nbOfSamplesStim = int(float(self.StimDuration_e.text()) / self.dt)
        self.i_inj = float(self.i_inj_e.text())
        self.tau = float(self.tau_e.text())
        self.nbStim = int(self.nbStim_e.text())
        self.varianceStim = int(float(self.varianceStim_e.text()) / self.dt)
        self.seed = int(self.seed_e.text())

        nb_Stim_Signals = len(self.List_PYR_Stimulated)

        self.dt = 1 / float(self.Fs_e.text())
        self.T = float(self.SimDuration_e.text())
        self.nbEch = int(self.T / self.dt)

        Stim_Signals_in = np.zeros((nb_Stim_Signals, self.nbEch))
        Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))

        Generate_Stim_Signals(Stim_Signals_in, self.seed, self.nbOfSamplesStim, self.i_inj, self.tau, self.nbStim, self.varianceStim, nb_Stim_Signals, self.nbEch, self.dt, Stim_Signals_out)

        self.Stim_Signals = Stim_Signals_out

        if not self.seed == 0:
            np.random.seed(self.seed)
        # else:
        #     np.random.seed()

    def update_model_with_same_param(self):
        List_Neurone_param = copy.deepcopy(self.CC.List_Neurone_param)
        self.CC.create_cells()
        self.createCells = False
        self.Reset_states_clicked()
        self.CC.List_Neurone_param = List_Neurone_param
        self.CC.Update_param_model()

    def update_Model(self):

        # self.Compute_synaptic_connections_sparse()
        self.CC.create_cells()
        self.createCells = False
        self.Reset_states_clicked()

    def modify_Model(self):
        try:
            if self.NewModify1NMM == None and self.NewModifyXNMM == None :
                self.NewModify1NMM = Modify_1_NMM(Dict_Param=self.CC.List_Neurone_param, List_Names=self.List_Names, List_Color=self.List_Colors)
                self.NewModify1NMM.Mod_OBJ.connect(self.ApplyMod1NMM)
                self.NewModify1NMM.Close_OBJ.connect(self.closeMod1NMM)
                self.NewModify1NMM.show()
            else:
                self.put_window_on_top()
        except:
            pass

    @pyqtSlot(list, list, list)
    def ApplyMod1NMM(self, Dict_Param, popName, popColor):
        for idx, p in enumerate(Dict_Param):
            for idx_v,key in enumerate(Dict_Param[idx].keys()):
                self.CC.List_Neurone_param[idx][key]=Dict_Param[idx][key]
        # self.CC.List_Neurone_param = Dict_Param
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def closeMod1NMM(self, ):
        self.NewModify1NMM.deleteLater()
        self.NewModify1NMM.destroyed.connect(self.on_destroyed_NewModify1NMM)
        # self.NewModify1NMM = None

    @pyqtSlot('QObject*')
    def on_destroyed_NewModify1NMM(self, o):
        self.NewModify1NMM = None

    def ModXNMMclicked(self, ):
        try:
            if self.NewModify1NMM == None and self.NewModifyXNMM == None:
                self.NewModifyXNMM = Modify_X_NMM(parent= self,List_Neurone_type=self.List_Neurone_type,Dict_Param=self.CC.List_Neurone_param, List_Names=self.List_Names, List_Color=self.List_Colors, initcell = 0, CellPosition=self.CellPosition)
                self.NewModifyXNMM.Mod_OBJ.connect(self.ApplyModXNMM)
                self.NewModifyXNMM.Close_OBJ.connect(self.close_ModXNMM)
                self.NewModifyXNMM.updateVTK_OBJ.connect(self.update_VTKgraph_from_ModXNMM)
                self.NewModifyXNMM.show()
            else:
                self.put_window_on_top()

        except:
            pass

    @pyqtSlot(list, list, list)
    def ApplyModXNMM(self, Dict_Param, popName, popColor):
        for idx, p in enumerate(Dict_Param):
            for idx2, p2 in enumerate(p):
                for idx_v,key in enumerate(p2.keys()):
                    self.CC.List_Neurone_param[idx][idx2][key]=Dict_Param[idx][idx2][key]
        # self.CC.List_Neurone_param = Dict_Param
        self.CC.Update_param_model()
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def close_ModXNMM(self, ):
        self.NewModifyXNMM.deleteLater()
        self.NewModifyXNMM.destroyed.connect(self.on_destroyed_NewModifyXNMM)
        # pass
        # self.NewModifyXNMM = None

    @pyqtSlot('QObject*')
    def on_destroyed_NewModifyXNMM(self, o):
        self.NewModifyXNMM = None
        print(self.NewModifyXNMM)

    @pyqtSlot(list,)
    def update_VTKgraph_from_ModXNMM(self,selectedcell):
        self.Graph_viewer.selected_cells = selectedcell
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()

    def update_ModXNMM_from_VTKgraph(self,seleced_cell):
        if not self.NewModifyXNMM == None:
            self.NewModifyXNMM.PopNumber.setCurrentIndex(seleced_cell)

    def put_window_on_top(self):
        self.parent.processEvents()
        if not self.NewModify1NMM == None:
            self.NewModify1NMM.activateWindow()
        elif not self.NewModifyXNMM == None:
            self.NewModifyXNMM.activateWindow()

    def update_samples(self):
        try:
            self.CC.tps_start = 0.
            self.CC.Reset_states()
        except:
            pass

    def Reset_states_clicked(self):
        try:
            self.CC.tps_start = 0.
            self.CC.Reset_states()
        except:
            pass

    def simulate(self):
        if self.CC.ImReady == False:
            msg_cri('The model is not ready to Simulate')
            return
        if not self.Colonne_cortical_Thread.isRunning():
            t0 = time.time()
            self.msg = msg_wait("Computation in progress\nPlease wait.")
            self.msg.setStandardButtons(QMessageBox.Cancel)
            # self.msg.buttonClicked.connect(self.Cancelpressed)
            self.parent.processEvents()

            Fs = int(self.Fs_e.text())
            self.dt = 1 / Fs
            self.T = float(self.SimDuration_e.text())
            self.nbEch = int(self.T / self.dt)
            self.CC.nbEch = self.nbEch
            self.CC.dt = self.dt
            # self.CC.update_samples(Fs,self.T)
            self.Reset_states_clicked()
            S_nbOfSamplesStim =  float(self.StimDuration_e.text())
            S_i_inj = float(self.i_inj_e.text())
            S_tau = float(self.tau_e.text())
            S_nbStim = int(self.nbStim_e.text())
            S_varianceStim = float(self.varianceStim_e.text())
            S_seed = int(self.seed_e.text())
            S_StimStart = float(self.StimStart_e.text())

            # if not S_seed == 0:
            #     np.random.seed(S_seed)
            # else:
            #     np.random.seed()
            self.CC.set_seed(S_seed)
            self.CC.Generate_Stims( I_inj=S_i_inj, tau=S_tau, stimDur=S_nbOfSamplesStim, nbstim=S_nbStim, varstim=S_varianceStim,StimStart=S_StimStart)

            TH_nbOfSamplesStim = float(self.TH_StimDuration_e.text())
            TH_i_inj = float(self.TH_i_inj_e.text())
            TH_tau = float(self.TH_tau_e.text())
            TH_nbStim = int(self.TH_nbStim_e.text())
            TH_deltamin = float(self.TH_deltamin_e.text())
            TH_delta = float(self.TH_delta_e.text())


            self.CC.Generate_input( I_inj=TH_i_inj, tau=TH_tau, stimDur=TH_nbOfSamplesStim, nbstim=TH_nbStim, deltamin=TH_deltamin, delta=TH_delta)



            if self.EField_OnOff_CB.isChecked():
                if self.EField_Const_RB.isChecked():
                    self.EField_Const_fun()
                    self.EField = self.EField_Const
                elif self.EField_File_RB.isChecked():
                    self.EField = self.EFieldFile
                self.CC.Generate_EField(self.EField,np.float(self.EField_OnOff_CB.isChecked()),bool(self.EField_Const_RB.isChecked()))
            else:
                self.EField = {'Er': np.array([0.,0.,0.])}
                self.CC.Generate_EField(self.EField,0.,True)


            A = float(self.EField_StimSig_A_LE.text())
            F = float(self.EField_StimSig_F_LE.text())
            stimOnOff = int(self.EField_OnOff_CB.isChecked())
            Start = float(self.EField_Start_LE.text())
            Length = float(self.EField_Length_LE.text())
            self.CC.Generate_EField_Stim(self.EField_StimSig_CB.currentText(),A,F, stimOnOff,Start = Start, Length = Length)

            self.t0 = time.time()

            self.t, self.pyrVs, self.pyrVd,self.pyrVa, self.PV_Vs, self.SST_Vs, self.VIP_Vs, self.RLN_Vs, self.DPYR_Vs, self.Th_Vs, self.pyrPPSE,self.pyrPPSI,self.pyrPPSI_s,self.pyrPPSI_a = self.CC.runSim()

            self.msg = msg_wait("Computation finished\nResults are currently displayed.\nPlease wait.")

            self.flatindex = []
            for i in range(len(self.CellPosition)):
                for j in range(len(self.CellPosition[i])):
                    self.flatindex.append([i, j])

            self.Color = []
            self.Sigs_dict = {}
            self.Sigs_dict["t"] = self.t + self.CC.tps_start
            self.CC.tps_start = self.t[-1]
            # count = 0
            # for i in range(self.pyrVs.shape[0]):
            #     if i in self.List_PYR_Stimulated:
            #         self.Color.append(self.List_Colors[count])
            #         self.Sigs_dict["Pyr_" + str(i) + 'Stim'] = self.pyrVs[i, :]
            #     else:
            #         self.Color.append(self.List_Colors[count])
            #         self.Sigs_dict["Pyr_" + str(i)] = self.pyrVs[i, :]
            #     count += 1
            #
            # for i in range(self.basketVs.shape[0]):
            #     self.Color.append(self.List_Colors[count])
            #     self.Sigs_dict["BAS_" + str(i)] = self.basketVs[i, :]
            #     count += 1
            #
            # for i in range(self.olmVs.shape[0]):
            #     self.Color.append(self.List_Colors[count])
            #     self.Sigs_dict["SOM_" + str(i)] = self.olmVs[i, :]
            #     count += 1
            #
            # for i in range(self.BisVs.shape[0]):
            #     self.Color.append(self.List_Colors[count])
            #     self.Sigs_dict["BIS_" + str(i)] = self.BisVs[i, :]
            #     count += 1

            # count = 0
            # for i in range(self.pyrVs.shape[0]):
            #     self.Color.append(self.List_Colors[self.flatindex[count][0]][self.flatindex[count][1]])
            #     self.Sigs_dict[self.List_Names[self.flatindex[count][0]][self.flatindex[count][1]]] = self.pyrVs[i, :]
            #     count += 1
            #
            # for i in range(self.PV_Vs.shape[0]):
            #     self.Color.append(self.List_Colors[self.flatindex[count][0]][self.flatindex[count][1]])
            #     self.Sigs_dict[self.List_Names[self.flatindex[count][0]][self.flatindex[count][1]]] = self.PV_Vs[i, :]
            #     count += 1
            #
            # for i in range(self.SST_Vs.shape[0]):
            #     self.Color.append(self.List_Colors[self.flatindex[count][0]][self.flatindex[count][1]])
            #     self.Sigs_dict[self.List_Names[self.flatindex[count][0]][self.flatindex[count][1]]] = self.SST_Vs[i, :]
            #     count += 1
            #
            # for i in range(self.VIP_Vs.shape[0]):
            #     self.Color.append(self.List_Colors[self.flatindex[count][0]][self.flatindex[count][1]])
            #     self.Sigs_dict[self.List_Names[self.flatindex[count][0]][self.flatindex[count][1]]] = self.VIP_Vs[i, :]
            #     count += 1
            #
            # for i in range(self.RLN_Vs.shape[0]):
            #     self.Color.append(self.List_Colors[self.flatindex[count][0]][self.flatindex[count][1]])
            #     self.Sigs_dict[self.List_Names[self.flatindex[count][0]][self.flatindex[count][1]]] = self.RLN_Vs[i, :]
            #     count += 1

            nb_pyr = 0
            nb_pv = 0
            nb_sst = 0
            nb_vip = 0
            nb_rln = 0
            count = 0
            for i in range(len(self.flatindex)):
                l = self.flatindex[i][0]
                n = self.flatindex[i][1]
                if self.List_Neurone_type[l][n] == 1:
                    self.Color.append(self.List_Colors[l][n])
                    self.Sigs_dict[self.List_Names[l][n]] = self.pyrVs[nb_pyr, :]
                    nb_pyr += 1
                elif self.List_Neurone_type[l][n] == 2:
                    self.Color.append(self.List_Colors[l][n])
                    self.Sigs_dict[self.List_Names[l][n]] = self.PV_Vs[nb_pv, :]
                    nb_pv += 1
                elif self.List_Neurone_type[l][n] == 3:
                    self.Color.append(self.List_Colors[l][n])
                    self.Sigs_dict[self.List_Names[l][n]] = self.SST_Vs[nb_sst, :]
                    nb_sst += 1
                elif self.List_Neurone_type[l][n] == 4:
                    self.Color.append(self.List_Colors[l][n])
                    self.Sigs_dict[self.List_Names[l][n]] = self.VIP_Vs[nb_vip, :]
                    nb_vip += 1
                elif self.List_Neurone_type[l][n] == 5:
                    self.Color.append(self.List_Colors[l][n])
                    self.Sigs_dict[self.List_Names[l][n]] = self.RLN_Vs[nb_rln, :]
                    nb_rln += 1
                count += 1

            for i in range(self.DPYR_Vs.shape[0]):
                self.Color.append("#000000")
                self.Sigs_dict['DPYR' + str(i)] = self.DPYR_Vs[i, :]
                count += 1
            for i in range(self.Th_Vs.shape[0]):
                self.Color.append("#999999")
                self.Sigs_dict['Th' + str(i)] = self.Th_Vs[i, :]
                count += 1
            if self.displaycurves_CB.isChecked():
                print('update draw')
                # self.mascene_EEGViewer.setWindowSizeWithoutRedraw(int(self.T))
                self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color, percentage=float(self.displaycurve_per_e.text()))
                print('finish update draw')

            print(str(datetime.timedelta(seconds=int(time.time()-t0))))
            self.msg.close()
            self.parent.processEvents()

    def displaycurves_CB_fonc(self):
        if hasattr(self, 'Sigs_dict'):
            if self.displaycurves_CB.isChecked():
                # self.mascene_EEGViewer.setWindowSizeWithoutRedraw(int(self.T))
                self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color, percentage=float(self.displaycurve_per_e.text()))

    def displayVTK_CB_fonc(self):
        if self.displayVTK_CB.isChecked():
            try:
                self.Graph_viewer.draw_Graph()
                self.Graph_viewer.set_center()
                self.parent.processEvents()
            except:
                pass

    def Compute_LFP_fonc(self,meth=0):
        if meth == 0:
            electrode_pos = np.array(self.electrode_pos)

            CellPosition = []
            for layer in range(len(self.CellPosition)):
                for n in range(self.CellPosition[layer].shape[0]):
                    if self.CC.List_celltypes[layer][n] == 0:
                        CellPosition.append(self.CellPosition[layer][n])
            CellPosition = np.array(CellPosition)
            #print(electrode_pos)
            Distance_from_electrode = distance.cdist([electrode_pos, electrode_pos], CellPosition, 'euclidean')[0, :]
            # vect direction??
            U = (CellPosition - electrode_pos) / Distance_from_electrode[:, None]

            Ssoma = np.pi * (self.somaSize / 2.) * ((self.somaSize / 2.) + self.somaSize * np.sqrt(5. / 4.))
            Stotal = Ssoma / self.p

            # potentials
            self.LFP = np.zeros(self.pyrVs.shape[1])
            Vs_d = self.pyrVs - self.pyrVd
            Vs_a = self.pyrVs - self.pyrVa

            for i in range(len(Vs_d)):
                Vdi = np.zeros((len(Vs_d[i, :]), 3))
                Vdi[:, 2] = Vs_d[i, :]
                Vm = np.sum(Vdi * U[i, :], axis=1)
                Vm = Vm / (4. * self.sigma * np.pi * Distance_from_electrode[i] * Distance_from_electrode[i])
                Vm = Vm * (self.dendriteSize + self.somaSize) / 2. * self.gc * Stotal
                self.LFP += Vm * 10.e02

            self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)
        elif meth == 1:
            electrode_pos = np.array(self.electrode_pos)
            PPS={}
            PPS['PPSE']=self.pyrPPSE
            PPS['PPSI']=self.pyrPPSI
            PPS['pyrPPSI_s']=self.pyrPPSI_s
            PPS['pyrPPSI_a']=self.pyrPPSI_a

            RP=RecordedPotential.LFP(self.Fs_e)
            self.LFP=RP.computeLFPmono(PPS,electrode_pos,self.CellPosition,self.CC.List_celltypes,self.CC.List_cellsubtypes)

            #
            # CellPosition_a = []
            # CellPosition_d = []
            # CellPosition_d1 = []
            # CellPosition_d23 = []
            # CellPosition_d4 = []
            # CellPosition_d5 = []
            # CellPosition_d6 = []
            # updown=[]
            # CellPosition_s_up =[]
            # CellPosition_s_down = []
            # target = Cell_morphology.Neuron(0, 1)
            # for layer in range(len(self.CC.Cellpos)):
            #     for n in range(self.CC.Cellpos[layer].shape[0]):
            #         if self.CC.List_celltypes[layer][n] == 0:
            #             pos=self.CC.Cellpos[layer][n]
            #             # subtype == 0  # TPC  subtype = 1  # UPC subtype = 2  # IPC subtype = 3  # BPC subtype = 4  # SSC
            #             subtype = self.CC.List_cellsubtypes[layer][n]
            #             target.update_type(0, layer=layer, subtype=subtype)
            #             d1 = np.array([pos[0],pos[1], self.CC.Layertop_pos[4]])
            #             d23 = np.array([pos[0], pos[1], self.CC.Layertop_pos[3]])
            #             d4 = np.array([pos[0], pos[1], self.CC.Layertop_pos[2]])
            #             d5 = np.array([pos[0], pos[1], self.CC.Layertop_pos[1]])
            #             d6 = np.array([pos[0], pos[1], self.CC.Layertop_pos[0]])
            #             s_down = np.array([pos[0], pos[1], pos[2]-target.hsoma/2])
            #             s_up = np.array([pos[0], pos[1], pos[2]+target.hsoma/2])
            #
            #             CellPosition_d1.append(d1)
            #             CellPosition_d23.append(d23)
            #             CellPosition_d4.append(d4)
            #             CellPosition_d5.append(d5)
            #             CellPosition_d6.append(d6)
            #             CellPosition_a.append(np.array([pos[0], pos[1], pos[2] + target.AX_up]))
            #
            #             if subtype in [0,1,3,4]:
            #                 CellPosition_s_up.append(s_up)
            #                 CellPosition_s_down.append(s_down)
            #                 CellPosition_d.append(np.array([pos[0],pos[1], pos[2]+target.Adend_l]))
            #
            #
            #
            #             else:
            #                 CellPosition_s_up.append(s_down)
            #                 CellPosition_s_down.append(s_up)
            #                 CellPosition_d.append(np.array([pos[0],pos[1], pos[2]-target.Adend_l]))
            #
            #
            #
            #
            #
            #
            #
            #
            # CellPosition_a = np.array(CellPosition_a)
            # CellPosition_d = np.array(CellPosition_d)
            # CellPosition_d1 = np.array(CellPosition_d1)
            # CellPosition_d23 = np.array(CellPosition_d23)
            # CellPosition_d4 = np.array(CellPosition_d4)
            # CellPosition_d5 = np.array(CellPosition_d5)
            # CellPosition_d6 = np.array(CellPosition_d6)
            # CellPosition_s_up = np.array(CellPosition_s_up)
            # CellPosition_s_down = np.array(CellPosition_s_down)
            #
            #
            # Distance_from_electrode_d = distance.cdist([electrode_pos, electrode_pos], CellPosition_d, 'euclidean')[0,
            #                             :]
            #
            # Distance_from_electrode_d1 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d1, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_d23 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d23, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_d4 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d4, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_d5 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d5, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_d6 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d6, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_s_up = distance.cdist([electrode_pos, electrode_pos], CellPosition_s_up, 'euclidean')[0,
            #                             :]
            # Distance_from_electrode_s_down = distance.cdist([electrode_pos, electrode_pos], CellPosition_s_down, 'euclidean')[0,
            #                             :]
            #
            # Distance_from_electrode_a = distance.cdist([electrode_pos, electrode_pos], CellPosition_a, 'euclidean')[0,
            #                             :]
            #
            # Res = np.zeros(PPSE[0].shape[1])
            # sigma = 33e-5
            # for k in range(CellPosition_s_up.shape[0]):
            #
            #     ### PPSE dendrite
            #     Res = Res + ((PPSE[0][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d1[k]))
            #     Res = Res - ((PPSE[0][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res + ((PPSE[1][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d23[k]))
            #     Res = Res - ((PPSE[1][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res + ((PPSE[2][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d4[k]))
            #     Res = Res - ((PPSE[2][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res + ((PPSE[3][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d5[k]))
            #     Res = Res - ((PPSE[3][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res + ((PPSE[4][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d6[k]))
            #     Res = Res - ((PPSE[4][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     ### PPSI dendrite
            #
            #     Res = Res - ((PPSI[0][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d1[k]))
            #     Res = Res + ((PPSI[0][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res - ((PPSI[1][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d23[k]))
            #     Res = Res + ((PPSI[1][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res - ((PPSI[2][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d4[k]))
            #     Res = Res + ((PPSI[2][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res - ((PPSI[3][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d5[k]))
            #     Res = Res + ((PPSI[3][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     Res = Res - ((PPSI[4][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d6[k]))
            #     Res = Res + ((PPSI[4][k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #
            #     ### PPSI soma
            #
            #     Res = Res - ((pyrPPSI_s[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            #     Res = Res + ((pyrPPSI_s[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d[k]))
            #
            #     ### PPSE Axon
            #
            #     Res = Res - ((pyrPPSI_a[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_a[k]))
            #     Res = Res + ((pyrPPSI_a[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_down[k]))
            #
            # self.LFP = Res
            self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)




    def Compute_LFPDisk_fonc(self):
        print('Electrod disk')
        t0 = time.time()
        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        print(self.electrode_pos)
        self.electrode_disk = [self.electrod_disk_CB.isChecked(),
                               float(self.electrode_radius_e.text()),
                               float(self.electrode_angle1_e.text()),
                               float(self.electrode_angle2_e.text())]
        Fs = int(self.Fs_e.text())
        ComputeLFP=RecordedPotential.LFP(Fs=Fs, re=self.electrode_disk[1], tx=self.electrode_disk[2], ty=self.electrode_disk[3], pos=self.electrode_pos)

        coords = ComputeLFP.get_electrode_coordinates()
        # coords = Ds.T
        self.Graph_viewer.draw_DiskPoints(coords)
        self.Graph_viewer.set_center()
        self.parent.processEvents()

        Vs_d = self.pyrVs - self.pyrVd
        Vs_a = self.pyrVs - self.pyrVa


        PCsubtypes_Per = np.cumsum(self.CC.PCsubtypes_Per, axis=1)
        Cellsubtypes = []
        for l in range(len(self.CC.Layer_nbCells)):
            for cell in range(int(self.CC.Layer_nbCells[l])):
                if (self.CC.List_celltypes[l][cell] == 0):  # getsubtype
                    if cell < PCsubtypes_Per[l][0]:
                        Cellsubtypes.append(0)  # TPC
                    elif (cell >= PCsubtypes_Per[l][0]) and (cell < PCsubtypes_Per[l][1]):
                        Cellsubtypes.append(1)  # UPC
                    elif (cell >= PCsubtypes_Per[l][1]) and (cell < PCsubtypes_Per[l][2]):
                        Cellsubtypes.append(2)  # IPC
                    elif (cell >= PCsubtypes_Per[l][2]) and (cell < PCsubtypes_Per[l][3]):
                        Cellsubtypes.append(3)  # BPC
                    elif (cell >= PCsubtypes_Per[l][3]) and (cell < PCsubtypes_Per[l][4]):
                        Cellsubtypes.append(4)  # SSC

        nb_pyr = 0
        cellspos = []
        layers = []
        for i in range(len(self.flatindex)):
            l = self.flatindex[i][0]
            n = self.flatindex[i][1]
            if self.CC.List_celltypes[l][n] == 0:
                cellspos.append(self.CellPosition[l][n])
                layers.append(l)
                nb_pyr += 1
        cellspos = np.array(cellspos)
        # print(cellspos)

        self.LFP = ComputeLFP.compute_dipoles(Vs_a, cellspos, Cellsubtypes,layers)
#        self.LFP1 = ComputeLFP.compute_dipoles(Vs_d, cellspos, Cellsubtypes,layers)

#        plt.plot(self.LFP)
#        plt.plot(self.LFP1)
#        plt.plot(self.LFP-self.LFP1)
#        plt.show()


        print(time.time() - t0)

        self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)

    def UpdateLFP(self):
        if self.Temporal_PSD_CB.isChecked():
            self.mascene_LFPViewer.updatePSD(shiftT=self.CC.tps_start - self.T)
        else:
            self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)

    def Compute_LFPDiskCoated_fonc(self):
        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        self.electrode_disk = [self.electrod_disk_CB.isChecked(),
                               float(self.electrode_radius_e.text()),
                               float(self.electrode_angle1_e.text()),
                               float(self.electrode_angle2_e.text())]

        ind = self.Compute_LFPDiskCoated_type_CB.currentIndex()
        if ind in [0, 1, 2]:
            coated = 0
        else:
            coated = 1
        e_g = Electrode.ElectrodeModel( re=self.electrode_disk[1] , Etype=ind, coated=coated )
        LFP = e_g.GetVelec(self.LFP, Fs = int(1000/self.dt))
        self.mascene_LFPViewer.addLFP(LFP,shiftT=self.CC.tps_start - self.T)

    def SaveSimul(self):
        extension = "txt"
        fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=extension + " (*." + extension + ")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName = (fileName[0] + '.' + extension, fileName[1])
        try:
            if fileName[1] == extension +" (*." + extension +")" :
                f = open(fileName[0] , 'w')
                f.write("D" + "\t" + str(self.CC.D) +"\n")
                f.write("L" + "\t" + str(self.CC.L) +"\n")
                f.write("Layer_d" + "\t" + str(self.CC.Layer_d) + "\n")

                f.write("Layer_nbCells" + "\t" + str(self.CC.Layer_nbCells.tolist()) + "\n")
                f.write("PYRpercent" + "\t" + str(self.CC.PYRpercent.tolist()) + "\n")
                f.write("PVpercent" + "\t" + str(self.CC.PVpercent.tolist()) + "\n")
                f.write("SSTpercent" + "\t" + str(self.CC.SSTpercent.tolist()) + "\n")
                f.write("VIPpercent" + "\t" + str(self.CC.VIPpercent.tolist()) + "\n")
                f.write("RLNpercent" + "\t" + str(self.CC.RLNpercent.tolist()) + "\n")

                f.write("PCsubtypes_Per" + "\t" + str(self.CC.PCsubtypes_Per.tolist()) + "\n")
                f.write("NB_PYR" + "\t" + str(self.CC.NB_PYR.tolist()) + "\n")
                f.write("NB_IN" + "\t" + str(self.CC.NB_IN.tolist()) + "\n")
                f.write("NB_PV" + "\t" + str(self.CC.NB_PV.tolist()) + "\n")
                f.write("NB_PV_BC" + "\t" + str(self.CC.NB_PV_BC.tolist()) + "\n")
                f.write("NB_PV_ChC" + "\t" + str(self.CC.NB_PV_ChC.tolist()) + "\n")
                f.write("NB_SST" + "\t" + str(self.CC.NB_SST.tolist()) + "\n")
                f.write("NB_VIP" + "\t" + str(self.CC.NB_VIP.tolist()) + "\n")
                f.write("NB_RLN" + "\t" + str(self.CC.NB_RLN.tolist()) + "\n")

                f.write("Afference_type" + "\t" + str(self.r0.isChecked()) + "\n")
                f.write("Afferences" + "\t" + str(self.CC.Afferences.tolist()) + "\n")

                f.write("StimDuration" + "\t" + self.StimDuration_e.text() +"\n")
                f.write("i_inj" + "\t" + self.i_inj_e.text() +"\n")
                f.write("tau" + "\t" + self.tau_e.text() +"\n")
                f.write("nbStim" + "\t" + self.nbStim_e.text() +"\n")
                f.write("varianceStim" + "\t" + self.varianceStim_e.text() +"\n")
                f.write("seed" + "\t" + self.seed_e.text() +"\n")

                f.write("TH_StimDuration_e" + "\t" + self.TH_StimDuration_e.text() +"\n")
                f.write("TH_i_inj_e" + "\t" + self.TH_i_inj_e.text() +"\n")
                f.write("TH_tau_e" + "\t" + self.TH_tau_e.text() +"\n")
                f.write("TH_nbStim_e" + "\t" + self.TH_nbStim_e.text() +"\n")
                f.write("TH_deltamin_e" + "\t" + self.TH_deltamin_e.text() +"\n")
                f.write("TH_delta_e" + "\t" + self.TH_delta_e.text() +"\n")

                f.write("SimDuration" + "\t" + self.SimDuration_e.text() +"\n")
                f.write("Fs" + "\t" + self.Fs_e.text() +"\n")

                f.write("Cellpos1" + "\t" + str(self.CC.Cellpos[0].tolist() ) + "\n")
                f.write("Cellpos23" + "\t" + str(self.CC.Cellpos[1].tolist() ) + "\n")
                f.write("Cellpos4" + "\t" + str(self.CC.Cellpos[2].tolist() ) + "\n")
                f.write("Cellpos5" + "\t" + str(self.CC.Cellpos[3].tolist() ) + "\n")
                f.write("Cellpos6" + "\t" + str(self.CC.Cellpos[4].tolist() ) + "\n")

                f.write("cell_placement_CB" + "\t" + self.cell_placement_CB.currentText() + "\n")
                f.write("cellplace" + "\t" + self.seed_place.text() + "\n")

                f.write("electrode_x" + "\t" + self.electrode_x_e.text() + "\n")
                f.write("electrode_y" + "\t" + self.electrode_y_e.text() + "\n")
                f.write("electrode_z" + "\t" + self.electrode_z_e.text() + "\n")

                f.write("List_Names" + "\t" + str(self.List_Names) + "\n")
                f.write("List_Colors" + "\t" + str(self.List_Colors) + "\n")
                f.write("List_Neurone_type" + "\t" + str(self.List_Neurone_type) + "\n")

                for dictlayer in self.CC.List_Neurone_param:
                    for dict_param in dictlayer:
                        f.write("dict_param" + "\t" + str(dict_param) + "\n")

                f.close()
        except:
            msg_cri('Not able to save the simulation')

    def LoadSimul(self):
        extension = "txt"
        fileName = QFileDialog.getOpenFileName(caption='Load parameters' ,filter= extension +" (*." + extension +")")
        if  (fileName[0] == '') :
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName= (fileName[0] + '.' + extension , fileName[1])
        if fileName[1] == extension +" (*." + extension +")" :
            f = open(fileName[0], 'r')
            line = f.readline()
            self.D_e.setText(line.split("\t")[-1].replace('\n','').replace(" ", ""))
            line = f.readline()
            self.L_e.setText(line.split("\t")[-1].replace('\n','').replace(" ", ""))
            line = f.readline()
            self.Layer_d	= [float(l) for l in line.split("\t")[-1].replace('\n','').replace("[", "").replace("]", "").split(" ")]

            self.Layer_d1_l.setText(str(self.Layer_d[0]))
            self.Layer_d23_l.setText(str(self.Layer_d[1]))
            self.Layer_d4_l.setText(str(self.Layer_d[2]))
            self.Layer_d5_l.setText(str(self.Layer_d[3]))
            self.Layer_d6_l.setText(str(self.Layer_d[4]))
            self.set_tissue_func()

            line = f.readline()
            Layer_nbCells= [int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]
            line = f.readline()
            PYRpercent = [float(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]
            line = f.readline()
            PVpercent = [float(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]
            line = f.readline()
            SSTpercent = [float(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]
            line = f.readline()
            VIPpercent = [float(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]
            line = f.readline()
            RLNpercent = [float(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")]

            line = f.readline()
            PCsubtypes_Per = np.array([ [float(r) for r in l.split(", ")]  for l in line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split("], [")])
            line = f.readline()
            NB_PYR = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])
            line = f.readline()
            NB_IN = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])
            line = f.readline()
            NB_PV = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])
            line = f.readline()
            NB_PV_BC = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",",                                                                                                                "").split(" ")])
            line = f.readline()
            NB_PV_ChC = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",", "").split(" ")])
            line = f.readline()
            NB_SST = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])
            line = f.readline()
            NB_VIP = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])
            line = f.readline()
            NB_RLN = np.array([int(l) for l in line.split("\t")[-1].replace('\n', '').replace("[", "").replace("]", "").replace(",","").split(" ")])



            self.CC.update_cellNumber(np.array(Layer_nbCells),
                                      np.array(PYRpercent),
                                      np.array(PVpercent),
                                      np.array(SSTpercent),
                                      np.array(VIPpercent),
                                      np.array(RLNpercent),
                                      PCsubtypes_Per=np.array(PCsubtypes_Per),
                                      NB_PYR=NB_PYR,
                                      NB_PV_BC=NB_PV_BC,
                                      NB_PV_ChC=NB_PV_ChC,
                                      NB_IN=NB_IN,
                                      NB_PV=NB_PV,
                                      NB_SST=NB_SST,
                                      NB_VIP=NB_VIP,
                                      NB_RLN=NB_RLN
                                      )

            self.nbcellsnb1.setText(str(int(self.CC.Layer_nbCells[0])))
            self.nbcellsnb23.setText(str(int(self.CC.Layer_nbCells[1])))
            self.nbcellsnb4.setText(str(int(self.CC.Layer_nbCells[2])))
            self.nbcellsnb5.setText(str(int(self.CC.Layer_nbCells[3])))
            self.nbcellsnb6.setText(str(int(self.CC.Layer_nbCells[4])))
            self.nbcellsnbtotal.setText(str(int(np.sum(self.CC.Layer_nbCells))))

            # self.update_cellNnumber()

            self.CC.update_inputNB()

            for i, l in enumerate(self.List_PYRpercent):
                l.setText(str(PYRpercent[i]))

            for i, l in enumerate(self.List_PVpercent):
                l.setText(str(PVpercent[i]))

            for i, l in enumerate(self.List_SSTpercent):
                l.setText(str(SSTpercent[i]))

            for i, l in enumerate(self.List_VIPpercent):
                l.setText(str(VIPpercent[i]))

            for i, l in enumerate(self.List_RLNpercent):
                l.setText(str(RLNpercent[i]))

            line = f.readline()
            Afference_type = line.split("\t")[-1].replace('\n','').replace(" ", "")
            if Afference_type == 'True':
                self.r0.setChecked(True)
            else:
                self.r1.setChecked(True)

            line = f.readline()
            Afferences = np.array([ [float(r) for r in l.split(", ")]  for l in line.split("\t")[-1].replace('\n', '').replace('[[', '').replace(']]', '').split("], [")])
            self.CC.update_connections(Afferences, fixed=not self.r0.isChecked())

            line = f.readline()
            self.StimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.i_inj_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.tau_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.nbStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.varianceStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.seed_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            self.TH_StimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.TH_i_inj_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.TH_tau_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.TH_nbStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.TH_deltamin_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.TH_delta_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            self.SimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Fs_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            Cellpos = []
            for s in sublist:
                Cellpos.append([float(s2) for s2 in s.split(',')])
            self.CC.Cellpos = [np.array(Cellpos)]
            line = f.readline()
            sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            Cellpos = []
            for s in sublist:
                Cellpos.append([float(s2) for s2 in s.split(',')])
            self.CC.Cellpos.append(np.array(Cellpos))
            line = f.readline()
            sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            Cellpos = []
            for s in sublist:
                Cellpos.append([float(s2) for s2 in s.split(',')])
            self.CC.Cellpos.append(np.array(Cellpos))
            line = f.readline()
            sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            Cellpos = []
            for s in sublist:
                Cellpos.append([float(s2) for s2 in s.split(',')])
            self.CC.Cellpos.append(np.array(Cellpos))
            line = f.readline()
            sublist = line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            Cellpos = []
            for s in sublist:
                Cellpos.append([float(s2) for s2 in s.split(',')])
            self.CC.Cellpos.append(np.array(Cellpos))
            self.CellPosition = self.CC.Cellpos

            line = f.readline()
            index = self.cell_placement_CB.findText(line.split("\t")[-1].replace('\n', ''),
                                                    Qt.MatchExactly | Qt.MatchCaseSensitive)
            self.cell_placement_CB.setCurrentIndex(index)

            line = f.readline()
            seed = line.split("\t")[-1].replace('\n', '').replace(" ", "")
            self.seed_place.setText(seed)
            seed = int(seed)
            self.CC.Conx = Connectivity.Create_Connectivity_Matrix(self.CC.C, self.CC.inputpercent, self.CC.NB_DPYR, self.CC.NB_Th, self.CellPosition)


            line = f.readline()
            self.electrode_x_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.electrode_y_e.setText(line.split("\t")[-1].replace('\n','').replace(" ", ""))
            line = f.readline()
            self.electrode_z_e.setText(line.split("\t")[-1].replace('\n','').replace(" ", ""))

            line = f.readline()
            List_Names = line.split("\t")[-1].replace('\n','').replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
            index = 0
            self.List_Names = []
            for l in range(len(self.CC.Cellpos)):
                length = len(self.CC.Cellpos[l])
                self.List_Names.append(List_Names[index:index+length])
                index += length
            line = f.readline()
            List_Colors = line.split("\t")[-1].replace('\n','').replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
            index = 0
            self.List_Colors = []
            for l in range(len(self.CC.Cellpos)):
                length = len(self.CC.Cellpos[l])
                self.List_Colors.append(List_Colors[index:index+length])
                index += length
            line = f.readline()
            List_Neurone_type = [int(l) for l in line.split("\t")[-1].replace('\n', '').replace('[', '').replace(']', '').replace("'",'').replace(" ", '').split(',')]
            index = 0
            self.List_Neurone_type = []
            for l in range(len(self.CC.Cellpos)):
                length = len(self.CC.Cellpos[l])
                self.List_Neurone_type.append(List_Neurone_type[index:index + length])
                index += length
            List_Neurone_param=[]
            for line in f:
                finalstring = line.split("\t")[-1].replace('\n','').replace("{", "").replace("}", "").replace("'", "").replace(" ", "").replace(" ", "")

                # Splitting the string based on , we get key value pairs
                listdict = finalstring.split(",")

                dictionary = {}
                for i in listdict:
                    # Get Key Value pairs separately to store in dictionary
                    keyvalue = i.split(":")

                    # Replacing the single quotes in the leading.
                    m = keyvalue[0].strip('\'')
                    m = m.replace("\"", "")
                    dictionary[m] = float(keyvalue[1].strip('"\''))
                List_Neurone_param.append(dictionary)

            index = 0
            List_Neurone_param2 = []
            for l in range(len(self.CC.Cellpos)):
                length = len(self.CC.Cellpos[l])
                List_Neurone_param2.append(List_Neurone_param[index:index+length])
                index += length

            self.createCells = True
            self.update_Model()
            self.CC.List_Neurone_param=List_Neurone_param2

            self.masceneCM.update()
            # self.update_graph()
            self.electrode_placement_func()

    def SaveRes(self):
        if hasattr(self, 'Sigs_dict'):
            exPopup = QuestionWhatToSave(self)
            if exPopup.exec_() == QDialog.Accepted:
                saveLFP = exPopup.save_lfp_CB.isChecked()
                savesignals = exPopup.save_signal_CB.isChecked()
                savecoords = exPopup.save_coord_CB.isChecked()
                if not saveLFP and not savesignals and not savecoords:
                    return
            else:
                return
            exPopup.deleteLater()
            if not savesignals:
                Sigs_dict={}
            else:
                Sigs_dict = copy.deepcopy(self.Sigs_dict)

            if not 't' in Sigs_dict.keys():
                Sigs_dict["t"] = self.t

            if hasattr(self, 'LFP') and saveLFP:
                if len(self.LFP) == len(Sigs_dict["t"]):
                    Sigs_dict['LFP']= self.LFP

            if hasattr(self, 'CellPosition') and savecoords:
                    Sigs_dict['Coordinates'] = self.CellPosition

            if savecoords:
                fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".mat (*.mat);;.data (*.data)")
            else:
                fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".mat (*.mat);;.data (*.data);;.csv (*.csv);;.dat (*.dat);;.bin (*.bin);;.edf (*.edf)")

            if (fileName[0] == ''):
                return
            tp = Sigs_dict['t']
            if fileName[1] == '.data (*.data)':
                file_pi = open(fileName[0], 'wb')
                pickle.dump(Sigs_dict, file_pi, -1)
                file_pi.close()
            elif fileName[1] == '.bin (*.bin)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1. / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                # file.write('aaa'+" ------"+ '\n')
                file.close()
                keys = list(Sigs_dict.keys())
                array = np.array(Sigs_dict[keys[0]])
                for s in keys[1:]:
                    array = np.vstack((array, Sigs_dict[s] * 1000))
                array = array.T.flatten()
                array.astype('float32')
                s = struct.pack('f' * len(array), *array)
                file = open(os.path.join(path, name + '.bin'), "wb")
                # array.tofile(file)
                file.write(s)
                file.close()

            elif fileName[1] == '.mat (*.mat)':
                scipy.io.savemat(fileName[0], mdict=Sigs_dict)
            elif fileName[1] == '.csv (*.csv)':
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter='\t', lineterminator='\n')
                w.writerow(Sigs_dict.keys())
                for values in Sigs_dict.values():
                    w.writerow(['{:e}'.format(var) for var in values])
                f.close()
            elif fileName[1] == '.dat (*.dat)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1. / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                #
                file.close()
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter=' ', lineterminator='\n')
                for idx in range(len(tp)):
                    line = []
                    for i_v, values in enumerate(Sigs_dict.values()):
                        if i_v == 0:
                            line.append(values[idx])
                        else:
                            line.append(values[idx] * 1000)
                    w.writerow(['{:f}'.format(var) for var in line])
                f.close()
            elif fileName[1] == '.edf (*.edf)':
                Fs = int(1. / (tp[1] - tp[0]))
                Sigs_dict.pop("t")
                N = len(Sigs_dict.keys())
                f = pyedflib.EdfWriter(fileName[0], N, file_type=1)

                lims = 0
                for key, value in Sigs_dict.items():
                    if key not in 't':
                        lims = max(lims,np.max(np.abs(value)))

                lims *= 2
                for i, key in enumerate(Sigs_dict.keys()):
                    f.setLabel(i, key)
                    f.setSamplefrequency(i, Fs)
                    f.setPhysicalMaximum(i, lims)
                    f.setPhysicalMinimum(i, -lims)
                f.update_header()

                kiter = len(tp) // Fs
                for k in range(kiter):
                    for i, key in enumerate(Sigs_dict.keys()):
                        f.writePhysicalSamples(Sigs_dict[key][k * Fs:(k + 1) * Fs].flatten())

                f.update_header()
                f.close()

        return

    def LoadModel(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Model", "", "Cortical column (*.py)")
        if fileName[0] == '':
            return
        if fileName[1] == "Cortical column (*.py)":
            if os.path.splitext(fileName[0])[1] == '':
                fileName[0] = (fileName[0] + '.py', fileName[1])
            fileName = str(fileName[0])
            (filepath, filename) = os.path.split(fileName)
            sys.path.append(filepath)
            (shortname, extension) = os.path.splitext(filename)
            self.Colonne_class = __import__(shortname)
            self.Colonne = getattr(self.Colonne_class, 'Colonne_cortical')
            self.CC = self.Colonne()
            self.CC.updateTime.something_happened.connect(self.updateTime)
            self.createCells = True

    def LoadRes(self):
        pass
        #self.mascene_EEGViewer.update(Sigs_dict=self.Sigs_dict, Colors=self.Color)

    def setmarginandspacing(self, layout):
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)


class QuestionWhatToSave(QDialog):
    def __init__(self, parent=None, item=None,Graph_Items=None):
        super(QuestionWhatToSave, self).__init__(parent)
        self.parent=parent
        self.Param_box = QGroupBox("Select information you wan to save:")
        self.Param_box.setFixedWidth(300)
        self.layout_Param_box = QVBoxLayout()
        self.item =item

        self.CB_layout = QVBoxLayout()
        self.save_lfp_CB = QCheckBox('LFP')
        self.save_signal_CB = QCheckBox('Signals')
        self.save_coord_CB = QCheckBox('Coordinates')
        self.save_coord_l = QLabel('Coordinates can only be save in .mat and .data format.')
        self.CB_layout.addWidget(self.save_lfp_CB)
        self.CB_layout.addWidget(self.save_signal_CB)
        self.CB_layout.addWidget(self.save_coord_CB)
        self.CB_layout.addWidget(self.save_coord_l)
        [cb.setChecked(True) for cb in [self.save_lfp_CB,self.save_signal_CB,self.save_coord_CB]]

        self.horizontalGroupBox_Actions = QWidget()
        self.horizontalGroupBox_Actions.setFixedSize(285,80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok=QPushButton('Ok')
        self.Button_Cancel=QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66,30)
        self.Button_Cancel.setFixedSize(66,30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.CB_layout)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        self.accept()


class CMViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(CMViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')#Figure()
        self.figure.subplots_adjust(left=0.03, bottom=0.02, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self):
        ConnectivityMatrix = self.parent.CC.Conx['connectivitymatrix']
        ExternalPreSynaptic_Cell_AMPA_DPYR = self.parent.CC.ExternalPreSynaptic_Cell_AMPA_DPYR
        ExternalPreSynaptic_Cell_AMPA_Th = self.parent.CC.ExternalPreSynaptic_Cell_AMPA_Th
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        ax = self.figure.add_subplot(111)
        if scipy.sparse.issparse(ConnectivityMatrix):
            im = ax.spy(self.ConnectivityMatrix,markersize=1)
        elif type(ConnectivityMatrix) == type([]):

            # raw=[]
            # col=[]
            # dat=[]
            #
            # for l in range(len(ConnectivityMatrix)):
            #     for c in ConnectivityMatrix[l]:
            #         raw.append(l)
            #         col.append(c)
            #         dat.append(1)
            #         nb+=1
            # s = sparse.coo_matrix((dat, (raw, col)), shape=(len(ConnectivityMatrix), len(ConnectivityMatrix)))
            # im = ax.spy(s,markersize=1)
            raw=[]
            col=[]
            colors=[]
            nb=0
            flat_list = [item for sublist in self.parent.List_Colors for item in sublist]
            for l in range(len(ConnectivityMatrix)):
                raw.append(-self.parent.CC.NB_Th-self.parent.CC.NB_DPYR-2)
                col.append(l)
                colors.append(flat_list[l])

                for c in ConnectivityMatrix[l]:
                    raw.append(c)
                    col.append(l)
                    colors.append(flat_list[c])
                    nb+=1
            for l in range(len(ExternalPreSynaptic_Cell_AMPA_DPYR)):
                for c in range(len(ExternalPreSynaptic_Cell_AMPA_DPYR[l])):
                    raw.append(-ExternalPreSynaptic_Cell_AMPA_DPYR[l][c]-1)
                    col.append(l)
                    colors.append("#000000")
            for l in range(len(ExternalPreSynaptic_Cell_AMPA_Th)):
                for c in range(len(ExternalPreSynaptic_Cell_AMPA_Th[l])):
                    raw.append(-ExternalPreSynaptic_Cell_AMPA_Th[l][c]-1-self.parent.CC.NB_DPYR)
                    col.append(l)
                    colors.append("#999999")
            if platform.system() in ["Windows","Linux"]:
                ax.scatter(raw, col, c=colors, s=10)
            elif platform.system() == "Darwin":
                ax.scatter(raw, col, c=colors, s=5)

        else:
            im = ax.imshow(ConnectivityMatrix)
        if platform.system() in ["Windows", "Linux"]:
            ax.set_title('ConnectivityMatrix' + str(nb), fontdict={'fontsize': 10})
        elif platform.system() == "Darwin":
            ax.set_title('ConnectivityMatrix' + str(nb), fontdict={'fontsize': 5})


        ax.set_xlabel('Sources')
        ax.set_ylabel('Targets')

        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1, linewidth=1, edgecolor='y', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1, linewidth=1, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((0, 0), self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1, self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

        self.canvas.draw_idle()
        self.canvas.show()

    # def closeEvent(self, event):
    #     plt.close(self.figure)

class StimViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(StimViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')#Figure()
        self.figure.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self, stim, th):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        if not stim == []:
            t = np.arange(stim.shape[1]) * self.parent.dt
            ax = self.figure.add_subplot(211)
            for i, s in enumerate(stim):
                ax.plot(t, s + i * float(self.parent.TH_i_inj_e.text()),c='#000000')
        if not th == []:
            t = np.arange(th.shape[1]) * self.parent.dt
            ax = self.figure.add_subplot(212)
            for i, s in enumerate(th):
                ax.plot(t, s + i * float(self.parent.i_inj_e.text()),c='#999999')

        self.canvas.draw_idle()
        self.canvas.show()

    # def closeEvent(self, event):
    #     plt.close(self.figure)

class LFPViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(LFPViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')
        self.figure.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def addLFP(self,lfp, shiftT=0.):
        self.LFP = self.parent.LFP
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t,self.LFP)
        ax.plot(t,lfp)

        self.canvas.draw_idle()
        self.canvas.show()

    def updatePSD(self, shiftT=0.):
        self.LFP = self.parent.LFP
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        fs = int(1 / self.parent.dt )
        # plt.figure()
        ax = self.figure.add_subplot(111)

        # f, t, Sxx = signal.spectrogram(self.LFP, fs* 1000, return_onesided=True,nperseg=fs*50, noverlap=fs*49, nfft=None)
        f, t, Sxx = signal.spectrogram(self.LFP, fs* 1000, return_onesided=True,nperseg=fs*20, noverlap=fs*19, nfft=fs*100)
        fmax=500
        indfmax = np.where(f>fmax)[0][0]
        # ax.pcolormesh(t, f[:indfmax], 10*np.log10(Sxx[:indfmax,:]) )
        ax.pcolormesh(t, f[:indfmax],  Sxx[:indfmax,:] )

        ax.set_ylabel('Frequency [Hz]')

        ax.set_xlabel('Time [sec]')


        self.canvas.draw_idle()
        self.canvas.show()

    def update(self, shiftT=0.):
        self.LFP = self.parent.LFP
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t,self.LFP)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_thresholding(self,sig,sigbin_perc, shiftT=0.,titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t,sig)
        ax.plot(t,sigbin_perc)
        ax.set_ylabel(r'% pyramid firing')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_AutoCorrelation(self,sig,autocorr, shiftT=0.,titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        # plt.figure()
        ax = self.figure.add_subplot(211)
        ax.plot(t, sig)
        ax.set_ylabel(r'% pyramid firing ')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        ax2 = self.figure.add_subplot(212)
        ax2.plot(np.arange(len(autocorr))*(t[1]-t[0]) - len(sig)*(t[1]-t[0]), autocorr)
        ax2.set_ylabel(r'autocorrelation ')
        ax2.set_xlabel(r'Time (s)')
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()


    def update_synchro_NearestDelay_scatter(self,Aps_centers, shiftT=0.,titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        for i in range(len(Aps_centers)):
            ax.scatter(Aps_centers[i],i* np.ones(len(Aps_centers[i])),c='b')

        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay_boxplot(self,Aps_delays, shiftT=0.,titre=''):

        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        Aps_delays_flatten=[]
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax.boxplot(Aps_delays_flatten)
        ax.set_ylabel(r'delays (ms)')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay(self, Aps_centers,Aps_delays, shiftT=0.,titre=''):

        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax1 = self.figure.add_subplot(121)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')
        ax1.set_title(titre)

        ax2 = self.figure.add_subplot(122)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay2(self, Aps_centers,Aps_delays, Aps_gauss_fit, shiftT=0.,titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax1 = self.figure.add_subplot(131)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')

        ax2 = self.figure.add_subplot(132)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        ax3 = self.figure.add_subplot(133)
        # ax3.boxplot(Aps_gauss_fit)
        ax3.scatter(range(len(Aps_gauss_fit)),Aps_gauss_fit,c='k')
        ax3.set_ylabel(r'gauss fit sigma')
        ax3.set_xlabel(r'Nb pyramid cell')
        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_ISI_distance(self,IntegralI,titre=''):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(IntegralI,cmap=current_cmap )
        self.figure.colorbar(im )
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_vanRossum_distance(self,vr_dist, titre='vanRossum_distance'):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vr_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_VictorPurpura_distance(self, vp_dist, titre='VictorPurpura_distance'):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        # plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vp_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    # def closeEvent(self, event):
    #     plt.close(self.figure)

class Rescalesize(QDialog):
    def __init__(self, parent=None, x=None, y=None, z=None):
        super(Rescalesize, self).__init__(parent)
        self.parent=parent
        self.x=x
        self.y=y
        self.z=z
        self.Param_box = QWidget()
        self.Param_box.setFixedWidth(400)
        self.layout_Param_box = QVBoxLayout()


        self.Layout_param = QHBoxLayout()
        self.scale_GB = QGroupBox("rescale")

        # tissue size
        self.tissue_scale_GB = QGroupBox(r'scale value')
        labelX, self.xs_e = Layout_grid_Label_Edit(label=['x'], edit=['1'])
        labelY, self.ys_e = Layout_grid_Label_Edit(label=['y'], edit=['1'])
        labelZ, self.zs_e = Layout_grid_Label_Edit(label=['z'], edit=['1'])
        self.xs_e = self.xs_e[0]
        self.ys_e = self.ys_e[0]
        self.zs_e = self.zs_e[0]
        grid = QGridLayout()
        self.tissue_scale_GB.setLayout(grid)
        # grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.xs_e.textChanged.connect(lambda state, s='x': self.rescale_axe(s))
        self.ys_e.textChanged.connect(lambda state, s='y': self.rescale_axe(s))
        self.zs_e.textChanged.connect(lambda state, s='z': self.rescale_axe(s))
        self.Layout_param.addWidget(self.tissue_scale_GB)

        self.tissue_size_GB = QGroupBox(r'tissue size')
        labelX, self.x_e = Layout_grid_Label_Edit(label=['x'], edit=[x])
        labelY, self.y_e = Layout_grid_Label_Edit(label=['y'], edit=[y])
        labelZ, self.z_e = Layout_grid_Label_Edit(label=['z'], edit=[z])
        self.x_e = self.x_e[0]
        self.y_e = self.y_e[0]
        self.z_e = self.z_e[0]
        grid = QGridLayout()
        self.tissue_size_GB.setLayout(grid)
        # grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.x_e.textChanged.connect(lambda state, s='x': self.resize_axe(s))
        self.y_e.textChanged.connect(lambda state, s='y': self.resize_axe(s))
        self.z_e.textChanged.connect(lambda state, s='z': self.resize_axe(s))
        self.Layout_param.addWidget(self.tissue_size_GB)

        self.horizontalGroupBox_Actions = QGroupBox("Actions")
        self.horizontalGroupBox_Actions.setFixedSize(285,80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok=QPushButton('Ok')
        self.Button_Cancel=QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66,30)
        self.Button_Cancel.setFixedSize(66,30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.Layout_param)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        self.accept()

    def resize_axe(self,s):
        try:
            if s == 'x':
                self.xs_e.blockSignals(True)
                self.xs_e.setText(str(float(self.x_e.text()) / float(self.x)))
                self.xs_e.blockSignals(False)
            elif s =='y':
                self.ys_e.blockSignals(True)
                self.ys_e.setText(str(float(self.y_e.text()) / float(self.y)))
                self.ys_e.blockSignals(False)
            elif s == 'z':
                self.zs_e.blockSignals(True)
                self.zs_e.setText(str(float(self.z_e.text()) / float(self.z)))
                self.zs_e.blockSignals(False)
        except:
            pass

    def rescale_axe(self,s):
        try:
            if s == 'x':
                self.x_e.blockSignals(True)
                self.x_e.setText(str(float(self.xs_e.text()) * float(self.x)))
                self.x_e.blockSignals(False)
            elif s =='y':
                self.y_e.blockSignals(True)
                self.y_e.setText(str(float(self.ys_e.text()) * float(self.y)))
                self.y_e.blockSignals(False)
            elif s == 'z':
                self.z_e.blockSignals(True)
                self.z_e.setText(str(float(self.zs_e.text()) * float(self.z)))
                self.z_e.blockSignals(False)
        except:
            pass

class Afferences_Managment(QDialog):
    def __init__(self,parent):
        super(Afferences_Managment, self).__init__()

        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.List_Var_GB = QWidget()
        label_source = QLabel('Target')
        label_Target = QLabel('S\no\nu\nr\nc\ne')
        label_PYR1 = QLabel('PYR')
        label_PYR2 = QLabel('PYR')
        label_BAS1 = QLabel('BAS')
        label_BAS2 = QLabel('BAS')
        label_OLM1 = QLabel('OLM')
        label_OLM2 = QLabel('OLM')
        label_BIS1 = QLabel('BIS')
        label_BIS2 = QLabel('BIS')
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        grid.setContentsMargins(5, 5, 5, 5, )
        grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignHCenter)
        grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignVCenter)

        # grid.addWidget(label_PYR1, 2, 1)
        # grid.addWidget(label_BAS1, 3, 1)
        # grid.addWidget(label_OLM1, 4, 1)
        # grid.addWidget(label_BIS1, 5, 1)
        # grid.addWidget(label_PYR2, 1, 2)
        # grid.addWidget(label_BAS2, 1, 3)
        # grid.addWidget(label_OLM2, 1, 4)
        # grid.addWidget(label_BIS2, 1, 5)


        # self.List_Var = self.List_Var / 15
        self.List_Var_e = []
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                edit = LineEdit_Int(str(self.parent.CC.Afferences[l, c]))
                self.List_Var_e.append(edit)
                grid.addWidget(edit, l + 2, c + 2)
        self.List_Var_GB.setLayout(grid)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        self.buttons.accepted.connect(self.myaccept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.List_Var_GB)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)



    def myaccept(self):
        matrice = np.zeros(self.parent.CC.Afferences.shape)
        ind = 0
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                matrice[l,c] =  float(self.List_Var_e[ind].text())
        self.parent.CC.Afferences = matrice *0.5
        self.accept()

class Afferences_ManagmentTable(QDialog):
    def __init__(self,parent):
        super(Afferences_ManagmentTable, self).__init__()
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.tableWidget = QTableWidget()

        fnt = QFont()
        fnt.setPointSize(8)
        self.tableWidget.setFont(fnt)

        Afferences = self.parent.CC.Afferences
        line, column = Afferences.shape

        self.tableWidget.horizontalHeader().setDefaultSectionSize(42)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        self.tableWidget.setRowCount(line+4)
        self.tableWidget.setColumnCount(column+2)

        self.tableWidget.setItemDelegate(Delegate())

        item_Layer = ["I","II/III","IV","V","VI"]
        item_type = ["PC","PV","SST","VIP","RLN"]#d0a9ce
        item_color = ["#d0a9ce","#c5e0b4","#dae0f3","#f8cbad","#ffe699"]


        for j in range(len(item_Layer)):
            item = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(2+j*5, 0, 5, 1)
            self.tableWidget.setItem(2+j*5, 0, item)
            item2 = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(0, 2+j*5, 1, 5)
            self.tableWidget.setItem(0, 2+j*5, item2)

        item = QTableWidgetItem("Thalamus")
        self.tableWidget.setSpan(2 + 24 + 1, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 1, 0, item)
        item = QTableWidgetItem("Distant Cortex")
        self.tableWidget.setSpan(2 + 24 + 2, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 2, 0, item)
        item = QTableWidgetItem("Sources")
        self.tableWidget.setItem(1, 0, item)
        item = QTableWidgetItem("Targets")
        self.tableWidget.setItem(0, 1, item)

        ind = 2
        self.List_possible_connexion = np.array(
            [[1, 1, 1, 1, 0],  # PC -> PC,PV,SST,VIP ,RLN  affinités de connexion entre cellules
             [1, 1, 0, 0, 0],  # PV -> PC,PV,SST,VIP ,RLN
             [1, 1, 0, 1, 1],  # SST -> PC,PV,SST,VIP ,RLN
             [0, 0, 1, 0, 0],  # VIP --> PC,PV,SST,VIP ,RLN
             [1, 1, 1, 1, 1]  # RLN --> PC,PV,SST,VIP ,RLN
             ])
        for i in range(5):
            for j in range(len(item_type)):
                item = QTableWidgetItem(item_type[j])
                item2 = QTableWidgetItem(item_type[j])
                item.setBackground(QColor(item_color[j]))
                item2.setBackground(QColor(item_color[j]))

                self.tableWidget.setItem(ind, 1, item)
                self.tableWidget.setItem(1, ind, item2)
                ind += 1

        for c in range(Afferences.shape[0]):
            type1 = int(np.mod(c,5))
            for l in range(Afferences.shape[1]):
                type2 = int(np.mod(l, 5))
                item = QTableWidgetItem(str(Afferences[c, l]))
                # if self.List_possible_connexion[type2,type1]:
                #     item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(c+2, l+2, item)

        # label_source = QLabel('Target')
        # label_Target = QLabel('S\no\nu\nr\nc\ne')
        # label_PYR1 = QLabel('PYR')
        # label_PYR2 = QLabel('PYR')
        # label_BAS1 = QLabel('BAS')
        # label_BAS2 = QLabel('BAS')
        # label_OLM1 = QLabel('OLM')
        # label_OLM2 = QLabel('OLM')
        # label_BIS1 = QLabel('BIS')
        # label_BIS2 = QLabel('BIS')
        # grid = QGridLayout()
        # grid.setAlignment(Qt.AlignTop)
        # grid.setContentsMargins(5, 5, 5, 5, )
        # grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignHCenter)
        # grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignVCenter)
        #
        # # grid.addWidget(label_PYR1, 2, 1)
        # # grid.addWidget(label_BAS1, 3, 1)
        # # grid.addWidget(label_OLM1, 4, 1)
        # # grid.addWidget(label_BIS1, 5, 1)
        # # grid.addWidget(label_PYR2, 1, 2)
        # # grid.addWidget(label_BAS2, 1, 3)
        # # grid.addWidget(label_OLM2, 1, 4)
        # # grid.addWidget(label_BIS2, 1, 5)
        #
        #
        # # self.List_Var = self.List_Var / 15
        # self.List_Var_e = []
        # for l in range(self.parent.CC.Afferences.shape[0]):
        #     for c in range(self.parent.CC.Afferences.shape[1]):
        #         edit = LineEdit_Int(str(self.parent.CC.Afferences[l, c]))
        #         self.List_Var_e.append(edit)
        #         grid.addWidget(edit, l + 2, c + 2)
        # self.List_Var_GB.setLayout(grid)

        self.math_param = QHBoxLayout()
        self.value_LE = LineEdit('0')
        self.add_PB = QPushButton('+')
        self.sub_PB = QPushButton('-')
        self.time_PB = QPushButton('x')
        self.divide_PB = QPushButton('/')
        self.roundup_PB = QPushButton('Î')
        self.Save_PB = QPushButton('Save')
        self.Load_PB = QPushButton('Load')
        self.math_param.addWidget(QLabel('Value'))
        self.math_param.addWidget(self.value_LE)
        self.math_param.addWidget(self.add_PB)
        self.math_param.addWidget(self.sub_PB)
        self.math_param.addWidget(self.time_PB)
        self.math_param.addWidget(self.divide_PB)
        self.math_param.addWidget(self.roundup_PB)
        self.math_param.addWidget(self.Save_PB)
        self.math_param.addWidget(self.Load_PB)
        self.add_PB.clicked.connect(lambda state, x='+': self.math_fun(x))
        self.sub_PB.clicked.connect(lambda state, x='-': self.math_fun(x))
        self.time_PB.clicked.connect(lambda state, x='x': self.math_fun(x))
        self.divide_PB.clicked.connect(lambda state, x='/': self.math_fun(x))
        self.roundup_PB.clicked.connect(lambda state, x='Î': self.math_fun(x))
        self.Save_PB.clicked.connect( self.Save_fun)
        self.Load_PB.clicked.connect( self.Load_fun)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        self.buttons.accepted.connect(self.myaccept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.tableWidget)
        self.layoutparam.addLayout(self.math_param)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)

    def Save_fun(self):
        # try:
        matrice = np.zeros(self.parent.CC.Afferences.shape)
        for l in range(self.parent.CC.Afferences.shape[0]):
            for c in range(self.parent.CC.Afferences.shape[1]):
                item = self.tableWidget.item(l+2,c+2)
                matrice[l,c] =  float(item.text())
        # except:
        #     msg_cri(s='The values in the table are not compatible.\nPlease check them.')

        extension = "csv"
        fileName = QFileDialog.getSaveFileName(caption='Save Matrix', filter=extension + " (*." + extension + ")")
        if  (fileName[0] == '') :
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName= (fileName[0] + '.' + extension , fileName[1])
        # try:
        if fileName[1] == extension +" (*." + extension +")" :
            np.savetxt(fileName[0], matrice, delimiter=";",fmt='%0.4f')
        # except:
        #     msg_cri(s='Impossible to save the file.\n')

    def Load_fun(self):
        fileName = QFileDialog.getOpenFileName(self, "Load Matrix", "", "csv (*.csv)")
        if fileName[0] == '':
            return
        if fileName[1] == "csv (*.csv)":
            matrice = np.loadtxt(fileName[0],delimiter=";")
            for l in range(matrice.shape[0]):
                for c in range(matrice.shape[1]):
                    item = self.tableWidget.item(l+2,c+2)
                    item.setText(str(matrice[l,c]))

    def math_fun(self,s=''):
        Items = self.tableWidget.selectedItems()
        for item in Items:
            if item.column()>1 and item.row()>1:
                # type1 = int(np.mod(item.column(), 5))
                # type2 = int(np.mod(item.row(), 5))
                # if self.List_possible_connexion[type2, type1]:
                value = float(self.value_LE.text())
                cell_val = float(item.text())
                if s == '+':
                    item.setText(str(cell_val + value))
                elif s == '-':
                    item.setText(str(cell_val - value))
                elif s == 'x':
                    item.setText(str(cell_val * value))
                elif s == '/':
                    if not value == 0:
                        item.setText(str(cell_val / value))
                elif s == 'Î':
                    item.setText(str(np.ceil(cell_val)))

    def myaccept(self):
        try:
            matrice = np.zeros(self.parent.CC.Afferences.shape)
            for l in range(self.parent.CC.Afferences.shape[0]):
                for c in range(self.parent.CC.Afferences.shape[1]):
                    item = self.tableWidget.item(l+2,c+2)
                    matrice[l,c] =  float(item.text())
        except:
            msg_cri(s='The values in the table are not compatible.\nPlease check them.')
        self.parent.CC.Afferences = matrice
        self.accept()


class Connection_ManagmentTable(QDialog):
    def __init__(self,parent):
        super(Connection_ManagmentTable, self).__init__()
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.parent = parent

        self.layoutparam = QVBoxLayout()
        self.tableWidget = QTableWidget()

        fnt = QFont()
        fnt.setPointSize(8)
        self.tableWidget.setFont(fnt)

        Afferences = self.parent.CC.inputpercent
        line, column = Afferences.shape

        self.tableWidget.horizontalHeader().setDefaultSectionSize(42)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        self.tableWidget.setRowCount(line+4)
        self.tableWidget.setColumnCount(column+2)

        self.tableWidget.setItemDelegate(Delegate())

        item_Layer = ["I","II/III","IV","V","VI"]
        item_type = ["PC","PV","SST","VIP","RLN"]#d0a9ce
        item_color = ["#d0a9ce","#c5e0b4","#dae0f3","#f8cbad","#ffe699"]

        for j in range(len(item_Layer)):
            item = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(2+j*5, 0, 5, 1)
            self.tableWidget.setItem(2+j*5, 0, item)
            item2 = QTableWidgetItem(item_Layer[j])
            self.tableWidget.setSpan(0, 2+j*5, 1, 5)
            self.tableWidget.setItem(0, 2+j*5, item2)

        item = QTableWidgetItem("Thalamus")
        self.tableWidget.setSpan(2 + 24 + 1, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 1, 0, item)
        item = QTableWidgetItem("Distant Cortex")
        self.tableWidget.setSpan(2 + 24 + 2, 0, 1, 2)
        self.tableWidget.setItem(2 + 24 + 2, 0, item)
        item = QTableWidgetItem("Sources")
        self.tableWidget.setItem(1, 0, item)
        item = QTableWidgetItem("Targets")
        self.tableWidget.setItem(0, 1, item)

        ind = 2
        for i in range(5):
            for j in range(len(item_type)):
                item = QTableWidgetItem(item_type[j])
                item2 = QTableWidgetItem(item_type[j])
                item.setBackground(QColor(item_color[j]))
                item2.setBackground(QColor(item_color[j]))

                self.tableWidget.setItem(ind, 1, item)
                self.tableWidget.setItem(1, ind, item2)
                ind += 1

        for c in range(Afferences.shape[0]):
            for l in range(Afferences.shape[1]):
                item = QTableWidgetItem(str(Afferences[c, l]))
                self.tableWidget.setItem(c+2, l+2, item)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.tableWidget)
        self.layoutparam.addWidget(self.buttons)
        self.setLayout(self.layoutparam)


class Delegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(Delegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter
    # def sizeHint(self, option, index):
    #     s = QStyledItemDelegate.sizeHint(self, option, index)
    #     return max(s.width(), s.height()) * QSize(1, 1)
    # def createEditor(self, parent, option, index):
    #     editor = LineEdit(parent)
    #     return editor

def msg_wait(s):
    msg = QMessageBox()
    msg.setIconPixmap(QPixmap(os.path.join('icons','wait.gif')).scaledToWidth(100))
    icon_label = msg.findChild(QLabel, "qt_msgboxex_icon_label")
    movie = QMovie(os.path.join('icons','wait.gif'))
    setattr(msg, 'icon_label', movie)
    icon_label.setMovie(movie)
    movie.start()

    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setModal(False)
    # msg.setStandardButtons(QMessageBox.Ok)
    msg.show()
    return msg


def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def questionsure(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    strname = s
    msg.setText(strname)
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    ok = msg.exec_()
    if ok == QMessageBox.Ok:
        return True
    else:
        return False


def main():
    app = QApplication(sys.argv)

    if platform.system() in ['Darwin', 'Linux']:
        app.setStyle('Windows')

    import logging
    logging.basicConfig(filename='logdata.log', filemode='w', level=logging.ERROR)

    def my_excepthook(type, value, tback):
        # log the exception here
        co = tback.tb_frame.f_code
        func_name = co.co_name
        line_no = tback.tb_frame.f_lineno
        filename = co.co_filename
        logging.error("main crashed. Error: %s\n%s\n%s", type, value, tback)
        logging.error('Tracing exception: %s "%s" \non line %s of file %s function %s' % (
        type.__name__, value, line_no, filename, func_name))
        import traceback
        string = traceback.format_stack(tback.tb_frame)
        for s in string:
            logging.error('%s\n' % (s))
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        sys.exit(app.exec())

    sys.excepthook = my_excepthook

    ex = ModelMicro_GUI(app)
    ex.setWindowTitle('Model micro GUI')
    # ex.showMaximized()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

