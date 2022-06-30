__author__ = 'Mariam'


import numpy as np
import matplotlib
from math import *
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import signal
import Cell_morphology
from scipy.spatial import distance


class LFP:
    def __init__(self, Fs=25000, re=10, tx=0, ty=0, pos=None):
        # Electrode properties
        #re: electrode radius
        #tx:  tetax angle
        #ty: teta y angle
        if pos is None:
            self.electrode_pos = [0e-3, 0e-3, 2050e-3]
        else:
            self.electrode_pos=pos
        self.r_e = re
        self.tx = tx
        self.ty = ty

        V = []
        f = []
        #Geometrical characteristics of the Soma
        dsoma = np.array([18, 11.77, 24.44,17.30])  # soma diameter for layers 2/3,4,5 and 6 resp in micrometres
        hsoma = np.array([6.32, 7.57, 23.90,16.94])  # soma diameter for layers 2/3,4,5 and 6 resp in micrometres
        dendritelength = np.array([275.8, 499, 792,593])# dendrite  for layers 2/3,4,5 and 6 resp in micrometres

        # surface of soma
        Ssoma = np.pi * (dsoma / 2) * ((dsoma / 2) + np.sqrt((dsoma / 2) ** 2 + hsoma ** 2))

        # proportion of soma
        p = [0.09, 0.04, 0.042, 0.062] #0.15

        #distance between sink and source
        lss = dsoma / 2 + dendritelength / 2
        Stotal = Ssoma / p


        gc = 1e-5  # intercompartment conductance
        fs = Fs  # Hz

        # Electrolyte conductivity
        sigma = 33 * 1e-5  # conductivity = 33.0 * 10e-5 ohm - 1.mm - 1


        # electrode_pos = [150e-3, 150e-3, 100e-3]
        self.K = gc * lss * Stotal / (4 * np.pi * sigma)

        self.nbptswiched = 1000

    def getDiscPts(self, rad, step=0.01):

        i_coords, j_coords = np.meshgrid(np.arange(-rad,rad,step), np.arange(-rad,rad,step), indexing='ij')
        corrds = np.stack((i_coords.flatten(), j_coords.flatten(), 0*i_coords.flatten()) ).T
        dist = np.linalg.norm(corrds-np.array([0,0,0]),axis=1)
        D  = corrds[dist<=rad]
        return D

    def getCylPts(self, rad, step=0.01,th=0.02):
        max_pts = int(2 * rad / step) ** 3
        a = np.zeros(shape=(3, max_pts))
        x = np.linspace(-rad, rad, int(2 * rad / step))
        i = 0
        for x_pt in x:
            y = np.linspace(-sqrt(abs(x_pt ** 2 - rad ** 2)), sqrt(abs(x_pt ** 2 - rad ** 2)),
                            int(2 * sqrt(abs(x_pt ** 2 - rad ** 2)) / step))
            for y_pt in y:
                a[:, i] = [x_pt, y_pt, 0]
                i += 1
        for x_pt in x:
            z = np.linspace(0, th, int(th / step))
            for z_pt in z:
                a[:, i] = [x_pt, -sqrt(abs(x_pt ** 2 - rad ** 2)), z_pt]
                i += 1
                a[:, i] = [x_pt, sqrt(abs(x_pt ** 2 - rad ** 2)), z_pt]
                i += 1
        D = a[:, 0:i]
        if i==0:
            D=[0,0,0]
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # plt.plot(D[0,:],D[1,:],D[2,:])
        # plt.show()
        return D

    def get_electrode_coordinates(self):
        A = np.pi * self.r_e * self.r_e
        D = self.getDiscPts(self.r_e,step=1/((1000/(np.pi*self.r_e*self.r_e))**0.5))
        i = len(np.transpose(D))
        rx = np.array([[1, 0, 0], [0, cos(radians(self.tx)), - sin(radians(self.tx))],
                       [0, sin(radians(self.tx)), cos(radians(self.tx))]])
        ry = np.array([[cos(radians(self.ty)), 0, sin(radians(self.ty))],
                       [0, 1, 0], [-sin(radians(self.ty)), 0, cos(radians(self.ty))]])
        # Ds = np.add(np.matmul( D,np.matmul(rx, ry)), np.array(
        #     [np.ones(i) * self.electrode_pos[0], np.ones(i) * self.electrode_pos[1],
        #      np.ones(i) * self.electrode_pos[2]]))
        # C(i)=[Cr(1) + self.electrode_pos(1), Cr(2) + self.electrode_pos(2), Cr(3) + self.electrode_pos(3)];
        Ds =  np.matmul(D, np.matmul(rx, ry)) + self.electrode_pos
        return Ds

    def ApplyElectrodeTransform(self, D, center=None, tx=0, ty=0):
        if center is None:
            center = [0, 0, 0]
        i = len(np.transpose(D))
        rx = np.array([[1, 0, 0], [0, cos(radians(tx)), - sin(radians(tx))],
                       [0, sin(radians(tx)), cos(radians(tx))]])
        ry = np.array([[cos(radians(ty)), 0, sin(radians(ty))],
                       [0, 1, 0], [-sin(radians(ty)), 0, cos(radians(ty))]])
        Ds = np.add(np.matmul(np.matmul(rx, ry), D), np.array(
            [np.ones(i) * center[0], np.ones(i) * center[1],
             np.ones(i) * center[2]]))
        # C(i)=[Cr(1) + self.electrode_pos(1), Cr(2) + self.electrode_pos(2), Cr(3) + self.electrode_pos(3)];
        return Ds

    def addnoise(self, lfp, SNR=35):
        Plfp = np.mean(lfp ** 2)
        Pnoise = Plfp / 10 ** (SNR / 10)
        noise = np.random.normal(0, np.sqrt(Pnoise), len(lfp))
        lfpn = lfp + noise
        return lfpn

    def compute_dipoles(self, Vsd, cellspos, Cellsubtypes,layers):#VSD of pyr cells #Cellpos of pyrcells

        cellspos = cellspos
        disc = self.get_electrode_coordinates()
        Vdip = np.array([0, 0, 1])
        w_int = np.zeros( Vsd.shape[0])
        v = np.zeros( Vsd.shape[1] )
        i = 0
        Cellsubtypes = np.array(Cellsubtypes)
        VdipTm = np.transpose(-Vdip)
        VdipT = np.transpose(Vdip)
        K = np.array([self.K[ind - 1] for ind in layers  ])



        A = (2*(1/((1000/(np.pi*self.r_e*self.r_e))**0.5) )**2) ** 0.5

        Normals = np.array([0,0,1])

        for di in  disc :

            dist = np.linalg.norm(di - cellspos, axis=1) ** 3
            U = np.zeros(cellspos.shape[0])
            vect_projectOnto = cellspos - di
            projection = (vect_projectOnto * np.sum(Normals * vect_projectOnto, axis=1)[:, None])/   np.sum(vect_projectOnto * vect_projectOnto, axis=1)[:, None]
            norm_vect_projectOnto = np.linalg.norm(vect_projectOnto,axis = 1)
            norm_projection = np.linalg.norm(projection + vect_projectOnto,axis = 1)
            U  = norm_vect_projectOnto - norm_projection

            U[U<0] = 0
            U[Cellsubtypes == 2] = -U[Cellsubtypes == 2]
            v += np.matmul(K*U/dist, Vsd) * A


        # V = v #/ (np.pi * self.r_e * self.r_e)

        # plt.plot(V)
        #b, a = signal.butter(2, 1024 / self.fs, 'low')
        #V_lfp = signal.lfilter(b, a, V)
        # plt.plot(V,'b',V_lfp,'r')
        # plt.show()
        return v * 1000

    def computeLFPmono(self,PPS,Epos,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos):
        #electrode position
        electrode_pos = Epos
        #postsynapticpotentials
        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        pyrPPSI_a = PPS['PPSI_a']

        #positions
        CellPosition_a = []
        CellPosition_d = []
        CellPosition_d1 = []
        CellPosition_d23 = []
        CellPosition_d4 = []
        CellPosition_d5 = []
        CellPosition_d6 = []
        updown = []
        CellPosition_s_up = []
        CellPosition_s_down = []
        target = Cell_morphology.Neuron(0, 1)
        for layer in range(len(Cellpos)):
            for n in range(Cellpos[layer].shape[0]):
                if List_celltypes[layer][n] == 0:
                    pos = Cellpos[layer][n]
                    # subtype == 0  # TPC  subtype = 1  # UPC subtype = 2  # IPC subtype = 3  # BPC subtype = 4  # SSC
                    subtype = List_cellsubtypes[layer][n]
                    target.update_type(0, layer=layer, subtype=subtype)
                    d1 = np.array([pos[0], pos[1], Layertop_pos[4]])
                    d23 = np.array([pos[0], pos[1], Layertop_pos[3]])
                    d4 = np.array([pos[0], pos[1], Layertop_pos[2]])
                    d5 = np.array([pos[0], pos[1], Layertop_pos[1]])
                    d6 = np.array([pos[0], pos[1], Layertop_pos[0]])
                    s_down = np.array([pos[0], pos[1], pos[2] - target.hsoma / 2])
                    s_up = np.array([pos[0], pos[1], pos[2] + target.hsoma / 2])

                    CellPosition_d1.append(d1)
                    CellPosition_d23.append(d23)
                    CellPosition_d4.append(d4)
                    CellPosition_d5.append(d5)
                    CellPosition_d6.append(d6)
                    CellPosition_a.append(np.array([pos[0], pos[1], pos[2] + target.AX_up]))

                    if subtype in [0, 1, 3, 4]:
                        CellPosition_s_up.append(s_up)
                        CellPosition_s_down.append(s_down)
                        CellPosition_d.append(np.array([pos[0], pos[1], pos[2] + target.Adend_l]))



                    else:
                        CellPosition_s_up.append(s_down)
                        CellPosition_s_down.append(s_up)
                        CellPosition_d.append(np.array([pos[0], pos[1], pos[2] - target.Adend_l]))

        CellPosition_a = np.array(CellPosition_a)
        CellPosition_d = np.array(CellPosition_d)
        CellPosition_d1 = np.array(CellPosition_d1)
        CellPosition_d23 = np.array(CellPosition_d23)
        CellPosition_d4 = np.array(CellPosition_d4)
        CellPosition_d5 = np.array(CellPosition_d5)
        CellPosition_d6 = np.array(CellPosition_d6)


        CellPosition_s_up = np.array(CellPosition_s_up)
        CellPosition_s_down = np.array(CellPosition_s_down)

        Distance_from_electrode_d = distance.cdist([electrode_pos, electrode_pos], CellPosition_d, 'euclidean')[0,
                                    :]

        Distance_from_electrode_d1 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d1, 'euclidean')[0,
                                     :]
        Distance_from_electrode_d23 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d23, 'euclidean')[0,
                                      :]
        Distance_from_electrode_d4 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d4, 'euclidean')[0,
                                     :]
        Distance_from_electrode_d5 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d5, 'euclidean')[0,
                                     :]
        Distance_from_electrode_d6 = distance.cdist([electrode_pos, electrode_pos], CellPosition_d6, 'euclidean')[0,
                                     :]
        Distance_from_electrode_s_up = distance.cdist([electrode_pos, electrode_pos], CellPosition_s_up, 'euclidean')[0,
                                       :]
        Distance_from_electrode_s_down = distance.cdist([electrode_pos, electrode_pos], CellPosition_s_down,
                                                        'euclidean')[0,
                                         :]

        Distance_from_electrode_a = distance.cdist([electrode_pos, electrode_pos], CellPosition_a, 'euclidean')[0,
                                    :]

        Res = np.zeros(PPSE[0].shape[1])
        sigma = 33e-5
        for k in range(CellPosition_s_up.shape[0]):
            ### PPSE dendrite
            Res = Res + ((PPSE[0,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d1[k]))
            Res = Res - ((PPSE[0,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res + ((PPSE[1,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d23[k]))
            Res = Res - ((PPSE[1,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res + ((PPSE[2,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d4[k]))
            Res = Res - ((PPSE[2,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res + ((PPSE[3,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d5[k]))
            Res = Res - ((PPSE[3,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res + ((PPSE[4,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d6[k]))
            Res = Res - ((PPSE[4,k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            ### PPSI dendrite

            Res = Res - ((PPSI[0, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d1[k]))
            Res = Res + ((PPSI[0, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res - ((PPSI[1, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d23[k]))
            Res = Res + ((PPSI[1, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res - ((PPSI[2, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d4[k]))
            Res = Res + ((PPSI[2, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res - ((PPSI[3, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d5[k]))
            Res = Res + ((PPSI[3, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            Res = Res - ((PPSI[4, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d6[k]))
            Res = Res + ((PPSI[4, k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))

            ### PPSI soma

            Res = Res - ((pyrPPSI_s[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_up[k]))
            Res = Res + ((pyrPPSI_s[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_d[k]))

            ### PPSE Axon

            Res = Res - ((pyrPPSI_a[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_a[k]))
            Res = Res + ((pyrPPSI_a[k, :]) / ((4 * np.pi * sigma) * Distance_from_electrode_s_down[k]))

        LFP = Res

        return LFP



    def computeMEAV(self, Vsd, cellsp, nbelectrodes=16): #for microelectrode array
        V = []
        x = np.linspace(-150e-3, 150e-3, 4)
        y = np.linspace(-150e-3, 150e-3, 4)
        xx, yy = np.meshgrid(x, y)

        c = np.zeros(shape=(3, 16))
        c[0, :] = np.reshape(xx, (1, 16))
        c[1, :] = np.reshape(yy, (1, 16))
        D = self.getDiscPts(20 * 1e-3, 0.01)
        for cpos in np.transpose(c):
            disc = self.ApplyElectrodeTransform(D, center=cpos + [300e-3, 300e-3, 100e-3])
            Vdip = np.array([0, 0, 1])
            w_int = np.zeros(shape=(1, len(cellsp)))
            v = np.zeros(shape=(len(np.transpose(Vsd)), len(np.transpose(cellsp))))
            i = 0
            for di in np.transpose(disc):
                j = 0
                for dj in cellsp:
                    w_int[0, j] = np.matmul(np.subtract(di, dj), np.transpose(Vdip)) / (
                        np.sum(np.square(np.subtract(di, dj)))) ** (3 / 2)
                    j += 1
                v[:, i] = self.K * np.matmul(w_int, Vsd)
                if i < len(np.transpose(cellsp)) - 1:
                    i += 1

            Vd = np.sum(v, axis=1) / i
            #        V = self.addnoise(V)
            # plt.plot(V)
            b, a = signal.butter(2, 1024 / self.fs, 'low')
            V_lfp = signal.lfilter(b, a, V)
            # plt.plot(V,'b',V_lfp,'r')
            # plt.show()
            V.append(Vd * 1000)
        return V


#  compute HFO content
# class FR_Lib:
#     def computeFR_Energy(S,Ti,tinit=0,tend=0):
#         formatted = (S * 255 / np.max(S)).astype('uint8')
#         ret, thresh = cv.threshold(formatted, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#         kernel = np.ones((3, 3), np.uint8)
#         opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#         sure_bg = cv.dilate(opening, kernel, iterations=3)
#         dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 0)
#         ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
#         sure_fg = np.uint8(sure_fg)
#         unknown = cv.subtract(sure_bg, sure_fg)
#         ret, markers = cv.connectedComponents(sure_fg)
#         markers = markers + 1
#         markers[unknown == 255] = 0
#         # plt.pcolormesh(markers)
#         # plt.show()
#         i = np.where(markers == 1)
#         T = i[1]
#         if (tinit==0)&(tend==0):
#             t0 = Ti[T.min()]
#             te = Ti[T.max()]
#         else:
#             t0=tinit
#             te=tend
#         # print(t0)
#         # print(te)
#         FR = np.sum(S[200:600, T.min():T.max()])
#         return FR,t0,te
#
#     def ComputeHFO(V,fs=25000,tinit=0,tend=0):
#         if fs>2048:
#             Vr = signal.resample(V, int(len(V) * 2048 / fs))
#         else:
#             Vr=V
#         T=len(Vr)/2048
#         coefs = wavedec(Vr, 'db4', level=3)
#         cA3, cD3, cD2, cD1 = coefs
#         fr = np.square(cD3)
#         # plt.plot(fr)
#         # plt.show()
#         #   print(len(fr))
#         w=30e-3*len(fr)/T
#         ti=range(int(w/2),len(fr),int(w/2))
#         # for s in range(len(ti)):
#         #     window=fr[ti[s]-int(w/2):ti[s]+int(w/2)]
#         #     if window.mean()>sqrt(fr.max()):
#         #         tinit=ti[s]-int(w/2)
#         #         break
#         # for s in range(len(ti)):
#         #     window=fr[ti[len(ti)-s-1]-int(w/2):ti[len(ti)-s-1]+int(w/2)]
#         #     if window.mean()>sqrt(fr.max()):
#         #         tend=ti[len(ti)-s-1]+int(w/2)
#         #         break
#
#         t0 = int(tinit *len(fr))
#         te = int(tend*len(fr))
#         # print(t0)
#         # print(te)
#         delta = (tend - tinit) * 1000
#         # print(delta)
#         if 10 < delta < 70:
#             HFO = np.sum(fr[t0:te])
#             print(HFO)
#         else:
#             HFO = 0
#         return HFO
#
#     def ComputeSBR(V,tinit,tend,fs=25000):
#         # T=len(V)/fs
#         t0 = int(tinit*fs)
#         te = int(tend*fs)
#         print(t0)
#         print(te)
#         delta = (tend-tinit) * 1000
#         Ss=V[t0:te]
#         Sb=np.concatenate([V[0:t0-1],V[te+1:len(V)-1]])
#         if 10 < delta < 70:
#             SBR=10*np.log10((len(Sb)*np.sum(np.square(Ss)))/((len(Ss)*np.sum(np.square(Sb)))))
#         else:
#             SBR = 0
#         print(SBR)
#         return SBR
#
#     def Get_FRSeg(S, tt, f):
#         # trasform to gray level image mapp
#         formatted = (S * 255 / np.max(S)).astype('uint8')
#         # Get image type from array and save
#         im = Image.fromarray(formatted)
#         im.save('spectre.png')
#         # Threshold the image
#         ret, thresh = cv.threshold(formatted, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#         kernel = np.ones((3, 3), np.uint8)
#         opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#         sure_bg = cv.dilate(opening, kernel, iterations=3)
#         dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
#         ret, sure_fg = cv.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
#         sure_fg = np.uint8(sure_fg)
#         unknown = cv.subtract(sure_bg, sure_fg)
#         ret, markers = cv.connectedComponents(sure_fg, connectivity=8)
#         markers = markers + 1
#         markers[unknown == 255] = 0
#         #######
#         #######Plot Watershed
#         # plt.subplot(131)
#         # plt.pcolormesh(tt, f,S,cmap=cmap)
#         # plt.subplot(132)
#         # plt.pcolormesh(tt, f, markers,cmap=cmap)
#         # plt.subplot(133)
#         im = cv.imread('spectre.png')
#         markers = cv.watershed(im, markers)
#         w = np.where(markers == 1)
#         T = w[1]
#         T0 = tt[T.min()]
#         T1 = tt[T.max()]
#         if (T1 - T0) > 0.1:
#             th = (T.max() - T.min()) / 2
#             c = np.where(T > th)
#             c = c[0]
#             if len(c) > len(T) / 2:
#                 Tn = T[np.where(T > th)]
#             else:
#                 Tn = T[np.where(T < th)]
#             T0 = tt[Tn.min()]
#             T1 = tt[Tn.max()]
#         # plt.pcolormesh(tt,f,markers,cmap=cmap)
#         # plt.show()
#         print(T0)
#         print(T1)
#
#         S[markers == -1] = S.max() * 2
#         # plt.pcolormesh(tt, f, S,cmap=cmap)
#         # plt.show()
#         return T0, T1


# file = loadmat('Test')  #file path
# Vs = file['Vs']
# Vd = file['Vd']
# NBPYR=file['NBPYR']
# Pos=file['PosCells']
# Pos=Pos[0]
# Pos=[np.array(Pos[i][0:NBPYR[0][i],:]) for i in [1,2,3,4]]
# LayerNB=file['layer_NbCells']
# PYRsubtypes=file['PYRSubtypes']
#
#
# Vsd = np.subtract(Vs, Vd)
# # for i in range(len(Vsd)):
# #     plt.plot(Vsd[i,:])
# #     plt.show()
#
# ComputeLFP=LFP()
#
# LFP=ComputeLFP.compute_dipoles(Vsd=Vsd,cellspos=Pos,Cellsubtypes=PYRsubtypes)
# plt.plot(LFP)
# plt.show()
