__author__ = 'Mariam, Maxime'

import numpy as np
from numba.experimental import jitclass
from numba import njit, types, typeof

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





@njit()
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
                 PS,
                 pyrVs,
                 pyrVd,
                 pyrVa,
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
                 seed,
                 pyrPPSE,
                 pyrPPSI,
                 pyrPPSI_s,
                 pyrPPSI_a,
                 pyrPPSE_Dpyr,
                 pyrPPSE_Th,
                 pyrI_S,
                 pyrI_d,
                 pyrI_A
):


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
        # print(tt, tp,t[-1])
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


        ########################################
        curr_pyr = 0
        for i in range(np.sum(Layer_nbCells)):
            neurone = indexS[i]

            if len(PS.PreSynaptic_Cell_AMPA[i]) > 0 or len(PS.ExternalPreSynaptic_Cell_AMPA_DPYR[i]) > 0 or len(PS.ExternalPreSynaptic_Cell_AMPA_Th[i]) > 0:
                length_pyr =len(PS.PreSynaptic_Cell_AMPA[i])
                Cell_AMPA = PS.PreSynaptic_Cell_AMPA[i]
                Weight= PS.PreSynapticWeight_AMPA[i]

                External_cell_Dpyr = PS.ExternalPreSynaptic_Cell_AMPA_DPYR[i]
                External_cell_Th = PS.ExternalPreSynaptic_Cell_AMPA_Th[i]


                W = np.ones(len(Cell_AMPA) + len(External_cell_Dpyr) + len(External_cell_Th))
                Vpre_AMPA = np.zeros(len(Cell_AMPA) + len(External_cell_Dpyr) + len(External_cell_Th))  # nb of external +internal AMPA inputs
                if not len(Weight)==0:
                    if len(Weight)>1:
                        W[0:len(Weight)] = Weight / np.max(Weight)

                for k, c in enumerate(Cell_AMPA):  # switch case afferences AMPA ---> PC
                    Vpre_AMPA[k] = List_PYR[indexS[c]].VAis()

                #add external input
                if not len(External_cell_Dpyr)==0:
                    for k, c in enumerate(External_cell_Dpyr):  # External afferences from DPYR
                        Vpre_AMPA[k+len(Cell_AMPA)] = List_DPYR[c].VAis()
                if not len(External_cell_Th)==0:
                    for k, c in enumerate(External_cell_Th):  # External afferences from Th
                        Vpre_AMPA[k+len(Cell_AMPA)+len(External_cell_Dpyr)] = List_Th[c].VAis()


                #compute external PPSE



                if typeS[i] == 0:  # PC
                    # add stim
                    I_AMPA = List_PYR[neurone].I_AMPA2(Vpre_AMPA)#*W
                    I_NMDA = List_PYR[neurone].I_NMDA2(Vpre_AMPA, tps_start)#*W

                    List_PYR[neurone].add_I_synDend_Bis(I_AMPA)
                    List_PYR[neurone].add_I_synDend_Bis(I_NMDA)

                    x = I_AMPA+ I_NMDA
                    ####external PPSE
                    if not len(External_cell_Dpyr) == 0:
                        pyrPPSE_Dpyr[curr_pyr, tt] = np.sum(x[len(Cell_AMPA):len(Cell_AMPA)+len(External_cell_Dpyr)])

                    if not len(External_cell_Th) == 0:
                        pyrPPSE_Th[curr_pyr, tt] = np.sum(x[len(Cell_AMPA)+len(External_cell_Dpyr):len(Cell_AMPA)+len(External_cell_Dpyr)+len(External_cell_Th)])

                    ###compute total current

                    pyrI_S[curr_pyr, tt]=- List_PYR[neurone].I_synSoma - (List_PYR[neurone].g_c / List_PYR[neurone].p_SD * ((List_PYR[neurone].Vs_)- (List_PYR[neurone].Vd_))) - (List_PYR[neurone].g_c / List_PYR[neurone].p_SA * ((List_PYR[neurone].Vs_)- (List_PYR[neurone].Va_)))
                    pyrI_d[curr_pyr, tt]=- List_PYR[neurone].I_synDend - (List_PYR[neurone].g_c / (1 - List_PYR[neurone].p_SD) * (-(List_PYR[neurone].Vs_) + (List_PYR[neurone].Vd_)))
                    pyrI_A[curr_pyr, tt]=- List_PYR[neurone].I_synAis  - (List_PYR[neurone].g_c / (1 - List_PYR[neurone].p_SA) * ( (List_PYR[neurone].Va_) - (List_PYR[neurone].Vs_)))
                    pyrPPSE[1,curr_pyr, tt] = np.sum(x[:length_pyr]*(PS.PreSynapticPos_AMPA[i]==4))
                    pyrPPSE[2,curr_pyr, tt] = np.sum(x[:length_pyr]*(PS.PreSynapticPos_AMPA[i]==3))
                    pyrPPSE[3,curr_pyr, tt] = np.sum(x[:length_pyr]*(PS.PreSynapticPos_AMPA[i]==2))
                    pyrPPSE[4,curr_pyr, tt] = np.sum(x[:length_pyr]*(PS.PreSynapticPos_AMPA[i]==1))





                elif typeS[i] == 1:  # PV
                    I_AMPA = List_PV[neurone].I_AMPA2(Vpre_AMPA)#*W
                    I_NMDA = List_PV[neurone].I_NMDA2(Vpre_AMPA, tps_start)#*W

                    List_PV[neurone].add_I_synSoma(I_AMPA)
                    List_PV[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 2:  # SST
                    I_AMPA = List_SST[neurone].I_AMPA2(Vpre_AMPA)#*W
                    I_NMDA = List_SST[neurone].I_NMDA2(Vpre_AMPA, tps_start)#*W

                    List_SST[neurone].add_I_synSoma(I_AMPA)
                    List_SST[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 3:  # VIP
                    I_AMPA = List_VIP[neurone].I_AMPA2(Vpre_AMPA)#*W
                    I_NMDA = List_VIP[neurone].I_NMDA2(Vpre_AMPA, tps_start)#*W

                    List_VIP[neurone].add_I_synSoma(I_AMPA)
                    List_VIP[neurone].add_I_synSoma(I_NMDA)

                elif typeS[i] == 4:  # RLN
                    I_AMPA = List_RLN[neurone].I_AMPA2(Vpre_AMPA)#*W
                    I_NMDA = List_RLN[neurone].I_NMDA2(Vpre_AMPA, tps_start)#*W

                    List_RLN[neurone].add_I_synSoma(I_AMPA)
                    List_RLN[neurone].add_I_synSoma(I_NMDA)



            ########################################################################
            ##GABA
            if len(PS.PreSynaptic_Cell_GABA[i]) > 0:

                Cell_GABA = PS.PreSynaptic_Cell_GABA[i]
                Vpre_GABA = np.zeros(len(Cell_GABA))

                Weight= PS.PreSynapticWeight_GABA[i]
                Weight_d= PS.PreSynaptic_Soma_GABA_d[i] * PS.PreSynapticWeight_GABA[i]
                Weight_s=1.#PS.PreSynaptic_Soma_GABA_s[i]
                Weight_a=1#PS.PreSynaptic_Soma_GABA_a[i]
                if not len(Weight) == 0:
                    if np.max(Weight_d)==0:
                        Weight_d = np.ones(len(Weight_d))
                    else:
                        Weight_d =  Weight_d / np.max(Weight_d)


                W = np.ones(len(Cell_GABA))
                if not len(Weight)==0:
                    for i_weigth in range(len(Weight)):
                        if PS.PreSynaptic_Soma_GABA_d[i][i_weigth] ==1:
                            W[i_weigth] = Weight_d[i_weigth]
                        elif PS.PreSynaptic_Soma_GABA_s[i][i_weigth] ==1:
                            W[i_weigth] = Weight_s
                        elif PS.PreSynaptic_Soma_GABA_a[i][i_weigth] ==1:
                            W[i_weigth] = Weight_a

                for k, c in enumerate(Cell_GABA):  # switch afferences
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
                    I_GABA = W*List_PYR[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_GABA_d[i], PS.PreSynaptic_Soma_GABA_s[i], PS.PreSynaptic_Soma_GABA_a[i])
                    List_PYR[neurone].add_I_synDend(I_GABA, PS.PreSynaptic_Soma_GABA_d[i])
                    List_PYR[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_GABA_s[i])
                    List_PYR[neurone].add_I_synAis(I_GABA, PS.PreSynaptic_Soma_GABA_a[i])
                    ###################################Add presynaptic currents per layer for dendrites#################
                    x = I_GABA* PS.PreSynaptic_Soma_GABA_d[i]
                    pyrPPSI[0,curr_pyr, tt] = np.sum(x * (PS.PreSynapticPos_GABA[i] == 5))
                    pyrPPSI[1,curr_pyr, tt] = np.sum(x * (PS.PreSynapticPos_GABA[i] == 4))
                    pyrPPSI[2,curr_pyr, tt] = np.sum(x * (PS.PreSynapticPos_GABA[i] == 3))
                    pyrPPSI[3,curr_pyr, tt] = np.sum(x * (PS.PreSynapticPos_GABA[i] == 2))
                    pyrPPSI[4,curr_pyr, tt] = np.sum(x * (PS.PreSynapticPos_GABA[i] == 1))

                    pyrPPSI_s[curr_pyr, tt] = np.sum(I_GABA* PS.PreSynaptic_Soma_GABA_s[i])
                    pyrPPSI_a[curr_pyr, tt] = np.sum(I_GABA* PS.PreSynaptic_Soma_GABA_a[i])

                elif typeS[i] == 1:  # interneuron PV
                    I_GABA = List_PV[neurone].I_GABA2(Vpre_GABA)*W
                    List_PV[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 2:  # interneuron SST
                    I_GABA = List_SST[neurone].I_GABA2(Vpre_GABA)*W
                    List_SST[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 3:  # interneuron VIP
                    I_GABA = List_VIP[neurone].I_GABA2(Vpre_GABA)*W
                    List_VIP[neurone].add_I_synSoma(I_GABA)

                elif typeS[i] == 4:  # RLN
                    I_GABA = List_RLN[neurone].I_GABA2(Vpre_GABA)*W
                    List_RLN[neurone].add_I_synSoma(I_GABA)
            if typeS[i] == 0:
                curr_pyr += 1
        #############################################
        ########Range Kutta#########################

        for i in range(NB_DPYR):
            List_DPYR[i].rk4()

        for i in range(NB_Th):
            List_Th[i].rk4()
        for i in range(np.sum(NB_PYR)):
            List_PYR[i].rk4()
        for i in range(np.sum(NB_PV)):
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



    tps_start += (t[len(t)-1] + dt)


    return t,pyrVs, pyrVd, pyrVa, PV_Vs, SST_Vs, VIP_Vs, RLN_Vs, DPYR_Vs, Th_Vs, pyrPPSE, pyrPPSI, pyrPPSI_s, pyrPPSI_a, pyrPPSE_Dpyr, pyrPPSE_Th, pyrI_S, pyrI_d, pyrI_A


