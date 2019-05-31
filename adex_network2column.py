# code for figure 3 from:
# Cohen, M.X (2014). Fluctuations in oscillation frequency control spike
# timing and coordinate neural networks. Journal of Neuroscience

# to run from the command line (unix): python adex_network2column.py <filename>.mat 1.5
#        where "1.5" indicates input strength


from brian import *
import scipy.io
import sys


# skip this simulation if output file exists
import os.path
if os.path.isfile(sys.argv[1])==True:
	quit()



writedata = True


## Specify parameters ---------------------!
Nexc         =    500  # per layer
Ninh         = Nexc/4  # per layer
Pbur         =     .2  # for layers 4/5

NsubgroupR   = Nexc/15 # cells/layer in ensemble
NsubgroupB   = int(Pbur*NsubgroupR) # cells/layer in ensemble

synweightE   =  .2
synweightI   = -3*synweightE # 3xE

synDelay     =   1*ms # (*2 for inter-laminar)
condensityE  =  .025
condensityI  =  4*condensityE # 4xE
ensemblecon  =  .3

initialACvar = 1.2
initialDCval = .2

# timing parameters (in ms)
simdur_init  = 0300
simdur_thal  = 5000
simdur_post  = 0000
## !----------------- end specify parameters



## Create neurons -------------------------!
execfile('eqs4reguspiking.py')
RS_neurons1       = NeuronGroup(int(Nexc*3-Nexc*2*Pbur),model=eqs_reguspiking,threshold=Vcut,reset='V=Vr;w+=b')
RS_neurons1.ACvar = initialACvar
RS_neurons1.DC    = initialDCval
RS_neurons1.Vcut  = (10*rand(len(RS_neurons1))-5)*mV+Vcut
RS_neurons1.Vr    = (10*rand(len(RS_neurons1))-5)*mV+Vr
RS_neurons2       = NeuronGroup(int(Nexc*3-Nexc*2*Pbur),model=eqs_reguspiking,threshold=Vcut,reset='V=Vr;w+=b')
RS_neurons2.ACvar = initialACvar
RS_neurons2.DC    = initialDCval
RS_neurons2.Vcut  = (10*rand(len(RS_neurons2))-5)*mV+Vcut
RS_neurons2.Vr    = (10*rand(len(RS_neurons2))-5)*mV+Vr

execfile('eqs4bursting.py')
BU_neurons1       = NeuronGroup(int(Nexc*2*Pbur),model=eqs_bursting,threshold=Vcut,reset='V=Vr;w+=b')
BU_neurons1.ACvar = initialACvar
BU_neurons1.DC    = initialDCval
BU_neurons1.Vcut  = (10*rand(len(BU_neurons1))-5)*mV+Vcut
BU_neurons1.Vr    = (10*rand(len(BU_neurons1))-5)*mV+Vr
BU_neurons2       = NeuronGroup(int(Nexc*2*Pbur),model=eqs_bursting,threshold=Vcut,reset='V=Vr;w+=b')
BU_neurons2.ACvar = initialACvar
BU_neurons2.DC    = initialDCval
BU_neurons2.Vcut  = (10*rand(len(BU_neurons2))-5)*mV+Vcut
BU_neurons2.Vr    = (10*rand(len(BU_neurons2))-5)*mV+Vr

execfile('eqs4fastspiking.py')
FS_neurons1       = NeuronGroup(Ninh*3,model=eqs_fastspiking,threshold=Vcut,reset='V=Vr;w+=b')
FS_neurons1.ACvar = initialACvar
FS_neurons1.DC    = initialDCval
FS_neurons1.Vcut  = (10*rand(len(FS_neurons1))-5)*mV+Vcut
FS_neurons1.Vr    = (10*rand(len(FS_neurons1))-5)*mV+Vr
FS_neurons2       = NeuronGroup(Ninh*3,model=eqs_fastspiking,threshold=Vcut,reset='V=Vr;w+=b')
FS_neurons2.ACvar = initialACvar
FS_neurons2.DC    = initialDCval
FS_neurons2.Vcut  = (10*rand(len(FS_neurons2))-5)*mV+Vcut
FS_neurons2.Vr    = (10*rand(len(FS_neurons2))-5)*mV+Vr
print "Finished creating neurons..."
## !--------------------- end create neurons



## Create poisson inputs ------------------!
poisN  = 200
poisR  = [0,50]
poiGrp = PoissonGroup(poisN,rates=linspace(poisR[0]*Hz,poisR[1]*Hz,poisN))
p2RS1  = Connection(poiGrp,RS_neurons1,'ge',weight=synweightE/2)
p2FS1  = Connection(poiGrp,FS_neurons1,'ge',weight=synweightE/2)
p2BU1  = Connection(poiGrp,BU_neurons1,'ge',weight=synweightE/2)
p2RS2  = Connection(poiGrp,RS_neurons2,'ge',weight=synweightE/2)
p2FS2  = Connection(poiGrp,FS_neurons2,'ge',weight=synweightE/2)
p2BU2  = Connection(poiGrp,BU_neurons2,'ge',weight=synweightE/2)
## !--------------------- end poisson inputs



## Create layers --------------------------!
L3_RS1 = RS_neurons1.subgroup(Nexc)
L3_FS1 = FS_neurons1.subgroup(Ninh)

L4_RS1 = RS_neurons1.subgroup(int(Nexc-Nexc*Pbur))
L4_BU1 = BU_neurons1.subgroup(int(Nexc*Pbur))
L4_FS1 = FS_neurons1.subgroup(Ninh)

L5_RS1 = RS_neurons1.subgroup(int(Nexc-Nexc*Pbur))
L5_BU1 = BU_neurons1.subgroup(int(Nexc*Pbur))
L5_FS1 = FS_neurons1.subgroup(Ninh)

L3_RS2 = RS_neurons2.subgroup(Nexc)
L3_FS2 = FS_neurons2.subgroup(Ninh)

L4_RS2 = RS_neurons2.subgroup(int(Nexc-Nexc*Pbur))
L4_BU2 = BU_neurons2.subgroup(int(Nexc*Pbur))
L4_FS2 = FS_neurons2.subgroup(Ninh)

L5_RS2 = RS_neurons2.subgroup(int(Nexc-Nexc*Pbur))
L5_BU2 = BU_neurons2.subgroup(int(Nexc*Pbur))
L5_FS2 = FS_neurons2.subgroup(Ninh)

print "Finished segmenting layers..."
## !---------------------- end create layers



## Connect neurons ------------------------!
# within L3
L3_RS2L3_RS1 = Connection(L3_RS1,L3_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e
L3_FS2L3_FS1 = Connection(L3_FS1,L3_FS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L3_RS2L3_FS1 = Connection(L3_RS1,L3_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i
L3_FS2L3_RS1 = Connection(L3_FS1,L3_RS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e
L3_RS2L3_RS2 = Connection(L3_RS2,L3_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e
L3_FS2L3_FS2 = Connection(L3_FS2,L3_FS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L3_RS2L3_FS2 = Connection(L3_RS2,L3_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i
L3_FS2L3_RS2 = Connection(L3_FS2,L3_RS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e
# subnetwork
L3_RS2L3_RS1.connect_random(L3_RS1[0:NsubgroupR],L3_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L3_RS2L3_RS2.connect_random(L3_RS2[0:NsubgroupR],L3_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)


# within L4
L4_RS2L4_RS1 = Connection(L4_RS1,L4_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L4_BU2L4_BU1 = Connection(L4_BU1,L4_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L4_RS2L4_BU1 = Connection(L4_RS1,L4_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L4_BU2L4_RS1 = Connection(L4_BU1,L4_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L4_FS2L4_FS1 = Connection(L4_FS1,L4_FS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L4_RS2L4_FS1 = Connection(L4_RS1,L4_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (rs)
L4_BU2L4_FS1 = Connection(L4_BU1,L4_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (bu)
L4_FS2L4_RS1 = Connection(L4_FS1,L4_RS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (rs)
L4_FS2L4_BU1 = Connection(L4_FS1,L4_BU1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (bu)
L4_RS2L4_RS2 = Connection(L4_RS2,L4_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L4_BU2L4_BU2 = Connection(L4_BU2,L4_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L4_RS2L4_BU2 = Connection(L4_RS2,L4_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L4_BU2L4_RS2 = Connection(L4_BU2,L4_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L4_FS2L4_FS2 = Connection(L4_FS2,L4_FS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L4_RS2L4_FS2 = Connection(L4_RS2,L4_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (rs)
L4_BU2L4_FS2 = Connection(L4_BU2,L4_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (bu)
L4_FS2L4_RS2 = Connection(L4_FS2,L4_RS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (rs)
L4_FS2L4_BU2 = Connection(L4_FS2,L4_BU2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (bu)
# subnetwork
L4_RS2L4_RS1.connect_random(L4_RS1[0:NsubgroupR],L4_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_RS2L4_BU1.connect_random(L4_RS1[0:NsubgroupR],L4_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_BU2L4_RS1.connect_random(L4_BU1[0:NsubgroupB],L4_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L4_BU1.connect_random(L4_BU1[0:NsubgroupB],L4_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_RS2L4_RS2.connect_random(L4_RS2[0:NsubgroupR],L4_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_RS2L4_BU2.connect_random(L4_RS2[0:NsubgroupR],L4_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_BU2L4_RS2.connect_random(L4_BU2[0:NsubgroupB],L4_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L4_BU2.connect_random(L4_BU2[0:NsubgroupB],L4_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)


# within L5
L5_RS2L5_RS1 = Connection(L5_RS1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L5_BU2L5_BU1 = Connection(L5_BU1,L5_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L5_RS2L5_BU1 = Connection(L5_RS1,L5_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L5_BU2L5_RS1 = Connection(L5_BU1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L5_FS2L5_FS1 = Connection(L5_FS1,L5_FS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L5_RS2L5_FS1 = Connection(L5_RS1,L5_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (rs)
L5_BU2L5_FS1 = Connection(L5_BU1,L5_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (bu)
L5_FS2L5_RS1 = Connection(L5_FS1,L5_RS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (rs)
L5_FS2L5_BU1 = Connection(L5_FS1,L5_BU1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (bu)
L5_RS2L5_RS2 = Connection(L5_RS1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L5_BU2L5_BU2 = Connection(L5_BU2,L5_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L5_RS2L5_BU2 = Connection(L5_RS2,L5_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (rs)
L5_BU2L5_RS2 = Connection(L5_BU2,L5_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2e (bu)
L5_FS2L5_FS2 = Connection(L5_FS2,L5_FS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2i
L5_RS2L5_FS2 = Connection(L5_RS2,L5_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (rs)
L5_BU2L5_FS2 = Connection(L5_BU2,L5_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay,column_access=False) # e2i (bu)
L5_FS2L5_RS2 = Connection(L5_FS2,L5_RS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (rs)
L5_FS2L5_BU2 = Connection(L5_FS2,L5_BU2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay,column_access=False) # i2e (bu)
# subnetwork
L5_RS2L5_RS1.connect_random(L5_RS1[0:NsubgroupR],L5_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_RS2L5_BU1.connect_random(L5_RS1[0:NsubgroupR],L5_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_BU2L5_RS1.connect_random(L5_BU1[0:NsubgroupB],L5_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L5_BU1.connect_random(L5_BU1[0:NsubgroupB],L5_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_RS2L5_RS2.connect_random(L5_RS2[0:NsubgroupR],L5_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_RS2L5_BU2.connect_random(L5_RS2[0:NsubgroupR],L5_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_BU2L5_RS2.connect_random(L5_BU2[0:NsubgroupB],L5_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L5_BU2.connect_random(L5_BU2[0:NsubgroupB],L5_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)


# L3->L4
L3_RS2L4_FS1 = Connection(L3_RS1,L4_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i
L3_RS2L4_FS2 = Connection(L3_RS2,L4_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i

# L3->L5
L3_RS2L5_RS1 = Connection(L3_RS1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L3_RS2L5_BU1 = Connection(L3_RS1,L5_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L3_RS2L5_FS1 = Connection(L3_RS1,L5_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L3_RS2L5_RS2 = Connection(L3_RS2,L5_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L3_RS2L5_BU2 = Connection(L3_RS2,L5_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L3_RS2L5_FS2 = Connection(L3_RS2,L5_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
# subnetwork
L3_RS2L5_RS1.connect_random(L3_RS1[0:NsubgroupR],L5_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L3_RS2L5_BU1.connect_random(L3_RS1[0:NsubgroupR],L5_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L3_RS2L5_RS2.connect_random(L3_RS2[0:NsubgroupR],L5_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L3_RS2L5_BU2.connect_random(L3_RS2[0:NsubgroupR],L5_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)


# L4->L3
L4_RS2L3_FS1 = Connection(L4_RS1,L3_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i (rs)
L4_BU2L3_FS1 = Connection(L4_BU1,L3_FS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i (bu)
L4_RS2L3_RS1 = Connection(L4_RS1,L3_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_BU2L3_RS1 = Connection(L4_BU1,L3_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L4_FS2L3_RS1 = Connection(L4_FS1,L3_RS1,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay*2,column_access=False) # e2e (rs)
L4_RS2L3_FS2 = Connection(L4_RS2,L3_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i (rs)
L4_BU2L3_FS2 = Connection(L4_BU2,L3_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2i (bu)
L4_RS2L3_RS2 = Connection(L4_RS2,L3_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_BU2L3_RS2 = Connection(L4_BU2,L3_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L4_FS2L3_RS2 = Connection(L4_FS2,L3_RS2,'gi',sparseness=condensityI,weight=synweightI,delay=synDelay*2,column_access=False) # e2e (rs)
# subnetwork
L4_RS2L3_RS1.connect_random(L4_RS1[0:NsubgroupR],L3_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L3_RS1.connect_random(L4_BU1[0:NsubgroupB],L3_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_RS2L3_RS2.connect_random(L4_RS2[0:NsubgroupR],L3_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L3_RS2.connect_random(L4_BU2[0:NsubgroupB],L3_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)

# L4->L5
L4_RS2L5_RS1 = Connection(L4_RS1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_RS2L5_BU1 = Connection(L4_RS1,L5_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_BU2L5_BU1 = Connection(L4_BU1,L5_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L4_BU2L5_RS1 = Connection(L4_BU1,L5_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L4_RS2L5_RS2 = Connection(L4_RS2,L5_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_RS2L5_BU2 = Connection(L4_RS2,L5_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L4_BU2L5_BU2 = Connection(L4_BU2,L5_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L4_BU2L5_RS2 = Connection(L4_BU2,L5_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
# subnetwork
L4_RS2L5_RS1.connect_random(L4_RS1[0:NsubgroupR],L5_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L5_RS1.connect_random(L4_BU1[0:NsubgroupB],L5_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_RS2L5_BU1.connect_random(L4_RS1[0:NsubgroupR],L5_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_BU2L5_BU1.connect_random(L4_BU1[0:NsubgroupB],L5_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_RS2L5_RS2.connect_random(L4_RS2[0:NsubgroupR],L5_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_BU2L5_RS2.connect_random(L4_BU2[0:NsubgroupB],L5_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L4_RS2L5_BU2.connect_random(L4_RS2[0:NsubgroupR],L5_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)
L4_BU2L5_BU2.connect_random(L4_BU2[0:NsubgroupB],L5_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)

# L5->L3
L5_RS2L3_RS1 = Connection(L5_RS1,L3_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L5_BU2L3_RS1 = Connection(L5_BU1,L3_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_RS2L3_RS2 = Connection(L5_RS2,L3_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (rs)
L5_BU2L3_RS2 = Connection(L5_BU2,L3_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
# subnetwork
L5_RS2L3_RS1.connect_random(L5_RS1[0:NsubgroupR],L3_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L3_RS1.connect_random(L5_BU1[0:NsubgroupB],L3_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_RS2L3_RS2.connect_random(L5_RS2[0:NsubgroupR],L3_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L3_RS2.connect_random(L5_BU2[0:NsubgroupB],L3_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)

# L5->L4
L5_RS2L4_RS1 = Connection(L5_RS1,L4_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_RS2L4_BU1 = Connection(L5_RS1,L4_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_BU2L4_BU1 = Connection(L5_BU1,L4_BU1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_BU2L4_RS1 = Connection(L5_BU1,L4_RS1,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_RS2L4_RS2 = Connection(L5_RS2,L4_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_RS2L4_BU2 = Connection(L5_RS2,L4_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_BU2L4_BU2 = Connection(L5_BU2,L4_BU2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
L5_BU2L4_RS2 = Connection(L5_BU2,L4_RS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*2,column_access=False) # e2e (bu)
# subnetwork
L5_RS2L4_RS1.connect_random(L5_RS1[0:NsubgroupR],L4_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L4_RS1.connect_random(L5_BU1[0:NsubgroupB],L4_RS1[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_RS2L4_BU1.connect_random(L5_RS1[0:NsubgroupR],L4_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_BU2L4_BU1.connect_random(L5_BU1[0:NsubgroupB],L4_BU1[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_RS2L4_RS2.connect_random(L5_RS2[0:NsubgroupR],L4_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_BU2L4_RS2.connect_random(L5_BU2[0:NsubgroupB],L4_RS2[0:NsubgroupR],ensemblecon,weight=synweightE)
L5_RS2L4_BU2.connect_random(L5_RS2[0:NsubgroupR],L4_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)
L5_BU2L4_BU2.connect_random(L5_BU2[0:NsubgroupB],L4_BU2[0:NsubgroupB],ensemblecon,weight=synweightE)


print "Finished building connections..."
## !-------------------- end connect neurons



## now connect across networks ------------!
# L5 -> L3 (simulating top-down connection)
L5_RS2L3_FS12 = Connection(L5_RS1,L3_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*5,column_access=False) # e2e (rs)
L5_RS2L3_FS12.connect_random(L5_RS1[0:NsubgroupR],L3_FS2,ensemblecon,weight=synweightE)
L5_BU2L3_FS12 = Connection(L5_BU1,L3_FS2,'ge',sparseness=condensityE,weight=synweightE,delay=synDelay*5,column_access=False) # e2e (rs)
L5_BU2L3_FS12.connect_random(L5_BU1[0:NsubgroupB],L3_FS2,ensemblecon,weight=synweightE)
## !------------------- end connect networks






## setup electrodes -----------------------!

# membrane potential
BU_volt1 = StateMonitor(L5_BU1,'V',record=True,timestep=10)
RS_volt1 = StateMonitor(L5_RS1,'V',record=True,timestep=10)
FS_volt1 = StateMonitor(L5_FS1,'V',record=True,timestep=10)
BU_volt2 = StateMonitor(L5_BU2,'V',record=True,timestep=10)
RS_volt2 = StateMonitor(L5_RS2,'V',record=True,timestep=10)
FS_volt2 = StateMonitor(L5_FS2,'V',record=True,timestep=10)

# action potentials
BU_APs1  = SpikeMonitor(L5_BU1,record=True)
RS_APs1  = SpikeMonitor(L5_RS1,record=True)
FS_APs1  = SpikeMonitor(L5_FS1,record=True)
BU_APs2  = SpikeMonitor(L5_BU2,record=True)
RS_APs2  = SpikeMonitor(L5_RS2,record=True)
FS_APs2  = SpikeMonitor(L5_FS2,record=True)

# LFPs (sum of synaptic activity)
RS_lfp1  = StateMonitor(RS_neurons1,'LFP',record=True,timestep=10)
BU_lfp1  = StateMonitor(BU_neurons1,'LFP',record=True,timestep=10)
RS_lfp2  = StateMonitor(RS_neurons2,'LFP',record=True,timestep=10)
BU_lfp2  = StateMonitor(BU_neurons2,'LFP',record=True,timestep=10)

# LFPs for export
L3_RS_lfp1 = StateMonitor(L3_RS1,'LFP',record=True,timestep=10)
L4_RS_lfp1 = StateMonitor(L4_RS1,'LFP',record=True,timestep=10)
L4_BU_lfp1 = StateMonitor(L4_BU1,'LFP',record=True,timestep=10)
L5_RS_lfp1 = StateMonitor(L5_RS1,'LFP',record=True,timestep=10)
L5_BU_lfp1 = StateMonitor(L5_BU1,'LFP',record=True,timestep=10)
L3_RS_lfp2 = StateMonitor(L3_RS2,'LFP',record=True,timestep=10)
L4_RS_lfp2 = StateMonitor(L4_RS2,'LFP',record=True,timestep=10)
L4_BU_lfp2 = StateMonitor(L4_BU2,'LFP',record=True,timestep=10)
L5_RS_lfp2 = StateMonitor(L5_RS2,'LFP',record=True,timestep=10)
L5_BU_lfp2 = StateMonitor(L5_BU2,'LFP',record=True,timestep=10)

# APs for export
L3_RS_APs1 = SpikeMonitor(L3_RS1,record=True)
L3_FS_APs1 = SpikeMonitor(L3_FS1,record=True)
L4_RS_APs1 = SpikeMonitor(L4_RS1,record=True)
L4_BU_APs1 = SpikeMonitor(L4_BU1,record=True)
L4_FS_APs1 = SpikeMonitor(L4_FS1,record=True)
L5_RS_APs1 = SpikeMonitor(L5_RS1,record=True)
L5_BU_APs1 = SpikeMonitor(L5_BU1,record=True)
L5_FS_APs1 = SpikeMonitor(L5_FS1,record=True)
L3_RS_APs2 = SpikeMonitor(L3_RS2,record=True)
L3_FS_APs2 = SpikeMonitor(L3_FS2,record=True)
L4_RS_APs2 = SpikeMonitor(L4_RS2,record=True)
L4_BU_APs2 = SpikeMonitor(L4_BU2,record=True)
L4_FS_APs2 = SpikeMonitor(L4_FS2,record=True)
L5_RS_APs2 = SpikeMonitor(L5_RS2,record=True)
L5_BU_APs2 = SpikeMonitor(L5_BU2,record=True)
L5_FS_APs2 = SpikeMonitor(L5_FS2,record=True)

print "Finished setting up monitors"
run(simdur_init*ms) # run a bit to settle transients
print "Finished simulation part 1 of 3"
## !------------------- end setup electrodes



## experiment: thalamic input -------------!
# L4 gets noisy inpupt
L4_RS1.DC    = float(sys.argv[2])
L4_BU1.DC    = float(sys.argv[2])

run(simdur_thal*ms)
print "Finished simulation part 2 of 3"

# reset
L4_RS1.DC    = initialDCval
L4_BU1.DC    = initialDCval

run(simdur_post*ms)
print "Finished simulation part 3 of 3"
## !------------------------- end experiment





## Save data to .mat file -----------------!
if writedata==True:
	lfptimes = BU_volt1.times/ms
	scipy.io.savemat(sys.argv[1],mdict={
	'L3_RS_lfp1':L3_RS_lfp1.values,
	'L4_RS_lfp1':L4_RS_lfp1.values,'L4_BU_lfp1':L4_BU_lfp1.values,
	'L5_RS_lfp1':L5_RS_lfp1.values,'L5_BU_lfp1':L5_BU_lfp1.values,
	'L3_RS_APs1':L3_RS_APs1.spikes,'L3_FS_APs1':L3_FS_APs1.spikes,
	'L4_RS_APs1':L4_RS_APs1.spikes,'L4_BU_APs1':L4_BU_APs1.spikes,'L4_FS_APs1':L4_FS_APs1.spikes,
	'L5_RS_APs1':L5_RS_APs1.spikes,'L5_BU_APs1':L5_BU_APs1.spikes,'L5_FS_APs1':L5_FS_APs1.spikes,
	'L3_RS_lfp2':L3_RS_lfp2.values,
	'L4_RS_lfp2':L4_RS_lfp2.values,'L4_BU_lfp2':L4_BU_lfp2.values,
	'L5_RS_lfp2':L5_RS_lfp2.values,'L5_BU_lfp2':L5_BU_lfp2.values,
	'L3_RS_APs2':L3_RS_APs2.spikes,'L3_FS_APs2':L3_FS_APs2.spikes,
	'L4_RS_APs2':L4_RS_APs2.spikes,'L4_BU_APs2':L4_BU_APs2.spikes,'L4_FS_APs2':L4_FS_APs2.spikes,
	'L5_RS_APs2':L5_RS_APs2.spikes,'L5_BU_APs2':L5_BU_APs2.spikes,'L5_FS_APs2':L5_FS_APs2.spikes,
	'lfptimes':lfptimes,'NsubgroupB':NsubgroupB,'NsubgroupR':NsubgroupR,
	'nRSperLayer':Nexc,'pBUperLayer':Pbur,'nFSperLayer':Ninh,'poisN':poisN,'poisR':poisR,'initialACvar':initialACvar})

## !----------------------------- end saving




