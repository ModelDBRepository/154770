
# Parameters
C      = 281   *pF
gL     =  30   *nS
taum   = C/gL
EL     = -70.6 *mV
VT     = -50.4 *mV
DeltaT =   2   *mV
Vcut   = VT + 5*DeltaT

# Bursting params below
tau_w  =   40  *ms
a      =    2  *nS
b      =  .08  *nA
Vr     = -68.  *mV 

# synaptic
Ee     =   0.  *mV
Ei     = -80.  *mV
tau_e  =   5   *ms
tau_i  =  10   *ms
tau_ns =  .1   *second**1


eqs_reguspiking='''
# membrane equations
 dV/dt   = (gL*(EL-V)+gL*DeltaT*exp((V-VT)/DeltaT)-w + Isyn+Inoise )/C : volt
 dw/dt   = (a*(V-EL)-w)/tau_w           : amp
 Vcut                                   : mV
 Vr                                     : mV

# input currents
 Isyn    = nS*(ge*(Ee-V) + gi*(Ei-V))   : amp
 dge/dt  = -ge/tau_e                    : 1
 dgi/dt  = -gi/tau_i                    : 1

# LFP
 LFP     = ge+gi                        : 1

# noise
 Inoise  = nA*(DC + ACvar*xii)          : amp
 dxii/dt = -xii/tau_ns + xi/second**.5  : 1
 DC                                     : 1
 ACvar                                  : 1
'''


