import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp,sqrt,cos,sin
#from mfpt import *
import pandas as pd
from plot_class import *
import os
import sys
#ser=1
#SIMS=["B111","B112","B113","B114","B110"]
#LEG=[r'$D=0.13\mu m^2/s \ \pi_o=5$',r'$D=0.13\mu m^2/s \ \pi_o=50$',r'$D=0.13\mu m^2/s \ \pi_o=200$',r'$D=0.13\mu m^2/s \ \pi_o=800$', r'$D=0.13\mu m^2/s \ PRE Model$']



#ser=2
#SIMS=["B121","B122","B123","B124","B120"]
#LEG=[r'$D=1.3\mu m^2/s \ \pi_o=5$',r'$D=1.3\mu m^2/s \ \pi_o=50$',r'$D=1.3\mu m^2/s \ \pi_o=200$',r'$D=1.3\mu m^2/s \ \pi_o=800$', r'$D=1.3\mu m^2/s \ PRE Model$']

#ser=3
#SIMS=["B131","B132","B133","B134","B130"]
#LEG=[r'$D=13\mu m^2/s \ \pi_o=5$',r'$D=13\mu m^2/s \ \pi_o=50$',r'$D=13\mu m^2/s \ \pi_o=200$',r'$D=13\mu m^2/s \ \pi_o=800$', r'$D=13\mu m^2/s \ PRE \ Model$']

#ser=41
#SIMS=["B411","B412","B413","B414","B410"]
#LEG=[r'$D=0.13\mu m^2/s \ \pi_o=5$',r'$D=0.13\mu m^2/s \ \pi_o=50$',r'$D=0.13\mu m^2/s \ \pi_o=200$',r'$D=0.13\mu m^2/s \ \pi_o=800$', r'$D=0.13\mu m^2/s \ PRE Model$']

#ser=42
#SIMS=["B421","B422","B423","B424","B420"]
#LEG=[r'$D=1.3\mu m^2/s \ \pi_o=5$',r'$D=1.3\mu m^2/s \ \pi_o=50$',r'$D=1.3\mu m^2/s \ \pi_o=200$',r'$D=1.3\mu m^2/s \ \pi_o=800$', r'$D=1.3\mu m^2/s \ PRE Model$']

#ser=43
#SIMS=["B431","B432","B433","B434","B430"]
#LEG=[r'$D=13\mu m^2/s \ \pi_o=5$',r'$D=13\mu m^2/s \ \pi_o=50$',r'$D=13\mu m^2/s \ \pi_o=200$',r'$D=13\mu m^2/s \ \pi_o=800$', r'$D=13\mu m^2/s \ PRE \ Model$']

ser=51
SIMS=["B114","B110","B124","B120","B134","B130"]
LEG=[r'$D=0.13\mu m^2/s \ \pi_o=800$',r'$D=0.13\mu m^2/s \ PRE \ Model$',r'$D=1.3\mu m^2/s \ \pi_o=800$',r'$D=1.3\mu m^2/s \ PRE \ Model$',r'$D=13\mu m^2/s \ \pi_o=800$', r'$D=13\mu m^2/s \ PRE \ Model$']
def fpt(SIMS,LEG):
    simlist=[SIMS[0]]
    a=analyse(simlist)
    fig_arr=a.plot_generator()
    #fig_arr=a.plot_boundary()
    for count in range(len(SIMS)):
        sim=SIMS[count]
        lab=LEG[count]
        print(sim,lab)
        simlist=[sim]
        a=analyse(simlist)
        r=result(simlist[0])
        r.mkdir(r.resdir)
        a.get_samdirs()
        [fig_arr,mean]=a.fpt_plot(fig_arr,count,lab)
    r.curdir=r.curdir=r.chandra+"fpt_06_Dec"
    r.mkdir(r.curdir)
    r.curdir=r.curdir+"/fpt"
    r.mkdir(r.curdir)
    a.axis_modify(fig_arr,r.curdir,r'$\tau$',r'$P(\tau)$','fpt'+str(ser),1)
fpt(SIMS,LEG)
