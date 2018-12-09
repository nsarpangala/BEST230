import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp,sqrt,cos,sin
#from mfpt import *
import pandas as pd
from plot_class import *
from joblib import Parallel, delayed
import multiprocessing
import os
import sys
def plot_function_multiple_data(x,y,legend_holder,xlab,ylab,fnam,log_scal):
    fig, ax =plt.subplots(figsize=(8,8))
    [lfsiz,llsiz]=[20,15]
    count=0
    for y1 in y:
        ax.plot(x,y1,'.-',label=legend_holder[count])
        count+=1
    ax.set_xlabel(xlab,fontname='serif',fontsize=lfsiz)
    ax.set_ylabel(ylab,fontname='serif',fontsize=lfsiz)
    if log_scal==1:
        ax.set_yscale('log')
    elif log_scal==2:
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif log_scal==3:
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.legend()
    #ax.set_ylim(1e-3,2)
    ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
    ax.tick_params(axis='both',which='major',length=10,width=1.5 )
    ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
    plt.show()
    fig.savefig(fnam+".png")
    fig.savefig(fnam+".svg",format='svg', dpi=1200)
    fig.savefig(fnam+".eps",format='eps', dpi=1000)
    
#data = pd.read_csv(os.getenv("HOME")+'/data/uttam_project/Test/1541192728/0.csv')
#data.columns = ['time','x','y','stat']
#print(data['time'])


#SIMS=["B111","B112","B113","B114","B122","B123","B124"]
#TITS=[r'$N=12 D=0.13\mu m^2/s \pi_0=5$',r'$N=12 D=0.13\mu m^2/s \pi_0=50$',r'$N=12 D=0.13\mu m^2/s \pi_0=200$',r'$N=12 D=0.13\mu m^2/s \pi_0=800$',r'$N=12 D=1.3\mu m^2/s \pi_0=50$',r'$N=12 D=1.3\mu m^2/s \pi_0=200$',r'$N=12 D=1.3\mu m^2/s \pi_0=800$']
SIMS=["B112"]
TITS=[r'$N=12 \, D=0.13 um \, \pi_0=50$']
#for num in range(len(SIMS)):
#    simlist=[SIMS[num]]
#    a=analyse(simlist)
#    r=result(simlist[0])
#    r.mkdir(r.resdir)
#    a.get_samdirs()
#    fig_arr=a.plot_boundary()
#    fig_arr=a.test_analysis_traject(fig_arr)
#    fig_ar=a.titler(fig_arr,TITS[num])
#    a.axis_modify(fig_arr,r.resdir,'x','y','plot_traject',0)

def traject(sim):
    simlist=[sim]
    a=analyse(simlist)
    r=result(simlist[0])
    r.mkdir(r.resdir)
    a.get_samdirs()
    fig_arr=a.plot_boundary()
    fig_arr=a.test_analysis_traject(fig_arr)
    fig_ar=a.titler(fig_arr,str(sim))
    r.curdir=r.curdir=r.chandra+"trajectories_05_Dec2"
    r.mkdir(r.curdir)
    r.curdir=r.curdir+"/100trajectN12"
    r.mkdir(r.curdir)
    a.axis_modify(fig_arr,r.curdir,'x','y','plot_traject'+str(sim),0)
SIMS=["B111","B112","B113","B114","B110","B121","B122","B123","B124","B120","B131","B132","B133","B134","B130"]
#SIMS=["B411","B412","B413","B414","B410","B421","B422","B423","B424","B420","B431","B432","B433","B434","B430"]
num_cores = int(sys.argv[1])
Parallel(n_jobs=num_cores)(delayed(traject)(SIMS[num]) for num in range(len(SIMS)))
#
#fig, ax =plt.subplots(figsize=(8,8))
#ax=data.plot(x='x',y='y',ax=ax)
#plt.show()



#df = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
