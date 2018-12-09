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

#SIMS=["A422","A441","A442","A443","A444","A432","A433","A434","A412","A413"]
#LEG=["N=96 D=0.6, ON=50","N=96 D=10 ON=5","N=96 D=10 ON=50","N=96 D=10 ON=200","N=96 D=10 ON=800","N=96 D=1.3 ON=50","N=96 D=1.3 ON=200","N=96 D=1.3 ON=800","N=96 D=0.06 ON=50","N=96 D=0.06 ON=200"]


#SIMS=["A412","A422","A432","A442","sim5"]
#LEG=["N=96 D=0.06 ON=50","N=96 D=0.6, ON=50","N=96 D=1.3 ON=50","N=96 D=10 ON=50","N=96 D=1.3 Infinite on rate"]
#SIMS=["A432","A433","A434","sim5"]
#LEG=["N=96 D=1.3 ION=50","N=96 D=1.3 ON=200","N=96 D=1.3 ON=800","N=96 D=1.3 Infinite on rate"]
#SIMS=["test_l_1","sim3"]
#LEG=["N=24 ON=800","N=24 Infinite on"]
#SIMS=["B122_2","B122_5","B122"]
#LEG=["S=200","S=500","S=2000"]
#SIMS=["B111","B112","B113","B114","B110","B121","B122","B123","B124","B120","B131","B132","B133","B134","B130"]
SIMS=["B411","B412","B413","B414","B410","B421","B422","B423","B424","B420","B431","B432","B433","B434","B430"]
#LEG=["5","50","200","800"]
def fpt(SIMS,LEG):
    simlist=[SIMS[0]]
    a=analyse(simlist)
    #fig_arr=a.plot_boundary()
    FPT_arr=[]
    for count in range(len(SIMS)):
        sim=SIMS[count]
        lab=LEG[count]
        print(sim,lab)
        simlist=[sim]
        a=analyse(simlist)
        r=result(simlist[0])
        r.mkdir(r.resdir)
        a.get_samdirs()
        fig_arr=a.plot_generator()
        [fig_arr,mean]=a.fpt_plot(fig_arr,count,lab)
        FPT_arr.append(mean)
        r.curdir=r.curdir=r.chandra+"fpt_05_Dec"
        r.mkdir(r.curdir)
        r.curdir=r.curdir+"/FPT_N96"
        r.mkdir(r.curdir)
        a.axis_modify(fig_arr,r.curdir,r'$\tau$',r'$P(\tau)$','fpt'+str(sim),1)
    FPT_arr=np.array(FPT_arr)
    a.fpt_heatgrid(r.curdir,FPT_arr,'MFPT (s)','MFPT, N=12')
    print(a)
#fpt(SIMS,SIMS)

def exponent_heatmap(SIMS,exponent):
    simlist=[SIMS[0]]
    a=analyse(simlist)
    #fig_arr=a.plot_boundary()
    FPT_arr=[]
    count=0
    sim=SIMS[count]
    simlist=[sim]
    a=analyse(simlist)
    r=result(simlist[0])
    r.mkdir(r.resdir)
    a.get_samdirs()
    #fig_arr=a.plot_generator()
    #[fig_arr,mean]=a.fpt_plot(fig_arr,count,lab)
    #FPT_arr.append(mean)
    r.curdir=r.curdir=r.chandra+"expoenent_05_Dec"
    r.mkdir(r.curdir)
    r.curdir=r.curdir+"/exponentN96"
    r.mkdir(r.curdir)
    #a.axis_modify(fig_arr,r.curdir,r'$\tau$',r'$P(\tau)$','fpt'+str(sim),1)
    a.fpt_heatgrid(r.curdir,exponent,'MSD exponent','Exponent , N:96')
    print(a)
#SIMS=["B411","B412","B413","B414","B410","B421","B422","B423","B424","B420","B431","B432","B433","B434","B430"]
#SIMS=["B111","B112","B113","B114","B110","B121","B122","B123","B124","B120","B131","B132","B133","B134","B130"]
#exp_arr=np.array([0.997,1.12,1.28,1.36,0.8503,float('nan'),0.97,1.017,1.08,0.731,0.814,0.785,0.742,0.692,0.523])
SIMS=["B411","B412","B413","B414","B410","B421","B422","B423","B424","B420","B431","B432","B433","B434","B430"]
exp_arr=np.array([1.054,1.481,1.683,1.809,1.502,0.9985,1.048,1.206,1.4016,1.4368,0.8033,0.724,0.755,0.906,1.401])
exponent_heatmap(["B110"],exp_arr)

def radial_distribution(SIMS):
    for count in range(len(SIMS)):
        sim=SIMS[count]
        lab=LEG[count]
        print(sim,lab)
        simlist=[sim]
        a=analyse(simlist)
        r=result(simlist[0])
        r.mkdir(r.resdir)
        a.get_samdirs()
        fig_arr=a.plot_boundary()
        fig_arr=a.rad_theta_distribution(fig_arr,lab)
        r.curdir=r.curdir=r.chandra+"27_Nov"
        r.mkdir(r.curdir)
        a.axis_modify(fig_arr,r.curdir,'x','y','density2'+str(sim))
#radial_distribution(SIMS)
#fig, ax =plt.subplots(figsize=(8,8))
#ax=data.plot(x='x',y='y',ax=ax)
#plt.show()



#df = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
