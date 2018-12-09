import numpy as np
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

simlist=[str(sys.argv[1])]
a=analyse(simlist)
r=result(simlist[0])
r.mkdir(r.resdir)
a.get_samdirs()
fig_arr=a.plot_boundary()
fig_arr=a.fpt_plot(fig_arr)
a.axis_modify(fig_arr,r.resdir,r'$\tau$',r'$P(\tau)$','fpt')
#
#fig, ax =plt.subplots(figsize=(8,8))
#ax=data.plot(x='x',y='y',ax=ax)
#plt.show()



#df = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
