import os
import matplotlib
matplotlib.use('Agg')
print(matplotlib.__version__)
import matplotlib.pyplot as plt
from numpy import fft
import numpy as np
from parameter_reader import *
from math import sqrt,atan2,acos,cos,tan,sin
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from random import random
from matplotlib.pyplot import close
#import seaborn as sns; sns.set()
home =os.getenv("HOME")
import matplotlib.ticker as ticker
import pandas as pd
class distinct_colors:
    def color(self):
        direc=home+"/uttam_project/"
        colr=open(direc+"distinct_colors.txt","r").read()
        lines=colr.split()
        return lines
c=distinct_colors()
lines=c.color()
class result:
    def __init__(self,sim):
        self.chandra=home+"/data/uttam_project/"
        self.resdir=self.chandra+str(sim)
        self.curdir="empty"
    def mkdir(self,direc):
        if os.path.isdir(direc)*1 ==0:
            os.mkdir(direc)
class cp_result:
    def __init__(self,simlist,lablist):
        [sim1,sim2,sim3]=simlist
        self.chandra=home+"/data/uttam_project/"
        self.resdirs=[self.chandra+str(sim1),self.chandra+str(sim2),self.chandra+str(sim3)]
        self.labels=lablist
        self.curdir="empty"
    def mkdir(self,direc):
        if os.path.isdir(direc)*1 ==0:
            os.mkdir(direc)
    def cum_plots(self,destn,string,lfsiz,llsiz):
        fig, ax =plt.subplots(figsize=(8,8))
        num=0
        for direc in self.resdirs:
            x1=np.loadtxt(direc+"/step_distribution_V7100/cum_stepxbins.txt")
            y1=np.loadtxt(direc+"/step_distribution_V7100/cum_stepxcum_val.txt")
            ax.plot(x1,y1,ls='-',color=lines[num],label=self.labels[num])
            num+=1
        ax.set_xlabel("d"+string+"$_{min} (\mu $m)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("P($\Delta$"+string+"< d"+string+"$_{min}$)",fontname='serif',fontsize=lfsiz)
        ax.minorticks_on()
        ax.legend()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"cum_step"+string
        ftxt=destn+"/"+"cum_step"+string
        #np.savetxt(ftxt+"bins.txt",bin)
        #np.savetxt(ftxt+"cum_val.txt",cum_val)
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        plt.close()
class analyse:
    def __init__(self,simlist):
        self.simlist=list(simlist)
        self.chandra=home+"/data/uttam_project/"
        self.samdirs=[]
        #input file taken from first uniq of first simname
        uniqs=os.listdir(self.chandra+simlist[0])
        self.in_file_path=self.chandra+str(simlist[0])+"/"+uniqs[0]+"/"+str(simlist[0])+".txt"
        self.p=Params(self.in_file_path)
        #self.cm=cargo_motor(self.in_file_path)
        #print("input file is :",self.in_file_path)
    def get_samdirs(self):  
        for sim in self.simlist:
            uniqs=os.listdir(self.chandra+sim)
            for uniq in uniqs:
                if os.path.isdir(self.chandra+sim+"/"+uniq):
                    self.samdirs.append(self.chandra+sim+"/"+uniq)
    def remove_uncomplete(self):
        rmlist=[]
        for sdir in self.samdirs:
            if not(os.path.isfile(sdir+"/max_time.txt")):
                rmlist.append(sdir)
                #self.samdirs.remove(str(sdir))
        for rm in rmlist:
            self.samdirs.remove(rm)
            #else:
                #print("else")
            #if os.path.isfile(sdir+"/max_time.txt"):
                #print("hi")
                #max_t=np.loadtxt(sdir+"/max_time.txt")
                #print(max_t)
    def plot_boundary(self):
        fig, ax=plt.subplots(figsize=(10,10))
        x1=np.linspace(self.p.delta,self.p.R,100)
        y1=np.zeros(100)
        theta=np.linspace(0,self.p.angle,100)
        x2=self.p.R*np.cos(theta)
        y2=self.p.R*np.sin(theta)
        x3=np.linspace(self.p.delta*cos(self.p.angle),self.p.R*cos(self.p.angle),100)
        y3=tan(self.p.angle)*x3
        x4=self.p.delta*np.cos(theta)
        y4=self.p.delta*np.sin(theta)
        
        ax.plot(x1,y1,'k-',x2,y2,'k-',x3,y3,'k-',x4,y4,'k-')
        return [fig,ax]
    def plot_generator(self):
        fig, ax=plt.subplots(figsize=(10,10))
        return [fig,ax]
    def axis_modify(self,fig_ar,destn,xlabel,ylabel,filename,legd):
        [fig,ax] = fig_ar
        lfsiz=20
        llsiz=15
        ax.set_xlabel(xlabel,fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(ylabel,fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        #ax.ticklabel_format(style='sci')
        if legd==1:
            ax.legend()
        #mean_off=sum(off_time)/(1.0*len(off_time))
        #np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+filename
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
    def titler(self,fig_ar,titl):
        [fig, ax] = fig_ar
        ax.set_title(titl)
        return [fig,ax]
    def plot_fit(self,fig_ar,mx,my,mxfit,myfit,plabel,fitlabel,num):
        [fig, ax]= fig_ar
        mx=np.array(mx)
        my=np.array(my)
        #ax.errorbar(mx,my,yerr=merr,color=lines[num], linestyle='None',marker='o',markerfacecolor=lines[num], markersize=6,label=plabel)
        ax.plot(mx,my,color=lines[num], linestyle='None',marker='o',markerfacecolor=lines[num], markersize=3,label=plabel)
        ax.plot(mxfit,myfit,color='k',linewidth=3.0,label=fitlabel)
        return [fig,ax]
    def fpt_heatgrid(self,destn,a,clabel,title):
        fig, ax= plt.subplots(figsize=(10,10))
        [ll,lf]=[15,15]
        ht=ax.imshow(a.reshape((3,5)),extent=(1,6,1,4))
        cbar=plt.colorbar(ht)
        cbar.set_label(clabel,fontsize=lf)
        ax.set_xticks([1.5,2.5,3.5,4.5,5.5])
        ax.set_yticks([1.5,2.5,3.5])
        ax.set_yticklabels([13,1.3,0.13])
        ax.set_xticklabels([5,50,200,800,'pre'])
        ax.set_xlabel(r'$\pi_o \ (s^{-1})$',fontsize=lf)
        ax.set_ylabel(r'$D \ (\mu m^2s^{-1})$',fontsize=lf)
        ax.set_title(title,fontsize=lf)
        ax.tick_params(axis='both', direction='out',which='both', labelsize=ll)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"heat_map_pdf"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
    def test_analysis_traject(self,fig_ar):
        [fig, ax] = fig_ar
        for sdir in self.samdirs:
            for num in range(100):
                if os.path.isfile(sdir+"/"+str(num)+'.csv'):
                    data = pd.read_csv(sdir+"/"+str(num)+'.csv')
                    #ax=data.plot(x='x',y='y',ax=ax)
                    rad=np.array(data['r'])
                    thet=np.array(data['theta'])
                    ax.plot(rad*np.cos(thet),rad*np.sin(thet))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
        return [fig,ax]
    def msd_calc(self):
        
        sd=0
        for sdir in self.samdirs:
            sd+=1
            for num in range(1000):
                if os.path.isfile(sdir+"/"+str(num)+'.csv'):
                
                    data = pd.read_csv(sdir+"/"+str(num)+'.csv')
                    #ax=data.plot(x='x',y='y',ax=ax
                    rad=np.array(data['r'])
                    thet=np.array(data['theta'])
                    xarr=rad*np.cos(thet)
                    yarr=rad*np.sin(thet)
                    if num==0:
                        msd=[]
                        msd=np.array((xarr-xarr[0])**2+(yarr-yarr[0])**2)
                        df=pd.DataFrame(msd,columns=['a'+str(sd)+str(num)])
                        time=np.array(data['time'])
                        time=time.tolist()
                    else:
                        msd=np.array((xarr-xarr[0])**2+(yarr-yarr[0])**2)
                        df2=pd.DataFrame(msd,columns=['a'+str(sd)+str(num)])
                        df=pd.concat([df,df2],axis=1)
                        time2=np.array(data['time'])
                        time2=time2.tolist()
                        if len(time2)>len(time):
                            time=time2
        out_arr=df.T
        df.to_csv('msd_indi.csv')
        out_arr.to_csv('out_arr.csv')
        MSD_arr=[]
        for ii in range(len(out_arr.columns)):
            sdt=np.array(out_arr[ii])
            sdt = sdt[~np.isnan(sdt)]
            msdt=np.mean(sdt)
            MSD_arr.append(msdt)
        MSD_arr=np.array(MSD_arr)
        msd_df=pd.DataFrame(np.column_stack((time,MSD_arr)),columns=['time','MSD'])
        msd_df.to_csv('msd.csv')
        return [time,MSD_arr]
        
        
    def r_msd_calc(self):
        
        sd=0
        for sdir in self.samdirs:
            sd+=1
            for num in range(1000):
                if os.path.isfile(sdir+"/"+str(num)+'.csv'):
                
                    data = pd.read_csv(sdir+"/"+str(num)+'.csv')
                    #ax=data.plot(x='x',y='y',ax=ax
                    rad=np.array(data['r'])
                    thet=np.array(data['theta'])
                    #xarr=rad*np.cos(thet)
                    #yarr=rad*np.sin(thet)
                    if num==0:
                        msd=[]
                        msd=np.array(rad**2)
                        df=pd.DataFrame(msd,columns=['a'+str(sd)+str(num)])
                        time=np.array(data['time'])
                        time=time.tolist()
                        print("hi")
                    else:
                        msd=np.array((rad-rad[0])**2)
                        df2=pd.DataFrame(msd,columns=['a'+str(sd)+str(num)])
                        df=pd.concat([df,df2],axis=1)
                        time2=np.array(data['time'])
                        time2=time2.tolist()
                        if len(time2)>len(time):
                            time=time2
        out_arr=df.T
        df.to_csv('r_msd_indi.csv')
        out_arr.to_csv('r_out_arr.csv')
        MSD_arr=[]
        for ii in range(len(out_arr.columns)):
            sdt=np.array(out_arr[ii])
            sdt = sdt[~np.isnan(sdt)]
            msdt=np.mean(sdt)
            MSD_arr.append(msdt)
        MSD_arr=np.array(MSD_arr)
        msd_df=pd.DataFrame(np.column_stack((time,MSD_arr)),columns=['time','MSD'])
        msd_df.to_csv('r_msd.csv')
        return [time,MSD_arr]
    def rad_theta_distribution(self,fig_ar,lab):
        [fig, ax] = fig_ar
        for sdir in self.samdirs:
            for num in range(100):
                if os.path.isfile(sdir+"/"+str(num)+'.csv'):
                    data = pd.read_csv(sdir+"/"+str(num)+'.csv')
                    #ax=data.plot(x='x',y='y',ax=ax)
                    ax.scatter(np.array(data['x']),np.array(data['y']),marker='.',color='b')
        ax.set_title(lab)
        return [fig,ax]
    def fpt_plot(self,fig_ar,num,sim):
        [fig, ax] = fig_ar
        for sdir in self.samdirs:
                if os.path.isfile(sdir+"/fpt.csv"):
                    data = pd.read_csv(sdir+"/fpt.csv")
                    arr=np.array(data['fpt'])
        b=max([max(arr),abs(min(arr))])
        val,bin0=np.histogram(arr,bins=int(max(arr)/10),range=(0,max(arr)))
        p=len(bin0)
        bin=[0.5*(bin0[num]+bin0[num+1]) for num in range(p-1)]
        bin=np.array(bin)
        val2=np.divide(val,sum(val)*1.0)
        cum_val=np.cumsum(val2)
        ax.plot(bin,val2,color=lines[num],marker="o", label=sim)
        ax.plot(np.mean(arr),0.05,marker="*",markersize=8,color=lines[num])
        return  [[fig,ax],np.mean(arr)]
