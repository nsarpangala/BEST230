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

    def hist_force(self,string,desnum,ind,destn,lfsiz,llsiz):
        S_tmax=[]
        T=[]
        Force=[]
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            for tstep in range(0,int(max_t),self.p.samrate):
                B=np.loadtxt(sdir+"/"+str(tstep)+"cargo.txt")
                H=np.loadtxt(sdir+"/"+str(tstep)+"head.txt")
                F=np.loadtxt(sdir+"/"+str(tstep)+"force.txt")
                bind_stat=np.loadtxt(sdir+"/"+str(tstep)+"bind_stat.txt")
                f=0
                for num in range(self.p.N):
                    if ~(np.isnan(F[num][0])):
                        f+=F[num][int(desnum)]
                        if F[num][int(desnum)]>10e-12:
                            print(sdir,tstep)
                        if ind==1:
                            if abs(F[num][int(desnum)]*1e12)>1e-5:
                                Force.append((F[num][int(desnum)])*1e12)
                if ind==0:
                    if abs(f*1e12)>1e-5:        
                        Force.append(f*1e12)
        fig, ax =plt.subplots(figsize=(8,8))
        ax.hist(Force, bins=100, align='mid')
        ax.set_xlabel("F_"+string+" (pN)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        if ind==0:
            fnam=destn+"/"+"hist_force_nor"+string
        else:
            fnam=destn+"/"+"hist_force_individual"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
        
    def plot_thet_phi_traj(self,s_num,main_dire,destn,lsiz,fsiz):
        max_fac=1
        max_t=np.loadtxt(main_dire+"/"+str(s_num)+"/max_time.txt")
        col=['b','r']
        lis=['--','-']
        p=self.p
        cm=self.cm
        for num in range(0,5):
            ThT=[]
            PhP=[]
            T=[]
            fig, axarr = plt.subplots(2, 1, figsize=(8,8))
            A1=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(0)+"anchor.txt")
            B1=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(0)+"cargo.txt") 
            for tstep in range(0,int(max_t/max_fac),p.samrate):
                A2=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"anchor.txt")
                B2=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"cargo.txt")
                bind_stat=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"bind_stat.txt")
                cm.B=B2
                [thet2,phi2]=cm.cart_polar(A2[num])
                ThT.append(thet2)
                PhP.append(phi2)
                T.append(tstep*p.dt)
            axarr[0].plot(T,ThT,ls='-',ms=4,color=lines[4])
            axarr[1].plot(T,PhP,ls='-',ms=4, color=lines[5])
            for ax in axarr:
                ax.minorticks_on()
                ax.tick_params(axis='both', direction='in',which='both', labelsize=lsiz)
                ax.tick_params(axis='both',which='major',length=10,width=1.5 )
                ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
            axarr[1].set_xlabel('time (s)',fontname='serif',fontsize=fsiz)
            axarr[0].set_ylabel(r'$ \theta \, (rad)$', fontname='serif', fontsize=fsiz)
            axarr[1].set_ylabel(r'$ \phi \, (rad)$', fontname='serif', fontsize=fsiz)
            plt.setp(axarr[0].get_xticklabels(), visible=False)
            fig.savefig(destn+"/tp2_time_"+str(s_num)+"_"+str(num)+".png")
            fig.savefig(destn+"/tp2_time_"+str(s_num)+"_"+str(num)+".svg",format='svg', dpi=1200)
            fig.savefig(destn+"/tp2_time_"+str(s_num)+"_"+str(num)+".eps",format='eps', dpi=1000)

    def plot_tp_traj(self,s_num,main_dire,destn):
            max_fac=1.0
            max_t=np.loadtxt(main_dire+"/"+str(s_num)+"/max_time.txt")
            col=['b','r']
            lis=['--','-']
            p=self.p
            cm=self.cm
            for num in range(0,5):
                fig, ax =plt.subplots(figsize=(8,8))
                A1=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(0)+"anchor.txt")
                B1=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(0)+"cargo.txt") 
                for tstep in range(0,int(max_t/max_fac),p.samrate):
                    A2=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"anchor.txt")
                    B2=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"cargo.txt")
                    bind_stat=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"bind_stat.txt")
                    cm.B=B2
                    [thet2,phi2]=cm.cart_polar(A2[num])
                    cm.B=B1
                    [thet1,phi1]=cm.cart_polar(A1[num])
                    #ax.plot([thet1,thet2],[phi1,phi2],ls=lis[int(bind_stat[num])],color=lines[num],ms=3)
                    ax.plot([thet1,thet2],[phi1,phi2],ls='-',color=col[int(bind_stat[num])],ms=3)
                    A1=A2
                    B1=B2
                ax.set_xlabel(r'$\theta \, (rad)$',fontname='serif',fontsize=20)
                ax.set_ylabel(r'$\phi \, (rad)$',fontname='serif',fontsize=20)
                ax.minorticks_on()
                ax.tick_params(axis='both', direction='in',which='both', labelsize=15)
                ax.tick_params(axis='both',which='major',length=10,width=1.5 )
                ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
                fig.savefig(destn+"/"+"tp_traj_"+str(s_num)+"_"+str(num)+".png")
                fig.savefig(destn+"/"+"tp_traj_"+str(s_num)+"_"+str(num)+".svg",format='svg', dpi=1200)
                fig.savefig(destn+"/"+"tp_traj_"+str(s_num)+"_"+str(num)+".eps",format='eps', dpi=1000)


    def vel_distri(self,scale,string,desnum,destn,lfsiz,llsiz):
        S_tmax=[]
        T=[]
        Vel=[]
        dirs=[self.samdirs[0]]
        #for sdir in dirs:
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            for tstep in range(scale*self.p.samrate,int(max_t/scale),self.p.samrate):
                B1=np.loadtxt(sdir+"/"+str(tstep-scale*self.p.samrate)+"cargo.txt")
                B2=np.loadtxt(sdir+"/"+str(tstep+scale*self.p.samrate)+"cargo.txt")
                Vel.append((B2[desnum]-B1[desnum])/(2*(scale*self.p.samrate*self.p.dt)*1e-6))
        fig, ax =plt.subplots(figsize=(8,8))
        b=max([max(Vel),abs(min(Vel))])
        Vel_val,Vel_bin=np.histogram(Vel,bins=int(20),range=(-1.0*b,b))
        Vel_pos_val=[]
        for num in range(int(len(Vel_val)/2.0)):
		        Vel_pos_val.append(Vel_val[len(Vel_val)-1-num]-Vel_val[num])
        p=len(Vel_bin)
        Vel_pos_bin=Vel_bin[int(p/2):p-1]
        Vel_pos_bin=list(Vel_pos_bin)
        Vel_pos_bin.reverse()
        ax.plot(Vel_pos_bin,Vel_pos_val,'ro-')
        #ax.hist(Vel, bins=int(400/scale), align='mid')#[give thermal velocity distribution]
        ax.set_xlabel("v_"+string+" ($\mu $m/s)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"hist_vel"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
    def active_distri(self,destn,lfsiz,llsiz):
        Run=[]
        Act_time=[]
        print(len(self.samdirs))
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            B=np.loadtxt(sdir+"/"+str(int(max_t))+"cargo.txt")
            Run.append(B[0])
            Act_time.append(max_t)
        fig, ax =plt.subplots(figsize=(8,8))
        Run=np.multiply(Run,1e6)
        ax.hist(Run, bins=int(10), align='mid')
        ax.set_xlabel("Run length($\mu $m)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"hist_run"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
        
        fig, ax =plt.subplots(figsize=(8,8))
        ax.hist(Act_time, bins=int(50), align='mid')
        ax.set_xlabel("Active lifetime (s)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"hist_active"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
        
    def motor_velocity_distribution(self,scale,string,desnum,destn,lfsiz,llsiz):
        S_tmax=[]
        T=[]
        Vel=[]
        dirs=[self.samdirs[0]]
        p=self.p
        #for sdir in dirs:
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            for tstep in range(scale*self.p.samrate,int(max_t/scale),self.p.samrate):
                B1=np.loadtxt(sdir+"/"+str(tstep-scale*self.p.samrate)+"head.txt")
                B2=np.loadtxt(sdir+"/"+str(tstep+scale*self.p.samrate)+"head.txt")
                bind2=np.loadtxt(sdir+"/"+str(tstep+scale*self.p.samrate)+"bind_stat.txt")
                bind1=np.loadtxt(sdir+"/"+str(tstep-scale*self.p.samrate)+"bind_stat.txt")
                for num in range(p.N):
                    if int(bind2[num])==1 and int(bind1[num])==1:
                        Vel.append((B2[num][desnum]-B1[num][desnum])/(2*(scale*self.p.samrate*self.p.dt)*1e-6))
        fig, ax =plt.subplots(figsize=(8,8))
#        b=max([max(Vel),abs(min(Vel))])
#        Vel_val,Vel_bin=np.histogram(Vel,bins=int(50),range=(-1.0*b,b))
#        Vel_pos_val=[]
#        for num in range(int(len(Vel_val)/2.0)):
#		        Vel_pos_val.append(Vel_val[len(Vel_val)-1-num]-Vel_val[num])
#        p=len(Vel_bin)
#        Vel_pos_bin=Vel_bin[int(p/2):p-1]
#        Vel_pos_bin=list(Vel_pos_bin)
#        Vel_pos_bin.reverse()
#        ax.plot(Vel_pos_bin,Vel_pos_val,'ro-')
        ax.hist(Vel, bins=int(50), align='mid')#[give thermal velocity distribution]
        ax.set_xlabel("v_"+string+" ($\mu $m/s)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"hist_mot_vel"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
    
    def gaus(self,x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def step_distribution(self,scale,string,desnum,destn,lfsiz,llsiz):
        S_tmax=[]
        T=[]
        Step=[]
        dirs=self.samdirs[:100]
        for sdir in dirs:
        #for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            for tstep in range(scale*self.p.samrate,int(max_t),self.p.samrate):
                B1=np.loadtxt(sdir+"/"+str(tstep-scale*self.p.samrate)+"cargo.txt")
                B2=np.loadtxt(sdir+"/"+str(tstep)+"cargo.txt")
                Step.append((B2[desnum]-B1[desnum])*(1e6))
#        fig, ax =plt.subplots(figsize=(8,8))
        b=max([max(Step),abs(min(Step))])
        val,bin0=np.histogram(Step,bins=int(50),range=(-1.0*b,b))
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        bin=np.array(bin)
        val2=np.divide(val,sum(val)*1.0)
        cum_val=np.cumsum(val2)
#        ax.plot(bin,cum_val,'r+-')
#        ax.set_xlabel("d"+string+"$_{min} (\mu $m)",fontname='serif',fontsize=lfsiz)
#        ax.set_ylabel("P($\Delta$"+string+"< d"+string+"$_{min}$)",fontname='serif',fontsize=lfsiz)
#        ax.minorticks_on()
#        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
#        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
#        fnam=destn+"/"+"cum_step"+string
#        ftxt=destn+"/"+"cum_step"+string
#        np.savetxt(ftxt+"bins.txt",bin)
#        np.savetxt(ftxt+"cum_val.txt",cum_val)
#        fig.savefig(fnam+".png")
#        fig.savefig(fnam+".svg",format='svg', dpi=1200)
#        fig.savefig(fnam+".eps",format='eps', dpi=1000)
#        plt.close()
#        b=max([max(Step),abs(min(Step))])
#        fig, ax =plt.subplots(figsize=(8,8))
#        val1,bin=np.histogram(Step,bins=int(200),range=(-1.0*b,b))
#        val=np.divide(val1,sum(val1)*1.0)
#        p=len(bin)
#        bin=[bin[num] for num in range(p-1)]
#        ax.plot(np.array(bin),val,'bo')
#        n = len(bin)
#        a=max(val)                          
        mean = np.sum(np.multiply(val2,bin))
#        print(val[1])
#        print(bin[1])  
#        print("mean",mean)                
#        #sigma = sum(val*(bin-mean)**2)
#        sigma=0.01
#        print(sigma)
#        popt,pcov = curve_fit(self.gaus,bin,val,p0=[a,mean,sigma])
#        ax.plot(np.array(bin),self.gaus(bin,*popt),'k-',label='fit Parameters [a,mean,sigma]'+str(popt))
#        ax.legend()
#        ax.set_xlabel("$\Delta$"+string+"($\mu $m)",fontname='serif',fontsize=lfsiz)
#        ax.set_ylabel("P($\Delta$"+string+")",fontname='serif',fontsize=lfsiz)
#        #ax.set_xlim(-8e-9,8e-9)
#        ax.minorticks_on()
#        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
#        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
#        fnam=destn+"/"+"hist_step"+string
#        fig.savefig(fnam+".png")
#        fig.savefig(fnam+".svg",format='svg', dpi=1200)
#        fig.savefig(fnam+".eps",format='eps', dpi=1000)  
        return [bin,cum_val,mean]
    def velocity_fft(self,scale,string,desnum,destn,lfsiz,llsiz):
        S_tmax=[]
        T=[]
        fx=[]
        x=[]
        #dirs=self.samdirs[:10]
        rx=[]
        #for sdir in dirs:
        for sdir in self.samdirs:
           max_t=np.loadtxt(sdir+"/max_time.txt")
           for tstep in range(scale*self.p.samrate,int(max_t),self.p.samrate):
               B1=np.loadtxt(sdir+"/"+str(tstep-scale*self.p.samrate)+"cargo.txt")
               B2=np.loadtxt(sdir+"/"+str(tstep)+"cargo.txt")
               fx.append((B2[desnum]-B1[desnum])/(scale*self.p.samrate*self.p.dt))
               x.append(tstep*self.p.dt)
               rx.append(B1[0])
        fx=np.array(fx)
        x=np.array(x)
        n=len(fx)
        dx=self.p.samrate*self.p.dt
        Fk = fft.fft(fx)/n # Fourier coefficients (divided by n)
        nu = fft.fftfreq(n,dx) # Natural frequencies
        Fk = fft.fftshift(Fk) # Shift zero freq to center
        nu = fft.fftshift(nu) # Shift zero freq to center
        fig, ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(nu, np.real(Fk)) # Plot Cosine terms
        ax[0].set_ylabel(r'$Re[F_k]$', size = 'x-large')
        ax[1].plot(nu, np.imag(Fk)) # Plot Sine terms
        ax[1].set_ylabel(r'$Im[F_k]$', size = 'x-large')
        ax[2].plot(nu, np.absolute(Fk)**2) # Plot spectral power
        ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large')
        ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
        fnam=destn+"/"+"fft_vel"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        plt.close()
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(x,fx,'r-')
        fnam=destn+"/"+"vel_"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        plt.close() 
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(x,rx,'b-')
        fnam=destn+"/"+"traject_"+string
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        plt.close()
        
    def mfpt_mot(self,destn):
       [llsiz,lfsiz]=[15,15]
       FPT_array=[]
       for sdir in self.samdirs:
           if os.path.isfile(sdir+"/fpt.txt")*1 ==1:
               fpt=np.loadtxt(sdir+"/fpt.txt")
               FPT_array.append(fpt)
       fig, ax =plt.subplots(figsize=(8,8))
       ax.hist(FPT_array, bins=int(50), align='mid')#[give thermal velocity distribution]
       ax.set_xlabel("fpt (s)",fontname='serif',fontsize=lfsiz)
       ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
       #ax.set_yscale('log')
       ax.minorticks_on()
       ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
       ax.tick_params(axis='both',which='major',length=10,width=1.5 )
       ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
       mean_fpt=sum(FPT_array)/(1.0*len(FPT_array))
       np.savetxt(destn+"/"+"mean_fpt.txt",[mean_fpt])
       fnam=destn+"/"+"fpt_distribution"
       fig.savefig(fnam+".png")
       fig.savefig(fnam+".svg",format='svg', dpi=1200)
       fig.savefig(fnam+".eps",format='eps', dpi=1000)
    def off_analysis(self,destn):
        [llsiz,lfsiz]=[15,15]
        off_time=[]
        for sdir in self.samdirs:
            count=1
            off_file=sdir+"/"+str(count)+"_offtime.txt"
            while os.path.isfile(off_file)*1 ==1:
                ot=np.loadtxt(off_file)
                off_time.append(ot)
                count+=1
        fig, ax =plt.subplots(figsize=(8,8))
        ax.hist(off_time, bins=int(50), align='mid')#[give thermal velocity distribution]
        ax.set_xlabel("off_time (s)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("No. of events",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        mean_off=sum(off_time)/(1.0*len(off_time))
        np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+"off_distribution"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
    def rebinding_theta_distribution(self,destn):
        [llsiz,lfsiz]=[15,20]
        Intensity_matrix=[]
        #243477
        time_arr=[count for count in range(100,50000,1000)]
        bin_size=20
        ang_theta=np.arange(0,pi,bin_size)
        cm=self.cm
        p=self.p
        print("started")
        for count in time_arr:
            theta_arr=[]
            for sdir in self.samdirs:
                max_time=int(np.loadtxt(sdir+"/"+"unbound_time.txt"))
                m=0
                if max_time>m:
                    m=max_time
                rebinding=int(np.loadtxt(sdir+"/"+"rebinding.txt")/p.dt)
                if count<=rebinding:
                    B=np.loadtxt(sdir+"/"+str(max_time)+"cargo.txt")
                    cm.B=B
                    A=np.loadtxt(sdir+"/rebind_anch"+str(count)+".txt")
                    [thet,phi]=cm.cart_polar(A)
                    theta_p=atan2(B[1],B[2])
                    theta_arr.append(thet-theta_p)
                #theta_arr.append(count*pi/1000)
            val,bin0=np.histogram(theta_arr,bins=int(bin_size),range=(0,pi))
            val2=np.divide(val,sum(val)*1.0)
            #for num in range(1,len(val2)-1):
                #val2[num]=val2[num]/sin(bin0[num])
            #cum_val=np.cumsum(val2)
            Intensity_matrix.append(val2)
            
        print(m)
        print("loop complete")
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        bin=np.array(bin)
        fig, ax =plt.subplots(figsize=(8,8))
        #X,Y=np.meshgrid(bin,time_arr)
        #print(Intensity_matrix)
        heatplot = ax.imshow(Intensity_matrix, vmin=0,vmax=np.matrix.max(np.matrix(Intensity_matrix)), cmap='viridis',extent=(min(bin),max(bin),max(time_arr)*self.p.dt,min(time_arr)*self.p.dt), aspect='auto')
        #ax.set_yticklabels(time_arr)
        #ax.set_xticks(np.linspace(min(bin), max(bin)+1, 5))
        fig.colorbar(heatplot)
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #ax.set_xticklabels(bin,rotation=45)
        #ax.ticklabel_format(axis='x', style='sci')
        ax.set_xlabel(r'$\theta$',fontname='serif',fontsize=lfsiz)
        ax.set_ylabel("time",fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        #mean_off=sum(off_time)/(1.0*len(off_time))
        #np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+"heat_map_pdf"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)  

    def bind_rebinding_time_distribution(self,destn):
        [llsiz,lfsiz]=[15,20]
        RB_time=[]
        BN_time=[]
        UB_time=[]
        cm=self.cm
        p=self.p
        bin_size=20
        for sdir in self.samdirs:
            if os.path.isfile(sdir+"/"+"unbound_time.txt"):
                UB_time.append(np.loadtxt(sdir+"/"+"unbound_time.txt")*p.dt)
                RB_time.append(np.loadtxt(sdir+"/"+"rebinding.txt"))
            BN_time.append(np.loadtxt(sdir+"/"+"initial_attachment.txt"))
        val11,bin0=np.histogram(np.log10(BN_time),bins=int(bin_size))
        val21=np.divide(val11,sum(val11)*1.0)
        p=len(bin0)
        bin11=[bin0[num] for num in range(p-1)]
        val12,bin0=np.histogram(np.log10(RB_time),bins=int(bin_size))
        val22=np.divide(val12,sum(val12)*1.0)
        p=len(bin0)
        bin12=[bin0[num] for num in range(p-1)]
        val13,bin0=np.histogram(np.log10(UB_time),bins=int(bin_size))
        val23=np.divide(val13,sum(val13)*1.0)
        p=len(bin0)
        bin13=[bin0[num] for num in range(p-1)]
        
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(bin11,np.cumsum(val21),'r-',label="binding time")
        ax.plot(bin12,np.cumsum(val22),'b-',label="rebinding time")
        ax.plot(bin13,np.cumsum(val23),'g-',label="unbinding time")
        ax.set_xlabel(r"log_{10}(t)",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r"p($t$)",fontname='serif',fontsize=lfsiz)
        #ax.set_xscale('log')
        ax.minorticks_on()
        leg=ax.legend(fontsize=lfsiz-4,frameon=True)
        leg.get_frame().set_edgecolor('k')
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"time_distribution"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)   
        val,bin0=np.histogram(RB_time,bins=int(bin_size))
        val2=np.divide(val,sum(val)*1.0)
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(bin,val2,'ro')
        ax.set_xlabel(r"binding time, $t_b$",fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r"p($t_b$)",fontname='serif',fontsize=lfsiz)
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"rebinding_time"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        close(fig)
        mean_bin=np.sum(BN_time)/(len(BN_time)*1.0)
        s1=np.sqrt(np.sum((BN_time-mean_bin)**2)/(len(BN_time)*1.0))
        mean_ubin=np.sum(UB_time)/(len(UB_time)*1.0)
        s2=np.sqrt(np.sum((UB_time-mean_ubin)**2)/(len(UB_time)*1.0))
        mean_reb=np.sum(RB_time)/(len(RB_time)*1.0)
        s3=np.sqrt(np.sum((RB_time-mean_reb)**2)/(len(RB_time)*1.0))
        print(mean_bin)
        data = {'binding time': mean_bin, 'unbinding time': mean_ubin, 'rebinding_time': mean_reb}
        yerrs=[s1,s2,s3]
        print(yerrs)
        print(len(BN_time))
        names = list(data.keys())
        values = list(data.values())
        fig, ax = plt.subplots(figsize=(10, 10))
        #ax.bar(names, values, yerr=yerrs)
        #print(s1,s2,s3)
        #ax.set_yscale('log')
        ax.set_ylabel("time (s)",fontname='serif',fontsize=lfsiz)
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='y',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"mean_values"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)

    def bind_rebinding_new_distribution(self):
        [llsiz,lfsiz]=[15,20]
        RB_time=[]
        BN_time=[]
        UB_time=[]
        cm=self.cm
        p=self.p
        bin_size=20
        dirs=self.samdirs[:10]
        for sdir in dirs:
        #for sdir in self.samdirs:
            if os.path.isfile(sdir+"/"+"unbound_time.txt"):
                RB_time.append(np.loadtxt(sdir+"/"+"rebinding.txt"))
            BN_time.append(np.loadtxt(sdir+"/"+"initial_attachment.txt"))
            UB_time.append(np.loadtxt(sdir+"/"+"max_time.txt")*p.dt)
        List=[RB_time,BN_time,UB_time]
        CUM_ARR=[]
        BIN_ARR=[]
        MEAN_ARR=[]
        for val_ar in List:
            val,bin0=np.histogram(np.log10(val_ar),bins=int(bin_size))
            p=len(bin0)
            bin=[bin0[num] for num in range(p-1)]
            mean=np.sum(val*bin)
            cum_val=np.cumsum(val)
            CUM_ARR.append(cum_val)
            BIN_ARR.append(bin)
            MEAN_ARR.append(mean)
        return [BIN_ARR,CUM_ARR,MEAN_ARR]
        
    def bind_distribution(self):
        [llsiz,lfsiz]=[15,20]
        BN_time=[]
        cm=self.cm
        p=self.p
        bin_size=20
        dirs=self.samdirs[:10]
        for sdir in dirs:
        #for sdir in self.samdirs:
            BN_time.append(np.loadtxt(sdir+"/"+"initial_attachment.txt"))
        for val_ar in List:
            val,bin0=np.histogram(np.log10(val_ar),bins=int(bin_size))
            p=len(bin0)
            bin=[bin0[num] for num in range(p-1)]
            mean=np.sum(val*bin)
            cum_val=np.cumsum(val)
            CUM_ARR.append(cum_val)
            BIN_ARR.append(bin)
            MEAN_ARR.append(mean)
        return [BIN_ARR,CUM_ARR,MEAN_ARR]
#        val21=np.divide(val11,sum(val11)*1.0)
#        p=len(bin0)
#        bin11=[bin0[num] for num in range(p-1)]
#        val12,bin0=np.histogram(RB_time,bins=int(bin_size))
#        val22=np.divide(val12,sum(val12)*1.0)
#        p=len(bin0)
#        bin12=[bin0[num] for num in range(p-1)]
#        val13,bin0=np.histogram(UB_time,bins=int(bin_size))
#        val23=np.divide(val13,sum(val13)*1.0)
#        p=len(bin0)
#        bin13=[bin0[num] for num in range(p-1)]
#        fig, ax =plt.subplots(figsize=(8,8))
#        ax.plot(bin11,val21,'ro',label="binding time")
#        ax.plot(bin12,val22,'bo',label="rebinding time")
#        ax.plot(bin13,val23,'go',label="unbinding time")
#        ax.set_xlabel(r"time, t",fontname='serif',fontsize=lfsiz)
#        ax.set_ylabel(r"p($t$)",fontname='serif',fontsize=lfsiz)
#        ax.set_xscale('log')
#        ax.minorticks_on()
#        leg=ax.legend(fontsize=lfsiz-4,frameon=True)
#        leg.get_frame().set_edgecolor('k')
#        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
#        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
#        fnam=destn+"/"+"time_distribution23"
#        fig.savefig(fnam+".png")
#        fig.savefig(fnam+".svg",format='svg', dpi=1200)
#        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
#        val,bin0=np.histogram(RB_time,bins=int(bin_size))
#        val2=np.divide(val,sum(val)*1.0)
#        p=len(bin0)
#        bin=[bin0[num] for num in range(p-1)]
#        fig, ax =plt.subplots(figsize=(8,8))
#        ax.plot(bin,val2,'ro')
#        ax.set_xlabel(r"binding time, $t_b$",fontname='serif',fontsize=lfsiz)
#        ax.set_ylabel(r"p($t_b$)",fontname='serif',fontsize=lfsiz)
#        ax.minorticks_on()
#        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
#        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
#        fnam=destn+"/"+"rebinding_time"
#        fig.savefig(fnam+".png")
#        fig.savefig(fnam+".svg",format='svg', dpi=1200)
#        fig.savefig(fnam+".eps",format='eps', dpi=1000)
#        close(fig)
#        mean_bin=np.sum(BN_time)/(len(BN_time)*1.0)
#        s1=np.sqrt(np.sum((BN_time-mean_bin)**2)/(len(BN_time)*1.0))
#        mean_ubin=np.sum(UB_time)/(len(UB_time)*1.0)
#        s2=np.sqrt(np.sum((UB_time-mean_ubin)**2)/(len(UB_time)*1.0))
#        mean_reb=np.sum(RB_time)/(len(RB_time)*1.0)
#        s3=np.sqrt(np.sum((RB_time-mean_reb)**2)/(len(RB_time)*1.0))
#        data = {'binding time': mean_bin, 'unbinding time': mean_ubin, 'rebinding_time': mean_reb}
#        print(data)
#        yerrs=[s1,s2,s3]
#        print(yerrs)
#        print(len(BN_time))
#        names = list(data.keys())
#        values = list(data.values())
#        import pandas as pd
#        df = pd.DataFrame.from_dict(data, orient="index")
#        df.to_csv(destn+"/"+"mean_times.csv")
        #fig, ax = plt.subplots(figsize=(10, 10))
        #ax.bar(['binding time','unbinding time','rebinding_time'], [mean_bin,mean_ubin,mean_reb], yerr=yerrs)
        #print(s1,s2,s3)
        #ax.set_yscale('log')
#        ax.set_ylabel("time (s)",fontname='serif',fontsize=lfsiz)
#        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
#        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#        ax.tick_params(axis='y',which='minor',length=4,width=0.5 )
#        fnam=destn+"/"+"mean_values"
#        fig.savefig(fnam+".png")
#        fig.savefig(fnam+".svg",format='svg', dpi=1200)
#        fig.savefig(fnam+".eps",format='eps', dpi=1000)
    def end_time_theta_distribution(self,destn):
        [llsiz,lfsiz]=[15,20]
        time_arr=[count for count in range(100,8000,100)]
        bin_size=20
        ang_theta=np.arange(0,pi,bin_size)
        cm=self.cm
        p=self.p
        theta_arr=[]
        print("started")
        for sdir in self.samdirs:
            max_time=int(np.loadtxt(sdir+"/"+"unbound_time.txt"))
            m=0
            if max_time>m:
                m=max_time
            rebinding=int(np.loadtxt(sdir+"/"+"rebinding.txt")/p.dt)
            B=np.loadtxt(sdir+"/"+str(max_time)+"cargo.txt")
            cm.B=B
            A=np.loadtxt(sdir+"/"+str(max_time)+"anchor.txt")
            [thet,phi]=cm.cart_polar(A)
            theta_p=atan2(B[1],B[2])
            theta_arr.append(thet-theta_p)
            #theta_arr.append(count*pi/1000)
        val,bin0=np.histogram(theta_arr,bins=int(bin_size),range=(0,pi))
        val2=np.divide(val,sum(val)*1.0)
        #for num in range(1,len(val2)-1):
            #val2[num]=val2[num]/sin(bin0[num])
        cum_val=np.cumsum(val2)
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(bin,val2,'ro')
        #ax.set_yticklabels(time_arr)
        #ax.set_xticks(np.linspace(min(bin), max(bin)+1, 5))
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #ax.set_xticklabels(bin,rotation=45)
        #ax.ticklabel_format(axis='x', style='sci')
        ax.set_xlabel(r'$\theta$',fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r'$P(\theta)$',fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        #mean_off=sum(off_time)/(1.0*len(off_time))
        #np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+"end_time_pdf"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
    def init_time_theta_distribution(self,destn):
        [llsiz,lfsiz]=[15,20]
        time_arr=[count for count in range(100,220041,100)]
        bin_size=20
        ang_theta=np.arange(0,pi,bin_size)
        cm=self.cm
        p=self.p
        theta_arr=[]
        print("started")
        for sdir in self.samdirs:
            max_time=int(np.loadtxt(sdir+"/"+"unbound_time.txt"))
            m=0
            if max_time>m:
                m=max_time
            rebinding=int(np.loadtxt(sdir+"/"+"rebinding.txt")/p.dt)
            B=np.loadtxt(sdir+"/"+str(max_time)+"cargo.txt")
            cm.B=B
            A=np.loadtxt(sdir+"/"+str(0)+"anchor.txt")
            #print(A)
            [thet,phi]=cm.cart_polar(A)
            theta_p=atan2(B[1],B[2])
            theta_arr.append(thet-theta_p)
            #theta_arr.append(thet)
            #theta_arr.append(count*pi/1000)
        val,bin0=np.histogram(theta_arr,bins=int(bin_size),range=(0,pi))
        val2=np.divide(val,sum(val)*1.0)
        #for num in range(1,len(val2)-1):
            #val2[num]=val2[num]/sin(bin0[num])
        cum_val=np.cumsum(val2)
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(bin,val2,'ro')
        #ax.set_yticklabels(time_arr)
        #ax.set_xticks(np.linspace(min(bin), max(bin)+1, 5))
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #ax.set_xticklabels(bin,rotation=45)
        #ax.ticklabel_format(axis='x', style='sci')
        ax.set_xlabel(r'$\theta$',fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r'$P(\theta)$',fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        #mean_off=sum(off_time)/(1.0*len(off_time))
        #np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+"heat_map_pdf"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)

    def run_distri(self):
        Run=[]
        Act_time=[]
        print(len(self.samdirs))
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            B=np.loadtxt(sdir+"/"+str(int(max_t))+"cargo.txt")
            Run.append(B[0])
            Act_time.append(max_t)
        Run=np.multiply(Run,1e6)
        b=max([max(Run),abs(min(Run))])
        Run_val1,Run_bin0=np.histogram(Run,bins=int(20),range=(min(Run),max(Run)))
        p=len(Run_bin0)
        sums=sum(Run_val1)
        Run_val=Run_val1/sums
        cum_val=np.cumsum(Run_val)
        Run_bin=[Run_bin0[num] for num in range(p-1)]
        mean=sum(Run_val*Run_bin)
        return [cum_val,Run_bin,mean]
      
        
    def master_plot_run_length(self,destn,MX,MY,LEG,xlabel,ylabel,plotname):
        [lfsiz,llsiz]=[20,15]
        fig, ax =plt.subplots(figsize=(8,8))
        MX=np.array(MX)
        MY=np.array(MY)
        l=MX.shape
        for num in range(l[0]):
            ax.plot(MX[num],MY[num],color=lines[num], linestyle='dashed', marker='o',
     markerfacecolor=lines[num], markersize=3,label=LEG[num])
        ax.set_xlabel(xlabel,fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(ylabel,fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        #ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.legend(fontsize=lfsiz-4,frameon=True)
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+plotname
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
    def act_time_distri(self):
        Act_time=[]
        print(len(self.samdirs))
        for sdir in self.samdirs:
            max_t=np.loadtxt(sdir+"/max_time.txt")
            #B=np.loadtxt(sdir+"/"+str(int(max_t))+"cargo.txt")
            Act_time.append(max_t*self.p.dt)
        #Run=np.multiply(Run,1e6)
        b=max([max(Act_time),abs(min(Act_time))])
        Run_val1,Run_bin0=np.histogram(Act_time,bins=int(20),range=(min(Act_time),max(Act_time)))
        p=len(Run_bin0)
        sums=sum(Run_val1)
        Run_val=Run_val1/sums
        cum_val=np.cumsum(Run_val)
        Run_bin=[Run_bin0[num] for num in range(p-1)]
        mean=sum(Run_val*Run_bin)
        return [cum_val,Run_bin,mean]
        
    def average_bound(self):
        Bound_number=[]
        Time=[]
        Error=[]
        #time_arr=[int(np.exp(count)/1000)*1000 for count in range(0,19)]
        time_arr=[int(np.exp(count)/1000)*1000 for count in np.linspace(7.5,18.7,30)]
        #time_arr=[int(np.exp(count)/1000)*1000 for count in np.linspace(7.5,16.0,16)]
        for tstep in time_arr:
            tbound=[]
            for sdir in self.samdirs:
                max_t=np.loadtxt(sdir+"/max_time.txt")
                if int(max_t)>tstep:
                    bin_stat=np.loadtxt(sdir+"/"+str(int(tstep))+"bind_stat.txt")
                    tbound.append(np.count_nonzero(bin_stat))
            tbound=np.array(tbound)
            mu=np.mean(tbound)
            Bound_number.append(mu)
            Error.append(np.sqrt(np.sum((tbound-mu)**2)/(len(tbound)*1.0)))
            Time.append(tstep*self.p.dt)
        return [Time,Bound_number,Error]
    def mot_num_access_region(self):
        access_number=[]
        Time=[]
        Error=[]
        #time_arr=[int(np.exp(count)/1000)*1000 for count in range(0,19)]
        time_arr=[int(np.exp(count)/1000)*1000 for count in np.linspace(7.5,18.7,30)]
        #time_arr=[int(np.exp(count)/1000)*1000 for count in np.linspace(7.5,16.0,16)]
        for tstep in time_arr:
            tbound=[]
            for sdir in self.samdirs:
                max_t=np.loadtxt(sdir+"/max_time.txt")
                if int(max_t)>tstep:
                    bin_stat=np.loadtxt(sdir+"/"+str(int(tstep))+"bind_stat.txt")
                    B=np.loadtxt(sdir+"/"+str(tstep)+"cargo.txt")
                    self.cm.B=B
                    A=np.loadtxt(sdir+"/"+str(tstep)+"anchor.txt")
                    #bind=np.loadtxt(sdir+"/"+str(count)+"bind_stat.txt")
                    #theta_p=atan2(B[1],B[2])
                    l=A.shape
                    count=0
                    for num in range(l[0]):
                        anch=A[num]
                        #[thet,phi]=cm.cart_polar(anch)
                        #abs_t=abs(thet-theta_p)
                        #if abs_t<
                        rmin=np.sqrt(anch[1]**2+anch[2]**2)
                        if rmin<self.p.Lmot:
                            count+=1                          
                    tbound.append(count)
            tbound=np.array(tbound)
            mu=np.mean(tbound)
            access_number.append(mu)
            Error.append(np.sqrt(np.sum((tbound-mu)**2)/(len(tbound)*1.0)))
            Time.append(tstep*self.p.dt)
        return [Time,access_number,Error]
    def master_plot_bound(self,destn,MX,MY,MERR,LEG,xlabel,ylabel,plotname):
        [lfsiz,llsiz]=[20,15]
        fig, ax =plt.subplots(figsize=(8,8))
        MX=np.array(MX)
        MY=np.array(MY)
        MERR=np.array(MERR)
        print(MERR)
        l=MX.shape
        for num in range(l[0]):
            ax.errorbar(MX[num],MY[num],yerr=MERR[num],color=lines[num], linestyle='dashed', marker='o',
     markerfacecolor=lines[num], markersize=6,label=LEG[num])
        ax.set_xlabel(xlabel,fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(ylabel,fontname='serif',fontsize=lfsiz)
        ax.set_xscale('log')
        #ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.legend(fontsize=lfsiz-4,frameon=True,loc='upper left')
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+plotname
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
        
    def heatmap_theta_distribution(self,destn):
        [llsiz,lfsiz]=[15,20]
        Intensity_matrix=[]
        #time_arr=[int(np.exp(count)/1000)*1000 for count in range(18,6,-2)]
        time_arr=[int(2**count/1000)*1000 for count in range(30,9,-2)]
        bin_size=20
        ang_theta=np.arange(0,pi,bin_size)
        cm=self.cm
        p=self.p
        for count in time_arr:
            theta_arr=[]
            for sdir in self.samdirs:
                max_time=int(np.loadtxt(sdir+"/"+"max_time.txt"))
#                m=0
#                if max_time>m:
#                    m=max_time
                #rebinding=int(np.loadtxt(sdir+"/"+"rebinding.txt")/p.dt)
                if count<=max_time:
                    B=np.loadtxt(sdir+"/"+str(count)+"cargo.txt")
                    bm=sqrt(B[1]**2+B[2]**2)
                    cm.B=B
                    A=np.loadtxt(sdir+"/"+str(count)+"anchor.txt")
                    bind=np.loadtxt(sdir+"/"+str(count)+"bind_stat.txt")
                    theta_p=atan2(B[1],B[2])
                    l=A.shape
                    for num in range(l[0]):
                        an=A[num]
                        if int(bind[num])!=0:
                            #[thet,phi]=cm.cart_polar(anch)
                            anm=sqrt((an[1]-B[1])**2+(an[2]-B[2])**2+(an[0]-B[0])**2)
                            theta=acos((B[1]*(an[1]-B[1])+B[2]*(an[2]-B[2]))/(bm*anm))
                            theta_arr.append(theta)
            val,bin0=np.histogram(theta_arr,bins=int(bin_size),range=(0,pi))
            val2=np.divide(val,sum(val)*1.0)
            #cum_val=np.cumsum(val2)
            Intensity_matrix.append(val2)
        print("loop complete")
        p=len(bin0)
        bin=[bin0[num] for num in range(p-1)]
        bin=np.array(bin)
        fig, ax =plt.subplots(figsize=(8,8))
        #X,Y=np.meshgrid(bin,time_arr)
        #print(Intensity_matrix)
        heatplot = ax.imshow(Intensity_matrix, vmin=0,vmax=np.matrix.max(np.matrix(Intensity_matrix)), cmap='viridis',extent=(min(bin),max(bin),np.log10(min(time_arr)*self.p.dt),np.log10(max(time_arr)*self.p.dt)), aspect='auto')
        #ax.set_yticklabels(time_arr)
        #ax.set_xticks(np.linspace(min(bin), max(bin)+1, 5))
        fig.colorbar(heatplot)
        #ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #ax.set_xticklabels(bin,rotation=45)
        #ax.ticklabel_format(axis='x', style='sci')
        ax.set_xlabel(r'$\theta$',fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r'$log_{10}(t)$',fontname='serif',fontsize=lfsiz)
        #ax.set_yscale('log')
        #ax.grid(b=None) #switches off gridlines
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        #mean_off=sum(off_time)/(1.0*len(off_time))
        #np.savetxt(destn+"/"+"mean_off_time.txt",[mean_off])
        fnam=destn+"/"+"heat_map_pdf"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000) 

    def mean_hight(self,destn):
        [llsiz,lfsiz]=[15,20]
        time_arr=[int(2**count/1000)*1000 for count in range(27,9,-1)]
        cm=self.cm
        p=self.p
        hight=[]
        for count in time_arr:
            hg=0
            cout=0
            for sdir in self.samdirs:
                max_time=int(np.loadtxt(sdir+"/"+"max_time.txt"))
                if count<=max_time:
                    B=np.loadtxt(sdir+"/"+str(count)+"cargo.txt")
                    hg+=np.sqrt(B[1]**2+B[2]**2)
                    cout+=1
            hight.append(hg/(cout*1.0))
        fig, ax =plt.subplots(figsize=(8,8))
        ax.plot(time_arr,hight,'r-')
        ax.set_xlabel('time',fontname='serif',fontsize=lfsiz)
        ax.set_ylabel(r'$r_{min}$',fontname='serif',fontsize=lfsiz)
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in',which='both', labelsize=llsiz)
        ax.tick_params(axis='both',which='major',length=10,width=1.5 )
        ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
        fnam=destn+"/"+"hight"
        fig.savefig(fnam+".png")
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        fig.savefig(fnam+".eps",format='eps', dpi=1000)
   # def step_distribution_subtract(self,scale,string,desnum,destn,lfsiz,llsiz):
             
#    def plot_theta_phi(tots):
#            fig, ax =plt.subplots(figsize=(8,8))
#            for s_num in range(0,tots+10):
#                max_t=np.loadtxt(main_dire+"/"+str(s_num)+"/max_time.txt")
#                for tstep in range(0,int(max_t),p.samrate):
#                    A=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"anchor.txt")
#                    B=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"cargo.txt")
#                    bind_stat=np.loadtxt(main_dire+"/"+str(s_num)+"/"+str(tstep)+"bind_stat.txt")
#                    cm.B=B
#                    col=['b','r']
#                    for num in range(p.N):
#                        [thet,phi]=cm.cart_polar(A[num])
#                        ax.plot(thet,phi,marker='.',color=col[int(bind_stat[num])])
#                        
#            ax.set_xlabel(r'$\theta$',fontname='serif',fontsize=20)
#            ax.set_ylabel(r'$\phi$',fontname='serif',fontsize=20)
#            ax.minorticks_on()
#            ax.tick_params(axis='both', direction='in',which='both', labelsize=15)
#            ax.tick_params(axis='both',which='major',length=10,width=1.5 )
#            ax.tick_params(axis='both',which='minor',length=4,width=0.5 )
#            fig.savefig(destn+"/theta_phi_10.png")
         
#class main:
#    def __init__(self,simname,unique):
#        self.simname=simname
#        self.unique=unique
#        
#class directory(main):
#    def create
    
#class plotter:
#    def __init__(self):
#        self.simname=simname
#        self.unique=unique
#        self.dire=os.getenv("HOME")+"/3dtransport/test/data/"+str(simname)+"/"+str(unique)+"/"+str(sample)+"/"
#        self.main_dire=os.getenv("HOME")+"/3dtransport/test/data/"+str(simname)+"/"+str(unique)
#        self.dest=os.getenv("HOME")+"/3dtransport/test/data/results"+str(simname)+"/"
#        self.destn=dest+str(unique)+"/"+str(sample)
#        self.max_iter=np.loadtxt(dire+"max_time.txt")
#        self.max_iter=int(max_iter)
#        c=distinct_colors()
#        self.lines=c.color()
#        self.f_in=self.dire+"../"+str(self.simname)+".txt"
#        self.p=Params(self.f_in)
#    def create_dire():
#        if os.path.isdir(self.dest)*1 ==0:
#            os.mkdir(self.dest)
#        if os.path.isdir(self.dest+"/"+str(self.unique))*1 ==0:
#            os.mkdir(self.dest+"/"+str(self.unique))
#        if os.path.isdir(self.dest+"/"+str(self.unique)+"/"+str(self.sample))*1 ==0:
#            os.mkdir(self.dest+"/"+str(self.unique)+"/"+str(self.sample))

