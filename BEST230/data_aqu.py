import pandas as pd
import numpy as np
import os
import shutil as sh
import time
import csv
from random import random, randint
class data_aq:
    def __init__(self,main):
        self.m=main
        self.scriptdir=os.getcwd()
    def mk_main_dir(self):
        #time.sleep(int(60*randint(1,100)))
        self.workingfolder="%s/data/%s/%s/" % (os.getenv("HOME"),"uttam_project",self.m.p.simname)
        if os.path.isdir(self.workingfolder)*1 ==0:
            os.mkdir(self.workingfolder)
        try:
            uniq=str(int(time.time()))+"/"
            self.uniqdir=self.workingfolder+uniq
            if os.path.isdir(self.uniqdir)*1 ==0:
                os.mkdir(self.uniqdir)
                print("Data being saved to:",str(self.uniqdir))
        except OSError:
            print("here")
            time.sleep(int(60*randint(1,100)))
            uniq=str(int(time.time()))+"/"
            self.uniqdir=self.workingfolder+uniq
            if os.path.isdir(self.uniqdir)*1 ==0:
                os.mkdir(self.uniqdir)
                print("Data being saved to:",str(self.uniqdir))
    def cp_uniqdir(self,filename):
        sh.copy(filename,self.uniqdir)
    def mk_sample_dir(self,sample):
        self.sample_path=self.uniqdir+str(sample)
        os.mkdir(self.sample_path)

    def save(self,mc,time):
        base=self.sample_path+"/"+str(time)
        np.savetxt(base+"cargo.txt",mc.B)
        np.savetxt(base+"anchor.txt",mc.A)
        np.savetxt(base+"head.txt",mc.H)
        np.savetxt(base+"force.txt",mc.F)
        np.savetxt(base+"bind_stat.txt",mc.bind_stat)

    def save_var(self,name,var):
        base=self.sample_path
        np.savetxt(base+"/"+name+".txt",var)
        
    def create_logfile(self,start_time_float,end_time_float,name):
        start_time=time.strftime("%Y-%m-%d %H:%M:%S", start_time_float)
        end_time=time.strftime("%Y-%m-%d %H:%M:%S", end_time_float)
        simulation_time_float=time.mktime(end_time_float)-time.mktime(start_time_float)

        sec=np.mod(simulation_time_float,60.0)
        minute=int(np.mod(simulation_time_float,3600.0)/60)
        hours=int(np.mod(simulation_time_float,24*3600.0)/3600.0)
        days=int(simulation_time_float/(24*3600.0))

        simulation_time="%i %i:%i:%i" %(days,hours, minute, sec)
        uniqdir=os.getcwd()
        if isinstance(name, int):
            filename=self.sample_path+"/"+str(name)+".log"
        elif isinstance(name, str):
            filename=self.uniqdir+"/"+str(name)+".log"

        logf=open(filename,'w')
        str1='Start time:  '+str(start_time)
        str2='End time:  '+str(end_time)
        str3='Simulation time: (d h:m:s) '+str(simulation_time)
        logf.write(str1+'\n')
        logf.write(str2+'\n')
        logf.write(str3+'\n')
        logf.close()    
    

#UNTESTED BELOW
    def save_data_ini(self,mc):
        np.savetxt("testing_data/ini_bead.txt",mc.B)
        np.savetxt("testing_data/ini_anchor.txt",mc.A)
        np.savetxt("testing_data/ini_head.txt",mc.H)
        np.savetxt("testing_data/ini_force.txt",mc.F)
        np.savetxt("testing_data/bind_stat.txt",mc.bind_stat)

    def load_data(self,mc):
       mc.B=np.loadtxt("testing_data/ini_bead.txt")
       mc.A=np.loadtxt("testing_data/ini_anchor.txt")
       mc.H= np.loadtxt("testing_data/ini_head.txt")
       mc.F= np.loadtxt("testing_data/ini_force.txt")
       mc.bind_stat=np.loadtxt("testing_data/bind_stat.txt")