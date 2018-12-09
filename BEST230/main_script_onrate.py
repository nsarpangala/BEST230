
#Shree durga devi namaha
#course project computational modeling for interdisciplinary bio
from math import sin,cos, sqrt,atan2,tan
import numpy as np
import matplotlib.pyplot as plt
import os
from parameter_reader import *
from os import getcwd, path, rename, mkdir, path
from random import random, seed, normalvariate
from wedge_transport_class import *
import pandas as pd
from data_aqu import *
from joblib import Parallel, delayed
import multiprocessing
print(sys.argv[1])
f_in="inputs/"+str(sys.argv[1])
#f_in="inputs/test06.txt"
w=wedge_transport(f_in)
d=data_aq(w)
d.mk_main_dir()
#COPY FILES TO DIR
d.cp_uniqdir(f_in)
d.cp_uniqdir("main_script_onrate.py")
d.cp_uniqdir("wedge_transport_class.py")
d.cp_uniqdir("parameter_reader.py")

print(w.p.dt)
def run(sample):
    w.reset()
    w.intialize()
    w.polar2cart()
    w.time=0
    w.stat=0
    df = pd.DataFrame([[w.time,w.x,w.y,w.stat,w.r,w.theta]],columns=['time','x','y','stat','r','theta'])
    dct=0
    while w.time<w.p.max_time:
        w.diffuse()
        w.wrap_theta()
        w.check_state_onrate()
        w.time+=w.p.dt
        if int(w.stat)==1:
            [stack,dct]=w.ballistic(dct)
            df2 = pd.DataFrame(stack,columns=['time','x','y','stat','r','theta'])
            df = df.append(df2, ignore_index=True)
            #w.theta=w.p.angle*random()
            #w.polar2cart()
        if w.r<w.p.delta:
#            df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat,w.r,w.theta]],columns=['time','x','y','stat','r','theta'])
#           df = df.append(df2, ignore_index=True)
            break
        if int(w.time/w.p.dt)%w.p.samrate==0 and int(int(w.time/w.p.dt)/w.p.samrate)==(dct+1):
            dct+=1
            df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat,w.r,w.theta+(w.p.angle*w.count)]],columns=['time','x','y','stat','r','theta'])
            df = df.append(df2, ignore_index=True)
    df.to_csv(d.uniqdir+'/'+str(sample)+'.csv')
    return w.time
#jobn=int(sys.argv[3])
fpt_arr=[]
#for sample in range(jobn*10,(jobn+1)*10):   
#    fpt=run(sample)
#    fpt_arr.append(fpt)
    
    
num_cores = int(sys.argv[2])     
fpt_arr = Parallel(n_jobs=num_cores)(delayed(run)(sample) for sample in range(int(w.p.samplesiz)))

df = pd.DataFrame(fpt_arr,columns=['fpt'])
df.to_csv(d.uniqdir+'/fpt.csv')  
