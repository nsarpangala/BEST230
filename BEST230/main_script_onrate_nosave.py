
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
f_in="inputs/"+str(sys.argv[1])
w=wedge_transport(f_in)
d=data_aq(w)
d.mk_main_dir()
#COPY FILES TO DIR
d.cp_uniqdir(f_in)
d.cp_uniqdir("main_script_onrate_nosave.py")
d.cp_uniqdir("wedge_transport_class.py")
d.cp_uniqdir("parameter_reader.py")

print(w.p.dt)
def run(sample):
    w.reset()
    w.intialize()
    w.polar2cart()
    w.time=0
    w.stat=0
   #df = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
    while w.time<w.p.max_time:
        w.diffuse()
        w.wrap_theta()
        w.check_state_onrate()
        w.time+=w.p.dt
        if int(w.stat)==1:
            df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
            df = df.append(df2, ignore_index=True)
            btime=w.ballistic()
            w.time+=btime
            #if int(w.time/w.p.dt)%w.p.samrate==0:
                #df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
                #df = df.append(df2, ignore_index=True)
            #w.theta=w.p.angle*random()
            #w.polar2cart()
        if w.r<w.p.delta:
            #df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
            #df = df.append(df2, ignore_index=True)
            break
        #if int(w.time/w.p.dt)%w.p.samrate==0:
            #df2 = pd.DataFrame([[w.time,w.x,w.y,w.stat]],columns=['time','x','y','stat'])
            #df = df.append(df2, ignore_index=True)
    #df.to_csv(d.uniqdir+'/'+str(sample)+'.csv')
    return w.time

fpt_arr=[]
for sample in range(int(w.p.samplesiz)):   
    fpt=run(sample)
    fpt_arr.append(fpt)

df = pd.DataFrame(fpt_arr,columns=['fpt'])
df.to_csv(d.uniqdir+'/fpt.csv')  
