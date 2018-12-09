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
from data_aqu import *
f_in="input.txt"
w=wedge_transport(f_in)

#Print all inputs
print("D",w.p.D)
print("v",w.p.v)
print("R",w.p.R)
print("dt",w.p.dt)
print("N",w.p.N)
print("delta",w.p.delta)
print("wedge angle",w.p.angle)
print("sample size",w.p.samplesiz)
print("samrate",w.p.samrate)
print("max_time",w.p.max_time)
print("initializing")
w.intialize()
print("intial point is")
print("radius =",w.r)
print("theta =",w.theta)
print("diff_fact=",w.factr)
w.polar2cart()
print("x",w.x)
print("y",w.y)
print("reset")
w.reset()
print("after reset")
print("radius =",w.r)
print("theta =",w.theta)
print("x",w.x)
print("y",w.y)
print("initialize again")
w.intialize()
print("intial point is")
print("radius =",w.r)
print("theta =",w.theta)
print("diff_fact=",w.factr)
print("converting to cartesian")

w.polar2cart()
print("x",w.x)
print("y",w.y)

print("x=3.0e-6, y=-1.0e-6")
w.x=3.0e-6
w.y=-1.0e-6
w.cart2polar()
print("r",w.r)
print("theta",w.theta)

print("checking diffuse")
print("x,y",[w.x,w.y])
w.diffuse()
print("x,y",[w.x,w.y])
w.cart2polar()
print("checking state switch")
print("theta",w.theta,"state",w.stat)
w.check_state()
print("state",w.stat)
w.theta=w.p.angle/2.0
w.stat=0
print("theta",w.theta,"state",w.stat)
w.check_state()
print("state",w.stat)

w.theta=(w.p.angle+0.02)
w.stat=0
print("theta",w.theta,"state",w.stat)
w.check_state()
print("state",w.stat)

print("checking ballistic motion")
print("current radius",w.r,"current state",w.stat)
w.ballistic()
print("radius after ballistic:", w.r,"state:",w.stat)

print("data aquisition")
d=data_aq(w)
d.mk_main_dir()
#COPY FILES TO DIR
d.cp_uniqdir(f_in)
d.cp_uniqdir("main_script.py")
d.cp_uniqdir("wedge_transport_class.py")
d.cp_uniqdir("parameter_reader.py")



x=4.0
y=2.0
tstep=10
#df = pd.DataFrame({'tstep':tstep,'x':x,'y':y})
#df2 = pd.DataFrame({'tstep':tstep,'x':x,'y':y})
df = pd.DataFrame([[tstep,x,y]],columns=['tstep','x','y'])
df2 = pd.DataFrame([[tstep,x,y]],columns=['tstep','x','y'])

df = df.append(df2)
sample=1
df.to_csv(d.uniqdir+'/'+str(sample)+'.csv')

da = pd.DataFrame([[tstep,x,y]],columns=['tstep','x','y'])
