from math import sin,cos, sqrt,atan2,tan
import numpy as np
import matplotlib.pyplot as plt
import os
from parameter_reader import *
from os import getcwd, path, rename, mkdir, path
from random import random, seed, normalvariate
#parameters

p=Params("input.txt")
factr=sqrt(2*p.D*p.dt)
vdx=p.v*p.dt


#state=1 active, state=0 diffusion
def run():
    [x,y]=[0,0]
    state=1
    thet=atan2(y,x)
    print(thet)
    x_a=[]
    y_a=[]
    noise=np.array([normalvariate(0,1.0) for i in range(2)])
    for tstep in range(0,10):
        if tstep%1000==0:
            x_a.append(x)
            y_a.append(y)
        time=tstep*p.dt
        thet=atan2(y,x)
        rad=sqrt(x**2+y**2)
        if (abs(thet-p.th0)<1e-5 or abs(thet)<1e-5) and state==0:
            state=1
        if state==1 and p.q<random():
            state=0
        if state==0:
            noise=np.array([normalvariate(0,1.0) for i in range(2)])
            x+=factr*noise[0]
            y+=factr*noise[1]
        if state==1:
            x+=vdx*cos(thet)
            y+=vdx*sin(thet)
        if rad>p.L:
            x1=np.linspace(0,p.L,100)
            y1=[0 for i in range(len(x1))]
            y2=tan(p.th0)*x1
            plt.plot(x1,y1,'r--')
            plt.plot(x1,y2,'r--')
            plt.plot(x_a,y_a,'k-')
            plt.savefig("test.png")
            return time
        x1=np.linspace(0,p.L,100)
        y1=[0 for i in range(len(x1))]
        y2=tan(p.th0)*x1
        plt.plot(x1,y1,'r--')
        plt.plot(x1,y2,'r--')
        plt.plot(x_a,y_a,'k-')
        plt.savefig("test.png")
    return time

time=run()
print(time)
