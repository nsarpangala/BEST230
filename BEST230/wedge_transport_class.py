from os import getcwd, path, rename, mkdir, path
import numpy as np
import sys
from parameter_reader import *
from math import sqrt, pi, sin, cos, exp, acos, atan, atan2
from random import random, seed, normalvariate
from pdb import *

class wedge_transport:
    def __init__(self,f_in):
        p=Params(f_in)
        self.p=p
        self.health=0
        self.factr=sqrt(2*p.D*p.dt)
        self.vdx=p.v*p.dt
        self.nana=float('nan')
        self.x=self.nana
        self.y=self.nana
        self.r=self.nana
        self.time=self.nana
        self.theta=self.nana
        self.stat=0
        self.tstep=0
        self.count=0
    def reset(self):
        self.x=self.nana
        self.y=self.nana
        self.r=self.nana
        self.time=self.nana
        self.theta=self.nana
        self.stat=0
        self.tstep=0
        self.count=0
    def polar2cart(self):
        self.x=self.r*cos(self.theta)
        self.y=self.r*sin(self.theta)
    def cart2polar(self):
        self.r=np.sqrt(self.x**2+self.y**2)
        self.theta=atan2(self.y,self.x)
    def intialize(self):
        self.r=self.p.R
        self.theta=self.p.angle*random()
    def wrap_theta(self):
        if self.theta>self.p.angle:
            self.theta=self.theta-self.p.angle
            self.count+=1
        if self.theta<0:
            self.theta=self.p.angle+self.theta
            self.count-=1
        self.polar2cart()
    def check_pos(self,temp_r):
        if temp_r<self.p.R:
            return 1
        else:
            return 0
    def diffuse(self):
        suc=0
        while suc==0:
            noise=[normalvariate(0,1.0) for i in range(2)]
            temp_r=sqrt((self.x+self.factr*noise[0])**2+(self.y+self.factr*noise[1])**2)
            suc=self.check_pos(temp_r)
        self.x+=self.factr*noise[0]
        self.y+=self.factr*noise[1]
        self.cart2polar()
#        if 
    def diffuse_test(self):
        noise=[normalvariate(0,1.0) for i in range(2)]
        self.x+=self.factr*noise[0]
        self.y+=self.factr*noise[1]
        self.cart2polar()
    def check_state(self):
        if self.theta>self.p.angle:
            self.stat=1
            self.theta=self.p.angle
        if  self.theta<0:
            self.stat=1
            self.theta=0
            
            
##    def check_state_onrate(self):
##        if self.theta>self.p.angle:
##            if random()<self.p.onrate*self.p.dt:
##                self.stat=1
##            self.theta=self.p.angle
##        if  self.theta<0:
##            if random()<self.p.onrate*self.p.dt:
##                self.stat=1
##            self.theta=0
        
    def check_state_onrate(self):
        if abs(self.theta-self.p.angle)<abs(self.p.mtr/self.r) or abs(self.theta)<abs(self.p.mtr/self.r):
            if random()<self.p.onrate*self.p.dt:
                self.stat=1
    def ballistic_old(self):
        #btime=0.1
        btime=np.random.exponential(scale=self.p.rate_const)
        self.r-=btime*self.p.v
        self.stat=0
        self.polar2cart()
        return btime
    
    def ballistic(self,dc):
        btime=np.random.exponential(scale=self.p.rate_const)
        tot=int(btime//self.p.dt)
        bas=int(self.time//self.p.dt)
        x=[]
        y=[]
        r=[]
        th=[]
        st=[]
        tim=[]
        cur_r=self.r
        for tstep in range(tot):
            if (int(tstep+bas)%self.p.samrate==0) and int(int(tstep+bas)//self.p.samrate)==(dc+1):
                dc+=1
                self.r=cur_r-tstep*self.p.dt*self.p.v
                self.polar2cart()
                x.append(self.x)
                y.append(self.y)
                r.append(self.r)
                st.append(1)
                th.append(self.theta+self.p.angle*self.count)
                tim.append(int(tstep+bas)*self.p.dt)
                if self.r<self.p.delta:
                    break
        self.stat=0
        self.time+=tot*self.p.dt
        return [np.column_stack((tim,x,y,st,r,th)),dc]
    
    
    
