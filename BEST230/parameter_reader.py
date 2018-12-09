label_attr_map = {
    "D": ["D", float],
    "v":["v",float],
    "R":["R",float],
    "dt":["dt",float],
    "N":["N",float],
    "delta":["delta",float],
    "max_time":["max_time",float],
    "samplesiz":["samplesiz",int],
    "samrate":["samrate",int],
    "rate_const":["rate_const",float],
    "onrate":["onrate",float],
    "mtr":["mtr",float],
    "simname":["simname",str],
}

import numpy as np
from math import pi
class Params(object):
    def __init__(self, input_file_name):
        with open(input_file_name, 'r') as input_file:
            for line in input_file:
                
                row = line.split(",")
        
                data = row[2]  # rest of row is data list
                label = row[1]
                

                attr = label_attr_map[label][0]
                datatypes = label_attr_map[label][1]

                value= (datatypes(data))
                self.__dict__[attr] = value
        self.angle=(2*pi)/(self.N*1.0)
        l=self.simname
        self.simname=l.strip()
#The code to use this class is.
##params = Params('input.txt')

#Following are the codes to test whether this class is working correctly
##print params.N
##print params.sampleSiz
##print params.kmot
##print params.velMean
##print params.velStdev
##print params.dt
##print params.tmax
##print params.samrate
##print params.Ktrap
##print params.load
##print params.Pon
##print params.dx
##print params.Lmot
##print params.Fs
##print params.w
##print params.radius
##print params.eps
##print params.theta
##print params.xintercept
##print params.beadposx
##print params.beadposy
##print params.viscosity
##print params.temp



