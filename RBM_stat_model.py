import cPickle
import numpy as np
import theano
import theano.tensor as T
import Tools as tools
import argparse
import math as m
import os

#-------------------------------------------------

class Stat_model(object):

    """ Class that builds a restricted Boltzmann Machine """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,n_v,n_h,T,SimPar,W,b,c):
       
        """ Constructor """
        
        self.MCS = 100000
        self.n_v = n_v
        self.n_h = n_h
        self.N = self.n_h + self.n_v
        self.T = T
        self.W = W.eval()
        self.b = b.eval()
        self.c = c.eval()
        
        self.e = 0
        self.m = 0

        self.E  = 0
        self.E2 = 0
        self.M  = 0
        self.M2 = 0
        self.M4 = 0
        self.Su = 0
        self.Cv = 0
        
    
    def setSpins(self,spins):

        v_state = spins[0:self.n_v]
        h_state = spins[self.n_v:self.N]

        return [v_state,h_state]


    def energy(self, v_state, h_state):
       
        en = np.dot(h_state,np.dot(v_state,self.W))
        en += np.dot(h_state,self.c)
        en += np.dot(v_state,self.b)

        self.e = -1.0*en

    def magnetization(self, v_state, h_state):
        
        m = 0
        for i in range(self.n_h):
            m += h_state[i]
        for j in range(self.n_v):
            m += v_state[j] 

        self.m = abs(m)

    def record(self):
        
        self.E  += self.e
        self.E2 += self.e * self.e
        self.M  += self.m
        self.M2 += self.m * self.m
        self.M4 += self.m * self.m * self.m * self.m
    
    def measure(self):

        #self.E /= 1.0 * self.N * self.MCS
        #self.M /= 1.0 * self.N * self.MCS
        self.Cv = self.E2/(1.0*self.MCS) - self.E*self.E/(1.0*self.MCS*self.MCS)
        self.Cv /= (self.T*self.T*1.0*self.N)

        self.Su = self.M2/(1.0*self.MCS) - self.M * self.M/(1.0*self.MCS*self.MCS)
        self.Su /= (self.T*1.0*self.N)
    
    def output(self,fileName):
        
        out = open(fileName,'w')
        
        out.write('%f' % self.T)
        out.write(' ')
        out.write('%f' % (self.E/(1.0 * self.N * self.MCS)))
        out.write(' ')
        out.write('%f' % (self.M/(1.0 * self.N * self.MCS)))
        out.write(' ')
        out.write('%f' % self.Cv)
        out.write(' ')
        out.write('%f' % self.Su)
        out.write('\n')
        
        out.close()





