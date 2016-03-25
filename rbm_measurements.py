import cPickle
import numpy as np
import theano
import argparse
import utilities.network as NET

#-------------------------------------------------

class rbmMeasure(object):

    """ Class that builds a restricted Boltzmann Machine """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,Network):
       
        """ Constructor """
       
        self.n_v = Network.infos['Visible Units'] 
        self.n_h = Network.infos['Hidden Units']
        self.N = self.n_h + self.n_v
        self.L = int(np.sqrt(self.n_v)) 
        self.W = Network.infos['Weights'].eval()
        self.b = Network.infos['V Bias'].eval()
        self.c = Network.infos['H Bias'].eval()
        
        self.e = 0.0
        self.m = 0.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_energy(self, v_state, h_state):
        
        """ Compute the energy """

        en = np.dot(h_state,np.dot(v_state,self.W))
        en += np.dot(h_state,self.c)
        en += np.dot(v_state,self.b)

        self.e = -1.0*en

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_magnetization(self, v_state, h_state):
        
        """ Compute the magnetization """

        m = 0.0

        for i in range(self.n_h):
            m += h_state[i]
        
        for j in range(self.n_v):
            m += v_state[j] 

        self.m = abs(m)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def record(self,outputFile):
        
        """ Calculate observables and write them on file """

        E  = self.e
        E2 = self.e * self.e
        M  = abs(self.m)
        M2 = self.m * self.m
        
        outputFile.write('%.5f  ' % E)
        outputFile.write('%.5f  ' % E2)
        outputFile.write('%.5f  ' % M)
        outputFile.write('%.5f  ' % M2)
        outputFile.write('\n')
 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    def build_inputNames(self,Network):
        
        name = 'data/samples/L'
        name += str(self.L)
        name += '/'
        name_H = name
        name_V = name
        name_V += NET.build_fileName(Network,'visible_samples')
        name_H += NET.build_fileName(Network,'hidden_samples')

        return [name_V,name_H]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
    def build_outputName(self,Network):
        
        name = 'data/measurements/L'
        name += str(self.L)
        name += '/'
        name += NET.build_fileName(Network,'RBM_measures')
        return name


