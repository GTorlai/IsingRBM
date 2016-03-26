import numpy as np
import utilities.network as NET
import argparse
from pylab import loadtxt
#-------------------------------------------------

class IsingMeasure(object):

    """ Class that builds a restricted Boltzmann Machine """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,Network,D):
        
        """ Contructor """

        self.D = D
        self.N = Network.infos['Visible Units']
        self.L = int(np.sqrt(self.N))
        self.Neighbors = np.zeros((self.N,2*self.D))
        self.e = 0.0
        self.m = 0.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_lattice(self):
        
        """ Build nearest neightbors connections """

        for i in range(self.N):
            for j in range(self.D):
                if ((i+self.L**j)%(self.L**(j+1)) < self.L**j):
                    self.Neighbors[i,j] = i+self.L**j - self.L**(j+1)
                else:
                    self.Neighbors[i,j] = i+self.L**j
        
                self.Neighbors[self.Neighbors[i,j],j+self.D] = i

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    def get_energy(self,spins):

        """ Compute the energy """

        self.e = 0.0
        for i in range(self.N):
            for j in range(2*self.D):
                self.e += -(2*spins[i]-1)*(2*spins[self.Neighbors[i,j]]-1)

        self.e /= 2.0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    def get_magnetization(self,spins):

        """ Compute the magnetization """

        self.m = 0.0
        
        for i in range(self.N):
            self.m += 2.0*spins[i]-1.0
    
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
    
    def build_inputName(self,Network):
        
        name = 'data/samples/L'
        name += str(self.L)
        name += '/'
        name += NET.build_fileName(Network,'visible_samples')
        return name

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
    def build_outputName(self,Network):
        
        name = 'data/measurements/L'
        name += str(self.L)
        name += '/'
        name += NET.build_fileName(Network,'Ising_measures')
        return name

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
    def print_lattice(self):

        """ Print NN connections """

        for i in range(self.N):
            print ('\nSpin %d:' % i)
            for j in range(2*self.D):
                print self.Neighbors[i][j]
    
#-------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--L',type=int)
    parser.add_argument('--D',default=2,type=int)
    parser.add_argument('--T',type=int)
    parser.add_argument('--hid',type=int)
    parser.add_argument('--model',default='Ising2d')
    parser.add_argument('--ep',default = 2000     ,type=int)
    parser.add_argument('--bs',default = 50       ,type=int)
    parser.add_argument('--lr',default = 0.01     ,type=float)
    parser.add_argument('--CD',default = 20       ,type=int)
    parser.add_argument('--L2',default = 0.0      ,type=float)

    args = parser.parse_args()
    
    n_v = args.L*args.L

    Network = NET.Network(n_v,
                         args.model,
                         args.hid,
                         args.ep,
                         args.bs,
                         args.CD,
                         args.lr,
                         args.L2,
                         args.T)

    ising = IsingMeasure(Network,2)
    ising.build_lattice()
 
    inputName = 'data/samples/Lprova'
    inputName += str(args.L)
    inputName += '/'
    inputName += NET.build_fileName(Network,'visible_samples')
    
    outputName = 'data/measurements/L'
    outputName += str(args.L)
    outputName += '/'
    outputName += NET.build_fileName(Network,'measurements')

    inputFile = open(inputName,'r')
    samples = loadtxt(inputFile)
    
    outputFile = open(outputName,'w')
    
    for k in range(len(samples)):
        ising.get_energy(samples[k])
        ising.get_magnetization(samples[k])
        ising.record(outputFile)



