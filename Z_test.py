import math as m
import numpy as np
import theano
import AIS as AIS
import Restricted_Boltzmann_Machine as RBM
import theano.tensor as T
import Tools as Tools

class Z(object):

    def __init__(self,n_visible,n_hidden):

        self.n_v = n_visible
        self.n_h = n_hidden
        self.N = self.n_h + self.n_v
        self.N_states = pow(2,self.N)

        self.config = np.zeros(self.N)

        self.p = []
        #self.energy = []
        self.Z = 0.0

    def get_parameters(self,params):
        
        self.p = []
        self.W = params[0].eval()
        self.b = params[1].eval()
        self.c = params[2].eval()

    def getEnergy(self):

        visible = self.config[0:self.n_v]
        hidden = self.config[self.n_v:]

        en = np.dot(self.c,hidden)
        en += np.dot(self.b,visible)
        en += np.dot(visible,np.dot(self.W,hidden))

        return -1.0*en

    def getZ(self):

        for i in range(self.N_states):
            st = bin(i)[2:].zfill(self.N)
            state = st.split()
            for j in range(self.N):
                self.config[j] = state[0][j]
            e = self.getEnergy()
            self.p.append(m.exp(-e))

        self.Z = np.sum(self.p)
        
        return self.Z

    def getProbabilityDistr(self):

        self.p /= self.Z
        self.p = np.array(self.p)

        return self.p



def main():

    n_hidden = 4
    n_visible = 4
    k = 100
    AIS_run = 100

    X = T.matrix()
    SimPar = Tools.SimPar(1,
                          1,
                          1,
                          1,
                          1,
                          )
    
    dataset = 'datasets/spins/L8/Ising2d_L8_spins_T0.pkl.gz'
    
    print ('\nInitializing the Restricted Boltzmann Machine\n')
    rbm = RBM.RBM(X,dataset,n_visible,n_hidden,SimPar)
    
    print ('\nInitializing the Partition Function module\n')
    exact = Z(n_visible,n_hidden)
    exact.get_parameters(rbm.params)

    print('\nEvaluating the Partition Function\n')
    exact.getZ()

    print ('\nThe exact Log-Partition Function is %f: \n' % np.log(exact.Z))
    
    annealed = AIS.AnnealedImportanceSampling(n_visible,n_hidden,k)
    annealed.get_parameters(rbm.params)
    AIS_Z = annealed.getZ(AIS_run)
    print ('Annealed Log-Partition Funcion: %f' % np.log(AIS_Z))


if __name__ == "__main__":
    main()







