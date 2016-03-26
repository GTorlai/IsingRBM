import math as m
import numpy as np
import theano
import cPickle
import network as NET
import argparse

#-------------------------------------------------

class Z(object):
    
    """ Class that computes the partition function  of
        a Restricted Boltzmann Machine exactly """
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,n_visible,n_hidden,K,M):

        self.n_v = n_visible
        self.n_h = n_hidden
        
        if (self.n_h > 150):
            self.n_hBR = 150
        else:
            self.n_hBR = self.n_h

        self.visible = np.zeros(self.n_v)
        self.hidden = np.zeros(self.n_h)

        self.baseRate_b = np.asarray(np.random.RandomState(4353).uniform(
                low = -1.0,
                high = 1.0,
                size = (self.n_v))
                )

        self.Z_BR = pow(2,self.n_hBR)
        
        for i in range(self.n_v):
            self.Z_BR *= (1. + m.exp(self.baseRate_b[i]))

        self.beta = []

        self.w = []
        self.Z = 0.0
        self.dZ = 0.0

        self.K = K
        self.M = M

        for i in range(self.K/10):
            self.beta.append(0.5*i/(self.K/10))
        
        for i in range(4*self.K/10):
            self.beta.append(0.5+0.4*i/(4*self.K/10))
        
        for i in range(self.K/2):
            self.beta.append(0.9+0.1*i/(self.K/2))
        
        self.beta.append(1.)
 
        #for i in range(self.K):
        #    self.beta.append(1.*i/self.K)
        #self.beta.append(1.)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_parameters(self,net):
        
        """ Load RBM parameters  """

        self.W = net['Weights'].eval()
        self.b = net['V Bias'].eval()
        self.c = net['H Bias'].eval()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def BaseRate_sampler(self):

        """ Sampler for base-rate RBM  """
        
        for j in range(self.n_v):
            p = sigmoid(self.baseRate_b[j])
            r = np.random.rand()
            
            if (p > r):
                self.visible[j] = 1
            else:
                self.visible[j] = 0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Gibbs_sampler(self,k):

        """ Sampler for RBM"""
        
        for i in range(self.n_h):
            p = sigmoid(self.beta[k]*(np.dot(self.visible,self.W)[i]+self.c[i]))
            r = np.random.rand()
            
            if (p > r):
                self.hidden[i] = 1
            else:
                self.hidden[i] = 0

        for j in range(self.n_v):
            p = sigmoid(self.beta[k]*(np.dot(self.W,self.hidden)[j]+self.b[j])+\
                        (1-self.beta[k])*self.baseRate_b[j])
            r = np.random.rand()
            
            if (p > r):
                self.visible[j] = 1
            else:
                self.visible[j] = 0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_prob(self,k):
        
        """ Compute product of probabilities"""

        p_A = m.exp((1-self.beta[k])*np.dot(self.baseRate_b,self.visible))
        p_A *= pow(2,self.n_hBR)

        p_B = m.exp(self.beta[k]*np.dot(self.b,self.visible))

        for i in range(self.n_h):
            acc = 0.0
            
            for j in range(self.n_v):
                acc += self.W[j,i]*self.visible[j]
            acc += self.c[i]
            p_B *= (1. + m.exp(self.beta[k]*acc))

        return p_A*p_B
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sweep(self):
        
        """ One AIS sweep"""

        W = 1.0

        for k in range(1,self.K+1):

            for i in range(10):
                self.Gibbs_sampler(k-1)

            W *= self.get_prob(k) / self.get_prob(k-1)

        return W
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_Z(self,outputFile):
        
        """ Compute Partition Function"""

        for i in range(self.M):
            
            self.BaseRate_sampler()
            #self.w.append(self.sweep())
            meas = self.sweep() 
            #self.w.append(meas)
            measure = m.log(self.Z_BR*meas)
            outputFile.write('%.5f\n', % measure)

        #r = np.mean(self.w)
        
        #self.Z = self.Z_BR * r
    
#-------------------------------------------------

def sigmoid(x):
    return 1./(1.+m.exp(-x))

#-------------------------------------------------
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',default='Ising2d',type=str)
    parser.add_argument('--L',type=int)
    parser.add_argument('--T',type=int)
    parser.add_argument('--hid',type=int)
    parser.add_argument('--ep',default = 2000     ,type=int)
    parser.add_argument('--bS',default = 50       ,type=int)
    parser.add_argument('--lr',default = 0.01     ,type=float)
    parser.add_argument('--CD',default = 20       ,type=int)
    parser.add_argument('--L2',default = 0.0      ,type=float)
    parser.add_argument('--K',default = 1000     ,type=float)
    parser.add_argument('--M',default = 100       ,type=float)
 
    args = parser.parse_args()
    
    n_v = args.L**2

    Network = NET.Network(n_v,
                  args.model,
                  args.hid,
                  args.ep,
                  args.bS,
                  args.CD,
                  args.lr,
                  args.L2,
                  args.T)
 
    pathToNetwork = '../'
    pathToNetwork += NET.build_networkPath(Network)
        
    Trained_Network = cPickle.load(open(pathToNetwork))
                
    Network.load(Trained_Network)
    
    outputName = '../data/measurements/L'
    outputName += str(args.L)
    outputName += '/'
    outputName += NET.build_fileName(Network,'logZ')

    outputFile = open(outputName, 'w')

    annealed = Z(n_v,args.hid,args.K,args.M)
    annealed.get_parameters(Network.infos)
    annealed.get_Z(outputFile)

    #logZ = np.log(annealed.Z)

    #print ('\nLog Partition Function: %f' % logZ) 
    
    #NET.update_logZ(pathToNetwork,Network,logZ,args.K,args.M)

#-------------------------------------------------
