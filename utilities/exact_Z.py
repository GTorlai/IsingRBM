import math as m
import numpy as np
import theano
import network as NET
import argparse
import cPickle

#-------------------------------------------------

class Z(object):
    
    """ Class that computes the partition function  of
        a Restricted Boltzmann Machine exactly """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def __init__(self,n_visible,n_hidden):
        
        """ Constructor """

        self.n_v = n_visible
        self.n_h = n_hidden
        self.N = self.n_h + self.n_v
        self.N_states = pow(2,self.N)

        self.config = np.zeros(self.N)

        self.p = []
        self.Z = 0.0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_parameters(self,net):
        
        """ Load RBM parameters  """
 
        self.W = net['Weights'].eval()
        self.b = net['V Bias'].eval()
        self.c = net['H Bias'].eval()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_energy(self):

        """ Compute the energy of the RBM """

        visible = self.config[0:self.n_v]
        hidden = self.config[self.n_v:]

        en = np.dot(self.c,hidden)
        en += np.dot(self.b,visible)
        en += np.dot(visible,np.dot(self.W,hidden))

        return -1.0*en
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_Z(self):

        """ Compute the Partition Function """

        for i in range(self.N_states):
            st = bin(i)[2:].zfill(self.N)
            state = st.split()
            for j in range(self.N):
                self.config[j] = state[0][j]
            e = self.get_energy()
            self.p.append(m.exp(-e))

        self.Z = np.sum(self.p)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getProbabilityDistr(self):

        """ Evaluate the full probability distribution """

        self.p /= self.Z
        self.p = np.array(self.p)

        return self.p
    
#-------------------------------------------------
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',help='Name of the model',default='Ising2d')
    parser.add_argument('--L', help='Linear size of the model',type=int)
    parser.add_argument('--D', help='Model dimension', default=2,type=int)
    parser.add_argument('--T', help='Temperature Index', type=int)
    parser.add_argument('--hid',help='Number of hidden units', type=int)
    parser.add_argument('--ep', help='Epochs', default = 2000,type=int)
    parser.add_argument('--bs', help='Batch Size', default = 50 ,type=int)
    parser.add_argument('--lr', help='Learning Rate', default = 0.01,type=float)
    parser.add_argument('--CD', help='Contrastive Divergence', default = 20,type=int)
    parser.add_argument('--L2', help='Regularization', default = 0.0 ,type=float)
    parser.add_argument('--l', help='Binning Level', type=str)
    parser.add_argument('--targ', help='Data',  type=str)
 
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
     
    Trained_Network = cPickle.load(open(pathToNetwork,'rb'))
                
    Network.load(Trained_Network)

    exact = Z(n_v,args.hid)
    exact.get_parameters(Network.infos)
    exact.get_Z()
    logZ = np.log(exact.Z)
    
    print ('\nLog Partition Function: %f' % logZ) 
    
    NET.update_logZ(pathToNetwork,Network,logZ,0,0,0)

#-------------------------------------------------
