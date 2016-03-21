import math as m
import numpy as np
import theano


class AnnealedImportanceSampling(object):

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


    def get_parameters(self,params):
        
        self.W = params[0].eval()
        self.b = params[1].eval()
        self.c = params[2].eval()

    def BaseRate_sampler(self):

        for j in range(self.n_v):
            p = sigmoid(self.baseRate_b[j])
            r = np.random.rand()
            
            if (p > r):
                self.visible[j] = 1
            else:
                self.visible[j] = 0


    def Gibbs_sampler(self,k):
        
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

    def get_prob(self,k):

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

    def sweep(self):

        W = 1.0

        for k in range(1,self.K+1):

            for i in range(10):
                self.Gibbs_sampler(k-1)

            W *= self.get_prob(k) / self.get_prob(k-1)

        return W

    def getZ(self):
        
        for i in range(self.M):
            
            print ('Sweep: %d' % i)
            self.BaseRate_sampler()
            self.w.append(self.sweep())

        r = np.mean(self.w)
        
        self.Z = self.Z_BR * r
       
        #delta_r = 0.0
        #
        #for i in range(self.M):
        #    delta_r += (1.0*self.w[i]-r)**2
        #
        #d_r = m.sqrt(delta_r/(self.M*(self.M-1)))
        #
        #self.dZ = self.Z_BR * d_r
        #print self.Z
        #print self.dZ

    def outputLogZ(self,network,T):

        logZ = m.log(self.Z)
        
        linear_size = int(np.sqrt(self.n_v))
        fileName = 'partition_functions/hid'
        fileName += str(self.n_h)
        fileName += '/sw'
        fileName += str(self.M)
        fileName += '/Z_k'
        fileName += str(self.K)
        fileName += '_sw'
        fileName += str(self.M)
        #fileName = 'partition_functions/Z_CD'
        #fileName += str(network['Informations']['CD_order'])
        #fileName += '_hid'
        #fileName += str(self.n_h)
        #fileName += '_bS'
        #fileName += str(network['Informations']['Batch Size'])
        #fileName += '_ep'
        #fileName += str(network['Informations']['Epochs'])
        #fileName += '_lr'
        #fileName += str(network['Informations']['Learning Rate'])
        #fileName += '_L'
        #fileName += str(network['Informations']['L2'])
        #fileName += '_Ising2d_L'
        #fileName += str(linear_size)
        fileName += '_T'
        fileName += str(T)
        fileName += '.txt'
        
        output = open(fileName,'w')
        output.write('%f' % logZ)
        output.close()

        #dlogZ = 1.0/(self.Z) * self.dZ
        #return logZ
        #return [logZ,dlogZ]


def sigmoid(x):
    return 1./(1.+m.exp(-x))
