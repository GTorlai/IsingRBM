import math as m
import numpy as np
import theano


class AnnealedImportanceSampling(object):

    def __init__(self,n_visible,n_hidden,K):

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

        self.K = K

        for i in range(self.K):
            self.beta.append(1.*i/self.K)
        self.beta.append(1.)


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

            for i in range(1):
                self.Gibbs_sampler(k-1)

            W *= self.get_prob(k) / self.get_prob(k-1)

        return W

    def getZ(self,M):

        for i in range(M):
            #print ('AIS run # %d' % i)
            self.BaseRate_sampler()
            self.w.append(self.sweep())

        r = np.mean(self.w)
        
        self.Z = self.Z_BR * r
        return self.Z


def sigmoid(x):
    return 1./(1.+m.exp(-x))
