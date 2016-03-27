import cPickle
import utilities.network as NET
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import argparse
import math as m
import os

#-------------------------------------------------

class ClassRBM(object):

    """ Class that performs binary classification using RBM"""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,Network):
       
        """ Constructor """

        # Number of visible and hidden unites
        self.n_v = Network['Visible Units']
        self.n_h = Network['Hidden Units']
        
        # Weights
        self.W = Network['Weights'].eval()          
        self.b = Network['V Bias'].eval()
        self.c = Network['H Bias'].eval()
        
        # Partition Function
        self.logZ = Network['logZ']

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def free_energy(self,v_state):

        """ Compute the free energy of the visible state """

        vis = np.dot(v_state,self.b)
        hid = np.dot(v_state,self.W) + self.c
        F = - vis - np.sum(np.log(1.0 + np.exp(hid)))
        
        return F
 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Neg_Log_Prob(self,v_state):

        """ Compute the Negative Log-Probability """

        Log_p = - self.free_energy(v_state) - self.logZ
        
        return -Log_p
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_NLP(self,pathToTestSet):

        f = open(pathToTestSet,'r')
        testSet = np.loadtxt(f)
        f.close()
        
        n_test = len(testSet)
        
        NLP_vector = np.zeros((n_test))

        for i in range(n_test):
            
            NLP_vector[i] = self.Neg_Log_Prob(testSet[i])
        
        return NLP_vector
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Test(self,NLP,outputFile):
        
        temperatureFile = open('data/datasets/Ising2d_Temperatures.txt','r')
        T_array = np.loadtxt(temperatureFile)

        n_machines = len(NLP)
        n_test     = len(NLP[0])
        n= n_test / n_machines

        prob = np.zeros((n_test,n_machines))

        for j in range(n_test):
            
            Norm = 0.0
            
            for i in range(n_machines):
                Norm += np.exp(-NLP[i,j])

            for i in range(n_machines):
                prob[j,i] = np.exp(-NLP[i,j]) / Norm

        accuracy = np.zeros((n_machines))

        for i in range(n_machines):
        
            for j in range(n):

                most_likely = np.argmax(prob[i*n+j])

                if (i<10):
                    if (most_likely < 10):
                        accuracy[i] += 1
                if (i>10):
                    if (most_likely > 10):
                        accuracy[i] += 1

            accuracy[i] /= n

        outputFile.write('#   T     Acc\n') 
        
        average_accuracy = 0.0

        for i in range(n_machines):
            if (i != 10):
                outputFile.write('%.3f  ' % T_array[i])
                outputFile.write('%.4f  ' % accuracy[i])
                outputFile.write('\n')
                average_accuracy += accuracy[i]
        
        average_accuracy /= (n_machines - 1)

        print ('\n\nAverage accuracy: %.1f%%' % 100*average_accuracy)

#-------------------------------------------------
        
def build_networkName(network,T_index):
    
    name  = 'data/networks/L'
    name += str(int(np.sqrt(network['Visible Units'])))
    name += '/RBM_CD'
    name += str(network['CD'])
    name += '_hid'
    name += str(network['Hidden Units'])
    name += '_bS'
    name += str(network['Batch Size'])
    name += '_ep'
    name += str(network['Epochs'])
    name += '_lr'
    name += str(network['Learning Rate'])
    name += '_L'
    name += str(network['L2'])
    name += '_'
    name += network['Model']
    name += '_L'
    name += str(int(np.sqrt(network['Visible Units'])))
    name += '_T'
    if (T_index<10):
        name += '0'
        name += str(T_index)
    else:
        name += str(T_index)
    name += '_model.pkl'

    return name

#-------------------------------------------------
        
def build_outputName(network):
    
    name  = 'data/observables/L'
    name += str(int(np.sqrt(network['Visible Units'])))
    name += '/RBM_CD'
    name += str(network['CD'])
    name += '_hid'
    name += str(network['Hidden Units'])
    name += '_bS'
    name += str(network['Batch Size'])
    name += '_ep'
    name += str(network['Epochs'])
    name += '_lr'
    name += str(network['Learning Rate'])
    name += '_L'
    name += str(network['L2'])
    name += '_'
    name += network['Model']
    name += '_L'
    name += str(int(np.sqrt(network['Visible Units'])))
    name += '_classification'
    name += '.dat'

    return name



