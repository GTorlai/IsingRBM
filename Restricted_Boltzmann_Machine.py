import cPickle
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utilities.network as NET
import argparse
import math as m
import os

#-------------------------------------------------

class RBM(object):

    """ Class that builds a restricted Boltzmann Machine """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,input,Network):
       
        """ Constructor """

        print ('..initializing parameters')
        
        self.Network = Network

        # Learning parameters
        self.epochs        = Network.infos['Epochs']
        self.batch_size    = Network.infos['Batch Size']
        self.learning_rate = Network.infos['Learning Rate']
        self.CD_order      = Network.infos['CD']
        self.L2_par        = Network.infos['L2']
        
        # Number of visible and hidden unites
        self.n_v = Network.infos['Visible Units']
        self.n_h = Network.infos['Hidden Units']
        
        # Network Parameters
        W = Network.infos['Weights']
        b = Network.infos['V Bias']
        c = Network.infos['H Bias']

        # Numpy & Theano random number generators
        np_rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(2**17)
        
        # Initialize rbm parameters
        # Weights
        if W is None:
            bound = 4.0 * np.sqrt(6.0/(self.n_h + self.n_v))

            w = np.asarray(np_rng.uniform(
                       low = -bound,
                       high = bound,
                       size = (self.n_v, self.n_h)),
                       dtype = theano.config.floatX
            )

            W = theano.shared(value = w,
                               name = 'Weights',
                               borrow = True)
        
        # Visible biases
        if b is None:
            b = theano.shared(value = np.zeros((self.n_v), 
                               dtype = theano.config.floatX),
                               name = 'Biases',
                               borrow = True)
        
        # Hidden biases    
        if c is None:
            c = theano.shared(value = np.zeros((self.n_h), 
                               dtype = theano.config.floatX),
                               name = 'Biases',
                               borrow = True)

        self.W = W
        self.b = b
        self.c = c
        
        # Parameters
        self.params = [self.W,self.b,self.c]
        
        # Regularization
        self.L2_reg = (self.W**2).sum()

        # Input of the rbm
        self.input = input

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sample_hidden(self,v_state):
        
        """ Sample hidden layers conditional to the state of the visible layer """

        h_pre_activation = T.dot(v_state,self.W) + self.c
        
        h_activation = T.nnet.sigmoid(h_pre_activation)
        
        h_state = self.theano_rng.binomial(size = h_activation.shape,
                                          n = 1, p = h_activation,
                                          dtype=theano.config.floatX)
        
        return [h_pre_activation,h_activation,h_state]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sample_visible(self,h_state):

        """ Sample visible layer conditional to the state of the hidden layer """

        v_pre_activation = T.dot(h_state,self.W.T) + self.b
        
        v_activation = T.nnet.sigmoid(v_pre_activation)
        
        v_state = self.theano_rng.binomial(size = v_activation.shape,
                                           n = 1, p = v_activation,
                                           dtype=theano.config.floatX)
        
        return [v_pre_activation,v_activation,v_state]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def reconstruction(self,v_state):

        [h_pre_act,h_act,h_state] = self.sample_hidden(v_state)
        [v_pre_act,v_act,v_state] = self.sample_visible(h_state)

        return [v_act,v_state,h_act,h_state]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def free_energy(self,v_state):

        """ Compute the free energy of the visible state """

        vis = T.dot(v_state,self.b)
        hid = T.dot(v_state,self.W) + self.c

        F = - vis - T.sum(T.log(1.0 + T.exp(hid)),axis = 1)
        
        return F
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def CD_k(self,k):

        """ Perform one step of k-th order contrastive divergence """
        
        [h_pre_act0,h_act0,h_state0] = self.sample_hidden(self.input)
        
        h_state = h_state0

        for i in range(k):
            [v_pre_act,v_act,v_state] = self.sample_visible(h_state)
            [h_pre_act,h_act,h_state] = self.sample_hidden(v_state)
        
        # Cost function
        cost = T.mean(self.free_energy(self.input))-\
               T.mean(self.free_energy(v_state)) + self.L2_par * self.L2_reg

        # Symbolic gradient
        gradient = T.grad(cost=cost,wrt=self.params,
                          consider_constant=[v_state])

        # Compute gradient descent updates
        updates = []
        for par,grad in zip(self.params,gradient):
            updates.append([par, par - T.cast(self.learning_rate,
                            dtype=theano.config.floatX) * grad])

        return updates
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Train(self,train_set,X):

        """ Train the model """

        n_train_batches=train_set.get_value(borrow=True).shape[0]/self.batch_size

        print ('...building the model graph\n') 
        
        # Symbolic scalar for the batch index
        batch_index = T.lscalar('batch index')

        updates = self.CD_k(self.CD_order)
        
        # Train function
        train_model = theano.function(
                inputs = [batch_index],
                updates = updates,
                givens = {
                    X: train_set[batch_index * self.batch_size: 
                              (batch_index + 1) * self.batch_size]
                },
                name = 'Train'
        )
        
        epoch = 0
       
        fileName = NET.build_fileName(self.Network,'model')
        
        NET.save_network(self.Network,fileName)

        print ('\n*****************************\n')
        print ('..training')
        
        # Training
        while (epoch < self.epochs):

            epoch = epoch + 1
            
            for minibatch_index in xrange(n_train_batches):
                
                train_model(minibatch_index)
            
            NET.update_parameters(fileName,
                                 self.Network,
                                 self.params)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    def sample(self,layer):
        
        """ Sample visible units from the rbm """
        
        eq_steps = 100000
        n_measure = 100000
        record_frequency = 10

        V_fileName = NET.build_fileName(self.Network,'visible_samples')
        H_fileName = NET.build_fileName(self.Network,'hidden_samples')

        outputVisible = open(V_fileName,'w')
                
        if (layer == 'full'):
            outputHidden  = open(H_fileName,'w')

        spin0 = np.zeros((self.n_v),dtype = theano.config.floatX)
        hidden0 = np.zeros((self.n_h),dtype = theano.config.floatX)
        
        visible_chain = theano.shared(value = spin0)
        hidden_chain = theano.shared(value = hidden0)

        ([v_act,v_state,h_act,h_state],updates) = theano.scan(
                self.reconstruction,
                outputs_info =[None,visible_chain,None,None],
                n_steps = record_frequency
         )

        updates.update({visible_chain: v_state[-1]})

        sample = theano.function(
                inputs = [],
                outputs = [v_act[-1],v_state[-1],h_act[-1],h_state[-1]],
                updates = updates,
        )

        for k in range(eq_steps):
            vis_activation, vis_state, hid_activation, hid_state = sample()
        
        for k in range(n_measure):
            
            vis_activation, vis_state, hid_activation, hid_state = sample()
            
            for j in range(self.n_v):
                outputVisible.write('%d' % vis_state[j])
                outputVisible.write(" ")
            
            if (layer == 'full'):
                for i in range(self.n_h):
                    outputHidden.write('%d' % hid_state[i])
                    outputHidden.write(" ")
            
            outputVisible.write('\n')
            
            if (layer == 'full'):
                outputHidden.write('\n')
        
        outputVisible.close()
        
        if (layer == 'full'):
            outputHidden.close()
 
