import gzip
import cPickle
import numpy as np
import theano
import argparse

#-------------------------------------------------

class Network(object):
    
    """ Class for the network parameters """
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, n_v,
                       model,
                       n_h = None,
                       epochs = None,
                       batch_size = None,
                       CD = None,
                       learning_rate = None,
                       L2 = None,
                       T_index = None,
                       W = None, b = None, c = None,
                       logZ = None):
        
        """ Constructor """
        
        # Write infos dictionary
        
        info = {}

        info['Hidden Units']  = n_h
        info['Visible Units'] = n_v
        info['Model']         = model
        info['Temperature']   = T_index
        info['Learning Rate'] = learning_rate
        info['Batch Size']    = batch_size
        info['Epochs']        = epochs
        info['CD']            = CD
        info['L2']            = L2
        info['logZ']          = logZ
        
        # Write parameters dictionary

        param = {}
        
        param[0] = W
        param[1]  = b
        param[2]  = c

        # Write network dictionary
        
        self.infos = {}

        self.infos['Parameters']   = param
        self.infos['Informations'] = info
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def load(self,trained_net):

        self.infos['Parameters'][0] = trained_net.infos['Parameters'][0]
        self.infos['Parameters'][1] = trained_net.infos['Parameters'][1]
        self.infos['Parameters'][2] = trained_net.infos['Parameters'][2]
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_network(network,fileName):

    """ Save network info on pickled file """

    f = open(fileName,'w')
    cPickle.dump(network,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def update_parameters(fileName,network,new_parameters):

    """ Update values of parameters and write on file"""

    network['Parameters'][0] = new_parameters[0]
    network['Parameters'][1]  = new_parameters[1]
    network['Parameters'][2]  = new_parameters[2]

    f = open(fileName,'w')
    cPickle.dump(network,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def update_logZ(fileName,network,logZ):

    """ Update values of parameters and write on file"""

    network.infos['Informations']['logZ'] = logZ

    f = open(fileName,'w')
    cPickle.dump(network,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_fileName(network,extension):

    """ Save network info on pickled file """

    name = 'RBM_CD'
    name += str(network['Informations']['CD'])
    name += '_hid'
    name += str(network['Informations']['Hidden Units'])
    name += '_bS'
    name += str(network['Informations']['Batch Size'])
    name += '_ep'
    name += str(network['Informations']['Epochs'])
    name += '_lr'
    name += str(network['Informations']['Learning Rate'])
    name += '_L'
    name += str(network['Informations']['L2'])
    name += '_'
    name += network['Informations']['Model']
    name += '_L'
    name += str(int(np.sqrt(network['Informations']['Visible Units'])))
    name += '_T'
    if (network['Informations']['Temperature']<10):
        name += '0'
        name += str(network['Informations']['Temperature'])
    else:
        name += str(network['Informations']['Temperature'])
    name += '_'
    name += extension
    if (extension == 'model'):
        name += '.pkl'
    else:
        name += '.txt'

    return name
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_networkPath(network):

    name = 'data/networks/L'
    name += str(int(np.sqrt(network.infos['Informations']['Visible Units'])))
    name += '/'
    name += build_fileName(network.infos,'model')
    
    return name

#------------------------------------------------

def print_network(net):
    
    """ Print the network infos """
    
    print '\n\n'
    print('*****************************')
    print('           NETWORK')
    print('\n\n')
    
    print ('RBM Informations\n')
    print ('\tModel: %s\n' % net['Informations']['Model'])
    print ('\tTemperature Index: %d\n' % net['Informations']['Temperature'])
    print ('\tVisible units: %d\n' % net['Informations']['Visible Units'])
    print ('\tHidden units: %d\n' % net['Informations']['Hidden Units'])
    print ('\n\nLearning Parameters:\n') 
    print ('\t- Epochs: %d\n' % net['Informations']['Epochs'])
    print ('\t- Batch size: %d\n' % net['Informations']['Batch Size'])
    print ('\t- Learning rate: %f\n' % net['Informations']['Learning Rate'])
    print ('\t- CD order: %d\n' % net['Informations']['CD'])
    print ('\t- Regularization: %f\n' % net['Informations']['L2'])
    print ('\n\nRBM Parameters')
    print ('\tWeights:')
    print net['Parameters'][0].eval()
    print '\n'
    print ('\tVisible Fields:')
    print net['Parameters'][1].eval()
    print '\n'
    print ('\tHidden Fields:')
    print net['Parameters'][2].eval()
    print '\n'
    print ('\tLog Partition Function:')
    print net['Informations']['logZ']
    print '\n'
 
#------------------------------------------------

def Format(oldNet):

    c = 0

#------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str)
    parser.add_argument('--net',type=str)

    args = parser.parse_args()

    if (args.command == 'print'):
        net= cPickle.load(open(args.net))
        print_network(net.infos)

    if (args.command == 'format'):
        Format(args.network)

