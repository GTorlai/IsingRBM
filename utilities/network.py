import gzip
import cPickle
import numpy as np
import theano
import argparse
import glob
import matplotlib.pyplot as plt
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
                       T_index = None):
        
        """ Constructor """
        
        # Write infos dictionary
        
        self.infos = {}

        self.infos['Hidden Units']  = n_h
        self.infos['Visible Units'] = n_v
        self.infos['Model']         = model
        self.infos['Temperature']   = T_index
        self.infos['Learning Rate'] = learning_rate
        self.infos['Batch Size']    = batch_size
        self.infos['Epochs']        = epochs
        self.infos['CD']            = CD
        self.infos['L2']            = L2
        
        # Initialize empty physical parameters

        self.infos['Weights']  = None
        self.infos['V Bias']   = None
        self.infos['H Bias']   = None
        self.infos['logZ']     = None
        self.infos['d_logZ']   = None
        self.infos['AIS beta'] = None
        self.infos['AIS runs'] = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def load(self,trained_net):

        """ Load physics parameters from trained network"""

        self.infos['Weights']  = trained_net['Weights']
        self.infos['V Bias']   = trained_net['V Bias']
        self.infos['H Bias']   = trained_net['H Bias']
        self.infos['logZ']     = trained_net['logZ']
        self.infos['d_logZ']   = trained_net['d_logZ']
        self.infos['AIS beta'] = trained_net['AIS beta']
        self.infos['AIS runs'] = trained_net['AIS runs']
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_network(network,fileName):

    """ Save network info on pickled file """

    f = open(fileName,'w')
    cPickle.dump(network.infos,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def update_parameters(fileName,network,new_parameters):

    """ Update values of parameters and write on file"""

    network.infos['Weights'] = new_parameters[0]
    network.infos['V Bias']  = new_parameters[1]
    network.infos['H Bias']  = new_parameters[2]

    f = open(fileName,'wb')
    cPickle.dump(network.infos,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def update_logZ(fileName,network,logZ,dlogZ,beta,runs):

    """ Update values of parameters and write on file"""

    network['logZ']     = logZ
    network['d_logZ']   = dlogZ
    network['AIS beta'] = beta
    network['AIS runs'] = runs
    
    if runs == 0:
        network['d_logZ'] = 0
    
    f = open(fileName,'wb')
    cPickle.dump(network,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def add_fields(fileName):

    """ Update values of parameters and write on file"""

    f = open(fileName,'rb')
    infos = cPickle.load(f)
    
    # ADD FIELD HERE

    f.close()
    
    f = open(fileName,'wb')
    cPickle.dump(infos,f)
    f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_fileName(network,extension):

    """ Save network info on pickled file """

    name = 'RBM_CD'
    name += str(network.infos['CD'])
    name += '_hid'
    name += str(network.infos['Hidden Units'])
    name += '_bS'
    name += str(network.infos['Batch Size'])
    name += '_ep'
    name += str(network.infos['Epochs'])
    name += '_lr'
    name += str(network.infos['Learning Rate'])
    name += '_L'
    name += str(network.infos['L2'])
    name += '_'
    name += network.infos['Model']
    name += '_L'
    name += str(int(np.sqrt(network.infos['Visible Units'])))
    name += '_T'
    if (network.infos['Temperature']<10):
        name += '0'
        name += str(network.infos['Temperature'])
    else:
        name += str(network.infos['Temperature'])
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
    name += str(int(np.sqrt(network.infos['Visible Units'])))
    name += '/'
    name += build_fileName(network,'model')
    
    return name

#------------------------------------------------

def print_network(netInfos):
    
    """ Print the network infos """
    
    print '\n\n'
    print('*****************************')
    print('           NETWORK')
    print('\n\n')
    
    print ('RBM Informations\n')
    print ('\tModel: %s\n' % netInfos['Model'])
    print ('\tTemperature Index: %d\n' % netInfos['Temperature'])
    print ('\tVisible units: %d\n' % netInfos['Visible Units'])
    print ('\tHidden units: %d\n' % netInfos['Hidden Units'])
    print ('\n\nLearning Parameters:\n') 
    print ('\t- Epochs: %d\n' % netInfos['Epochs'])
    print ('\t- Batch size: %d\n' % netInfos['Batch Size'])
    print ('\t- Learning rate: %f\n' % netInfos['Learning Rate'])
    print ('\t- CD order: %d\n' % netInfos['CD'])
    print ('\t- Regularization: %f\n' % netInfos['L2'])
    
    print ('\n\nRBM Parameters\n\n')
    print ('\tWeights:\n')
    print netInfos['Weights'].eval()
    print '\n'
    print ('\tVisible Fields:\n')
    print netInfos['V Bias'].eval()
    print '\n'
    print ('\tHidden Fields:\n')
    print netInfos['H Bias'].eval()
    
    if netInfos['logZ'] is not None:
        print '\n\nPartition Function\n\n'
        print ('\tLog Partition Function: %f\n' % netInfos['logZ'])
        if netInfos['d_logZ'] is not None:
            print ('\t- Error: %f\n' % netInfos['d_logZ'])
        print ('\t- AIS beta: %d\n' % netInfos['AIS beta'])
        print ('\t- AIS runs: %d\n' % netInfos['AIS runs'])
 

#------------------------------------------------

def export_parameters(T,par):
    
    """ Print information of the network """

    n_v = 64 
    n_h = 64
    
    networkName = '../data/networks/L8/RBM_CD20_hid64_bS50_ep2000_lr0.01_L0.0_Ising2d_L8_T'
    if (T<10):
        networkName += '0'
    networkName += str(T)
    networkName += '_model.pkl'
    
    net = cPickle.load(open(networkName))

    W = net['Weights'].get_value(borrow=True)
    b = net['V Bias'].get_value(borrow=True)
    c = net['H Bias'].get_value(borrow=True)

    outputName = '../data/networks/L8/parameters_txt/RBM_CD20_hid64_bS50_ep2000_lr0.01_L0.0_Ising2d_L8_T'
    if (T<10):
        outputName += '0'
    outputName += str(T)
    outputName += '_par_'
    outputName += par
    outputName += '_.txt'
    output = open(outputName,'w')
    
    if (par is 'W'):
        np.savetxt(output,W,fmt='%8f')
    if (par is 'b'):
        np.savetxt(output,b,fmt='%8f')
    if (par is 'c'):
        np.savetxt(output,c,fmt='%8f')
    
#------------------------------------------------

if __name__ == "__main__":
   

    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str)
    parser.add_argument('--net',type=str)
    parser.add_argument('--par',type=str)
    parser.add_argument('--test',type=str)
    parser.add_argument('--T',type=int)
    args = parser.parse_args()

    if (args.command == 'print'):
        net= cPickle.load(open(args.net))
        #print_network(net)

    if (args.command == 'add'):
        path = '../data/networks/L10/*.pkl'
        files = glob.glob(path)
        counter = 0
        for fileName in files:
            add_fields(fileName)
     
    if (args.command == 'export_par'):

        export_parameters(args.T,args.par)
