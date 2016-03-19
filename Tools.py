import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import glob
import argparse
from pylab import loadtxt
#-------------------------------------------------

class SimPar(object):
    
    """ A class containing all the hyper parameters"""

    def __init__(self,epochs_,
                      batch_size_,
                      CD_order_,
                      lr_,
                      L2_par_,
                      ):

        self.epochs = epochs_
        self.batch_size = batch_size_
        self.CD_order = CD_order_
        self.lr = lr_
        self.L2_par = L2_par_

#-------------------------------------------------

def Save_network(fileName,net):
   
    """ Save the information and parameters of the network """

    information = {}
    fullNet = {}
    
    information['Hidden Units']  = net.n_h
    information['Learning Rate'] = net.learning_rate
    information['Batch Size']    = net.batch_size
    information['Epochs']        = net.epochs
    information['CD_order']      = net.CD_order
    information['L2']            = net.L2_par
    #Collecting everything
    fullNet['Parameters']            = net.params
    fullNet['Informations']          = information

    #dictionary['Costs'] = costs

    f = open(fileName,'w')
    cPickle.dump(fullNet,f)
    f.close()


#------------------------------------------------

def Print_network(netName):
    
    """ Print information of the network """

    f = open(netName)
    dictionary = cPickle.load(f)
    f.close()

    print('*****************************')
    print('           NETWORK')
    print('\nInfos:')
    
    for par in dictionary['Informations']:
        print (par,dictionary['Informations'][par])
    
    #for par in dictionary['Parameters']:
    W = dictionary['Parameters'][0].get_value(borrow=True)
    for j in range(64):

        for i in range(int(dictionary['Informations']['Hidden Units'])):

            print ('%f' % W[j][i])
            print ('\n')

#------------------------------------------------

def Parameter_histogram(network,parameter):
    
    """ Print information of the network """

    f = open(network)
    dictionary = cPickle.load(f)
    f.close()

    print('*****************************')
    print('           NETWORK')
    print('\nInfos:')
    

    if (parameter == 'W'):
        temp = dictionary['Parameters'][0].get_value(borrow=True)
        par = []
        for j in range(64):
            for i in range(int(dictionary['Informations']['Hidden Units'])):
                par.append(temp[j][i])

        par = np.array(par)
    
    elif (parameter == 'b'):
        par = dictionary['Parameters'][1].get_value(borrow=True)
    
    elif (parameter == 'c'):
        par = dictionary['Parameters'][2].get_value(borrow=True)

    plt.hist(par, bins=100, normed=False)
    plt.title("%s Histogram" % parameter)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

#-------------------------------------------------

def Pickle_datasets():
    
    """ Create a compact pickled version of the dataset.
        Arguments: name    -> Name for the output file
        Input    : file with the following names:
                   Training_Images.dat, Training_Labels.dat, 
                   Validation_Images.dat, Validation_Labels.dat,
                   header.txt file with description.
        Output   : Pickled file ready to be loaded.
        
    """
    
    path = 'datasets/raw/L8_ergodic/*.txt'

    files = glob.glob(path)
    Tcounter = 0

    for fileName in files:
        with open(fileName) as f:
            
            print 'Opening file:'
            print fileName

            data = []

            for images in f:
                image = images.split()
                pixel = [i for i in image]
                data.append(pixel)
                
            data = np.array(data,dtype='float32')

        outputName = 'datasets/spins/L8/Ising2d_ergodic_L8_spins_T'
        outputName += str(Tcounter)
        outputName += '.pkl.gz'

        with gzip.open(outputName,'wb') as output:
            cPickle.dump(data,output,protocol=cPickle.HIGHEST_PROTOCOL)
        output.close()
        Tcounter += 1

#------------------------------------------------

def Format_Dataset_CPP(fileName):
    
    f = gzip.open(fileName)
    sizePath = 9

    header, dic = cPickle.load(f)
    outputName = fileName[sizePath:-7]
    outputName += str('.txt')
    output = open(outputName,'w')

    tempSteps = len(dic['Temperatures'])
    configNumber = len(dic[dic['Temperatures'][0]])
    size = len(dic[dic['Temperatures'][0]][0])
 
    print ('\nNumber of temperatures in the datasets: %i' % tempSteps)
    print ('\nNumber of configuration per temperature: %i' % configNumber)
    print ('\nNumber of spins: %i' % size) 
    #print dic[dic['Temperatures'][0]][0][1] 
    for i in range(tempSteps):
        for j in range(configNumber):
            for k in range(size):
                output.write('%d' % dic[dic['Temperatures'][i]][j][k])
                output.write(" ") 
            output.write('\n')

    output.close()


#------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str)
    parser.add_argument('--file',type=str)
    parser.add_argument('--model',type=str)
    parser.add_argument('--par',type=str)

    args = parser.parse_args()

    if args.command == 'pickle':
        Pickle_datasets()

    elif args.command == 'format':
        Format_Dataset_CPP(args.file)

    elif args.command == 'print':
        Print_network(args.model)
    
    elif args.command == 'histogram':
        Parameter_histogram(args.model,args.par)

