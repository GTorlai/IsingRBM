import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import glob
import argparse

#-------------------------------------------------

class SimPar(object):
    
    """ A class containing all the hyper parameters"""

    def __init__(self,epochs_,
                      batch_size_,
                      CD_order_,
                      lr_,
                      L2_par_,
                      AISchains_,
                      AISsweeps_
                      ):

        self.epochs = epochs_
        self.batch_size = batch_size_
        self.CD_order = CD_order_
        self.lr = lr_
        self.L2_par = L2_par_
        self.AISchains = AISchains_
        self.AISsweeps = AISsweeps_

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
    information['Partition Function'] = net.Z
    information['AIS chains']       = net.AIS_chains
    information['AIS sweeps']       = net.AIS_sweeps

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

    fileName = 'networks/'
    fileName += str(netName)
    fileName += str('_model.pkl')

    f = open(fileName)
    dictionary = cPickle.load(f)
    f.close()

    print('*****************************')
    print('           NETWORK')
    print('\nInfos:')
    
    for par in dictionary['Informations']:
        print (par,dictionary['Informations'][par])
    
    for par in dictionary['Parameters']:
        print par
        print par.get_value(borrow=True).shape


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
    
    path = 'datasets/raw/L16/*.txt'

    #outputName = name + str('.pkl.gz')        
    
    files = glob.glob(path)
    #dic = {}
    #sizePath = 9
    #tempList = []
    Tcounter = 0

    for fileName in files:
        with open(fileName) as f:
            
            print 'Opening file:'
            print fileName

            #setTemp = float(fileName[sizePath+7:-4])
            #tempList.append(setTemp)
            data = []

            for images in f:
                image = images.split()
                pixel = [i for i in image]
                data.append(pixel)
                
            data = np.array(data,dtype='float32')

        #dic[setTemp] = data
        outputName = 'datasets/spins/L16/Ising2d_L16_spins_T'
        outputName += str(Tcounter)
        outputName += '.pkl.gz'

        with gzip.open(outputName,'wb') as output:
            cPickle.dump(data,output,protocol=cPickle.HIGHEST_PROTOCOL)
        output.close()
        Tcounter += 1

    #dic['Temperatures'] = tempList
    #headerPath = 'datasets/header.dat' 
    #with open (headerPath, "r") as myfile:
    #    header = myfile.read()
            
    # Build test set
    #fullSet = (header,dic)

    # Pickle and gzip the dataset 
    #with gzip.open(outputName, 'wb') as output:
    #    cPickle.dump(fullSet, output, protocol=cPickle.HIGHEST_PROTOCOL)

    #output.close()
    
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


    
    #print len(dic[dic['Temperatures'][0]][0])
#------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str)
    parser.add_argument('--file',type=str)
    parser.add_argument('--model',type=str)

    args = parser.parse_args()

    if args.command == 'pickle':
        Pickle_datasets()

    elif args.command == 'format':
        Format_Dataset_CPP(args.file)

    elif args.command == 'print':
        Print_network(args.model)
    
