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

def build_TrainPath(Model,Temperature,Size):
    
    name = 'data/datasets/train/L'
    name += str(Size)
    name += '/'
    name += Model
    name += '_L'
    name += str(Size)
    name += '_spins_T'
    if (Temperature < 10):
        name += '0'
        name += str(Temperature)
    else:
        name += str(Temperature)
    name += '.pkl.gz'
    
    return name

#-------------------------------------------------

def build_TestPath(L):

    path  = 'data/datasets/test/Ising2d_L'
    path += str(L) 
    path += '_Test.txt'
    
    return path

#-------------------------------------------------

def pickle_datasets():
    
    """ Create a compact pickled version of the dataset.
        Arguments: name    -> Name for the output file
        Input    : file with the following names:
                   Training_Images.dat, Training_Labels.dat, 
                   Validation_Images.dat, Validation_Labels.dat,
                   header.txt file with description.
        Output   : Pickled file ready to be loaded.
        
    """
    
    path = '../data/datasets/temp/*.txt'

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

        outputName = '../data/datasets/train/L12/Ising2d_L12_spins_T'
        if (Tcounter < 10):
            outputName += '0'
            outputName += str(Tcounter)
        else:
            outputName += str(Tcounter)
        outputName += '.pkl.gz'

        with gzip.open(outputName,'wb') as output:
            cPickle.dump(data,output,protocol=cPickle.HIGHEST_PROTOCOL)
        output.close()
        Tcounter += 1

#------------------------------------------------

def format_dataset_CPP(fileName):
    
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

    args = parser.parse_args()

    if args.command == 'pickle':
        pickle_datasets()

    elif args.command == 'format':
        Format_Dataset_CPP(args.file)

