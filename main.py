import Restricted_Boltzmann_Machine as RBM
import RBM_stat_model as RBM_SM
import argparse
import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np
import Tools
from pylab import loadtxt

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command',type=str)
    parser.add_argument('dataset',type=str)
    parser.add_argument('--hid',type=int)
    parser.add_argument('--net',type=str)
    parser.add_argument('--ep',default = 100,type=int)
    parser.add_argument('--bs',default = 100,type=int)
    parser.add_argument('--lr',default = 0.1,type=float)
    parser.add_argument('--CD',default = 1,type=int)
    parser.add_argument('--L2',default = 0.001, type=float)
    parser.add_argument('--l',type=str)

    args = parser.parse_args()

    X = T.matrix()

    f = gzip.open(args.dataset,'rb')
    spins = cPickle.load(f)
    f.close()
    
    for i in range(len(args.dataset)):
        if (args.dataset[i] == 'T'):

            temperatureIndex = args.dataset[i+1]
            if (args.dataset[i+2] != '.'):
                temperatureIndex += args.dataset[i+2]
            break

    train_X = theano.shared(np.asarray(spins,
        dtype = theano.config.floatX), borrow = True)
    
    n_v = len(train_X.get_value()[0])
    
    SimPar = Tools.SimPar(args.ep,
                          args.bs,
                          args.CD,
                          args.lr,
                          args.L2,
                          )

    if args.command == 'train' :

        n_h = args.hid

        rbm = RBM.RBM(X,args.dataset,n_v,n_h,SimPar)
        
        rbm.Train(train_X,X,temperatureIndex)

    elif args.command == 'sample':

        pathToNetwork = args.net
        
        Network = cPickle.load(open(pathToNetwork))
        
        n_h = Network['Informations']['Hidden Units']

        rbm = RBM.RBM(X,args.dataset,n_v,n_h,SimPar,
                Network['Parameters'][0], Network['Parameters'][1],
                Network['Parameters'][2])
        
        if (args.l == 'visible'):
            rbm.sample_Ising(Network,temperatureIndex)
        
        if (args.l == 'full'):
            rbm.sample_Full(Network,temperatureIndex)
    
    elif args.command == 'measure':
        
        print "Initializing model..."
        print "\n"

        pathToNetwork = args.net
        
        Network = cPickle.load(open(pathToNetwork))
        
        n_h = Network['Informations']['Hidden Units']
        
        tempFile = open("datasets/temperatures/Ising2d_Temps.txt",'r')
        temps = loadtxt(tempFile)
        
        linear_size = int(np.sqrt(n_v))

        modelName = 'RBM_CD'
        modelName += str(Network['Informations']['CD_order'])
        modelName += '_hid'
        modelName += str(n_h)
        modelName += '_bS'
        modelName += str(Network['Informations']['Batch Size'])
        modelName += '_ep'
        modelName += str(Network['Informations']['Epochs'])
        modelName += '_Ising2d_L'
        modelName += str(linear_size)
        
        
        pathToSamples = 'samples/'
        pathToSamples += modelName
        pathToSamples += '_T'
        pathToSamples += str(temperatureIndex) 
        pathToSamples += str('_full_samples.txt')
        
        pathToOutput = 'measurements/raw/'
        pathToOutput += modelName
        pathToOutput += '_T'
        if (temperatureIndex < 10):
            pathToOutput += '0'
        pathToOutput += str(temperatureIndex) 
        pathToOutput += str('_measure.txt')

        StatRBM = RBM_SM.Stat_model(n_v,n_h,temps[int(temperatureIndex)],SimPar,
                             Network['Parameters'][0],
                             Network['Parameters'][1],
                             Network['Parameters'][2]
                             )

       
        spins = []
        spins_samples = open(pathToSamples,'r')
        
        print "Sampling..."
        
        for configurations in spins_samples:
            configuration = configurations.split()
            spin_list = [(2*int(i)-1) for i in configuration]
            spins.append(spin_list)

        spins = np.array(spins)

        for i in range(StatRBM.MCS):
            [v,h] = StatRBM.setSpins(spins[i])
            
            StatRBM.energy(v,h)
            StatRBM.magnetization(v,h)
            StatRBM.record()
        
        print "...Done"
        print '\n\n'

        StatRBM.measure()
        StatRBM.output(pathToOutput)

if __name__ == "__main__":
    main()
