import Restricted_Boltzmann_Machine as RBM
import utilities.network as NET
import utilities.dataset as DAT
#import RBM_stat_model as RBM_SM
import argparse
import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np
#import Tools
#import AIS
#import Exact_Z
#import RBM_Classifier
from pylab import loadtxt


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command',type=str)
    parser.add_argument('--model',default='Ising2d',type=str)
    parser.add_argument('--L',type=int)
    parser.add_argument('--T',type=int)
    parser.add_argument('--hid',type=int)
    parser.add_argument('--ep',default = 2000     ,type=int)
    parser.add_argument('--bs',default = 50       ,type=int)
    parser.add_argument('--lr',default = 0.01     ,type=float)
    parser.add_argument('--CD',default = 20       ,type=int)
    parser.add_argument('--L2',default = 0.0      ,type=float)
    parser.add_argument('--l' ,default = 'visible',type=str)
    parser.add_argument('--sw',default = 100      ,type=int)
    parser.add_argument('--k' ,default = 1000     ,type=int)

    args = parser.parse_args()

    X = T.matrix()
    n_v = args.L**2

    Network = NET.Network(n_v,
                         args.model,
                         args.hid,
                         args.ep,
                         args.bs,
                         args.CD,
                         args.lr,
                         args.L2,
                         args.T)
 
    if args.command == 'train' :
        
        pathToDataset = DAT.build_dataPath(args.model,args.T,args.L)

        f = gzip.open(pathToDataset,'rb')
        spins = cPickle.load(f)
        f.close()
    
        train_set = theano.shared(np.asarray(spins,
            dtype = theano.config.floatX), borrow = True)
        
        #NET.print_network(Network.infos)
        
        rbm = RBM.RBM(X,Network)
        
        rbm.Train(train_set,X)

    elif args.command == 'sample':
        
        pathToNetwork = NET.build_networkPath(Network)
        
        Trained_Network = cPickle.load(open(pathToNetwork))
                
        Network.load(Trained_Network)

        NET.print_network(Network.infos)
 
        rbm = RBM.RBM(X,Network)
        
        rbm.sample(args.l)

#    elif args.command == 'classify':
#
#        pathToNetwork = 'networks/'
#        pathToNetwork += args.net
#        
#        Network = cPickle.load(open(pathToNetwork))
# 
#        n_h = Network['Informations']['Hidden Units']
#                
#        pathToZ = 'partition_functions/L4/Exact_Z'
#        #pathToZ += str(n_h)
#        #pathToZ += '/sw'
#        #pathToZ += str(args.sw)
#        #pathToZ += '/Z_k10000_sw'
#        #pathToZ += str(args.sw)
#        pathToZ += '_T'
#        pathToZ += str(temperatureIndex)
#        pathToZ += '.txt'
#    
#        pathToTestSet = 'datasets/spins/Ising2d_L4_Test.txt'
#
#        f_Z = open(pathToZ,'r')
#        logZ = loadtxt(f_Z)
#        logZ = 1.0 * logZ
#        print logZ
#        
#        ClassRBM = RBM_Classifier.Classifier(n_v,n_h,Network,logZ)
#        ClassRBM.Test(pathToTestSet,temperatureIndex,args.sw)
#
#
#    elif args.command == 'partitionFunction':
#
#        pathToNetwork = 'networks/'
#        pathToNetwork += args.net
#        
#        Network = cPickle.load(open(pathToNetwork))
#        
#        n_h = Network['Informations']['Hidden Units']
#        
#        parameters = [Network['Parameters'][0],
#                      Network['Parameters'][1],
#                      Network['Parameters'][2]]
#
#        exact = Exact_Z.Z(n_v,n_h)
#        exact.get_parameters(parameters)
#        exact.getZ()
#        exact.outputLogZ(Network,temperatureIndex)
#        print ("Exact Partition Funcion: %f\n" % np.log(exact.Z))
#        
#        print 'Annealing..'
#        #annealed = AIS.AnnealedImportanceSampling(n_v,n_h,args.k,args.sw)
#        #annealed.get_parameters(parameters)
#        #annealed.getZ()
#        #annealed.outputLogZ(Network,temperatureIndex)
#        #print ("AIS Partition Funcion: %f\n" % np.log(annealed.Z))        
#
#    elif args.command == 'measure':
#        
#        print "Initializing model..."
#        print "\n"
#        
#        pathToNetwork = 'networks/'
#        pathToNetwork += args.net
#        
#        Network = cPickle.load(open(pathToNetwork))
#        
#        n_h = Network['Informations']['Hidden Units']
#        
#        tempFile = open("datasets/temperatures/Ising2d_Temps.txt",'r')
#        temps = loadtxt(tempFile)
#        
#        linear_size = int(np.sqrt(n_v))
#
#        modelName = 'RBM_CD'
#        modelName += str(Network['Informations']['CD_order'])
#        modelName += '_hid'
#        modelName += str(n_h)
#        modelName += '_bS'
#        modelName += str(Network['Informations']['Batch Size'])
#        modelName += '_ep'
#        modelName += str(Network['Informations']['Epochs'])
#        modelName += '_lr0.01_L0.0'
#        modelName += '_Ising2d_L'
#        modelName += str(linear_size)
#        
#        
#        pathToSamples = 'samples/'
#        pathToSamples += modelName
#        pathToSamples += '_T'
#        pathToSamples += str(temperatureIndex) 
#        pathToSamples += str('_full_samples.txt')
#        
#        pathToOutput = 'measurements/raw/'
#        pathToOutput += modelName
#        pathToOutput += '_T'
#
#        if (int(temperatureIndex) < 10):
#            pathToOutput += str(0)
#        pathToOutput += str(temperatureIndex) 
#        pathToOutput += str('_measure.txt')
#
#        StatRBM = RBM_SM.Stat_model(n_v,n_h,temps[int(temperatureIndex)],SimPar,
#                             Network['Parameters'][0],
#                             Network['Parameters'][1],
#                             Network['Parameters'][2]
#                             )
#
#       
#        spins = []
#        spins_samples = open(pathToSamples,'r')
#        
#        print "Sampling..."
#        
#        for configurations in spins_samples:
#            configuration = configurations.split()
#            spin_list = [(2*int(i)-1) for i in configuration]
#            spins.append(spin_list)
#
#        spins = np.array(spins)
#
#        for i in range(StatRBM.MCS):
#            [v,h] = StatRBM.setSpins(spins[i])
#            
#            StatRBM.energy(v,h)
#            StatRBM.magnetization(v,h)
#            StatRBM.record()
#        
#        print "...Done"
#        print '\n\n'
#
#        StatRBM.measure()
#        StatRBM.output(pathToOutput)

if __name__ == "__main__":
    main()
