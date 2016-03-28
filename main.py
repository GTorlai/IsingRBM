import Restricted_Boltzmann_Machine as RBM
import utilities.network as NET
import utilities.dataset as DAT
import utilities.statistics as STAT
import utilities.plot as plot
import rbm_classifier 
import ising_measurements
import rbm_measurements
import argparse
import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np
from pylab import loadtxt

#-------------------------------------------------

def main():
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                            """ MAIN EXECUTABLE """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    parser = argparse.ArgumentParser()
    parser.add_argument('command',type=str)
    parser.add_argument('--model',help='Name of the model',default='Ising2d')
    parser.add_argument('--L', help='Linear size of the model',type=int)
    parser.add_argument('--D', help='Model dimension', default=2,type=int)
    parser.add_argument('--T', help='Temperature Index', type=int)
    parser.add_argument('--hid',help='Number of hidden units', type=int)
    parser.add_argument('--ep', help='Epochs', default = 2000,type=int)
    parser.add_argument('--bs', help='Batch Size', default = 50 ,type=int)
    parser.add_argument('--lr', help='Learning Rate', default = 0.01,type=float)
    parser.add_argument('--CD', help='Contrastive Divergence', default = 20,type=int)
    parser.add_argument('--L2', help='Regularization', default = 0.0 ,type=float)
    parser.add_argument('--l' , help='Layer sampling', default = 'visible',type=str)
    parser.add_argument('--targ', help='Data Analysis target')
    parser.add_argument('--e', help='Estimator to plot',type=str) 
    
    args = parser.parse_args()

    X = T.matrix()
    n_v = args.L*args.L
    n_machines = 21

    Network = NET.Network(n_v,
                         args.model,
                         args.hid,
                         args.ep,
                         args.bs,
                         args.CD,
                         args.lr,
                         args.L2,
                         args.T)

    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                           """ TRAIN THE NETWORK """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
    if args.command == 'train' :
        
        pathToDataset = DAT.build_TrainPath(args.model,args.T,args.L)

        f = gzip.open(pathToDataset,'rb')
        spins = cPickle.load(f)
        f.close()
    
        train_set = theano.shared(np.asarray(spins,
            dtype = theano.config.floatX), borrow = True)
        
        rbm = RBM.RBM(X,Network)
        
        rbm.Train(train_set,X)
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                           """ SAMPLE FROM NETWORK"""
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif args.command == 'sample':
        
        pathToNetwork = NET.build_networkPath(Network)
        
        Trained_Network = cPickle.load(open(pathToNetwork))
                
        Network.load(Trained_Network)

        rbm = RBM.RBM(X,Network)
        
        rbm.sample(args.l)
   
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                           """ MEASURE OBS ON RBM """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
    elif args.command == 'measureRBM':
        
        pathToNetwork = NET.build_networkPath(Network)
        
        Trained_Network = cPickle.load(open(pathToNetwork))
                
        Network.load(Trained_Network)
 
        rbm = rbm_measurements.rbmMeasure(Network)
 
        [inputName_V,inputName_H] = rbm.build_inputNames(Network)
        outputName = rbm.build_outputName(Network)

        inputFile_V = open(inputName_V,'r')
        inputFile_H = open(inputName_H,'r') 
        outputFile = open(outputName,'w')
        samples_V = loadtxt(inputFile_V)
        samples_H = loadtxt(inputFile_H)

        for k in range(len(samples_V)):
            rbm.get_energy(samples_V[k],samples_H[k])
            rbm.get_magnetization(samples_V[k],samples_H[k])
            rbm.record(outputFile)
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                             """ DATA ANALYSIS """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif args.command == 'statistics':
    
        print ('Analyzing temperature %d\n' % args.T)

        inputName  = STAT.build_inputName(Network,args.targ)
        outputName = STAT.build_outputName(Network,args.targ)
        
        inputFile  = open(inputName, 'r')
        outputFile = open(outputName,'w')
       
        if (args.T == 0) :
            STAT.write_header(outputFile)

        temperature = STAT.load_temperature(args.T)
        
        [obs,data] = STAT.load_data(inputFile)
       
        [avg,err] = STAT.analyze(args.targ,data,obs,n_v,temperature)
        
        if args.targ == 'Ising_measures':
            STAT.write_output(outputFile,avg,err,temperature)

        if args.targ == 'logZ':
            pathToNetwork = NET.build_networkPath(Network)
            Trained_Network = cPickle.load(open(pathToNetwork))  
            NET.update_logZ(pathToNetwork,Trained_Network,avg,err,1000,100)
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                               """ RBM CLASSIFIER"""
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif args.command == 'classify':
        
        outputName = rbm_classifier.build_outputName(Network.infos) 
        outputFile = open(outputName, 'w')

        pathToTestSet = DAT.build_TestPath(args.L)
        
        NLP_matrix = np.zeros((n_machines,210000))

        for k in range(n_machines):
            
            print ('Computing machine # %d' % k) 
            
            pathToNetwork = rbm_classifier.build_networkName(Network.infos,k)
            Trained_Network = cPickle.load(open(pathToNetwork))
            
            CRBM = rbm_classifier.ClassRBM(Trained_Network)
            
            NLP_matrix[k] = CRBM.get_NLP(pathToTestSet)
        
        
        CRBM.Test(NLP_matrix,outputFile)











#    elif args.command == 'measureIsing':
#        
#        ising = ising_measurements.IsingMeasure(Network,2)
#        ising.build_lattice()
# 
#        inputName = ising.build_inputName(Network)
#        outputName =ising.build_outputName(Network)
#
#        inputFile = open(inputName,'r')        
#        outputFile = open(outputName,'w')
#        samples = loadtxt(inputFile)
#        
#        outputFile.write('#  E  E2  M  M2\n')
#        for k in range(len(samples)):
#            ising.get_energy(samples[k])
#            ising.get_magnetization(samples[k])
#            ising.record(outputFile)
#    
#
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
#    elif args.command == 'statistics':
#    
#        print ('Analyzing temperature %d\n' % args.T)
#        
#        inputName = 'data/measurements/L'
#        inputName += str(args.L)
#        inputName += '/MC_Ising2d_L'
#        inputName += str(args.L)
#        inputName += '_T'
#        if (args.T<10):
#            inputName += '0'
#        inputName += str(args.T)
#        inputName += '.txt'
#        
#        outputName = 'data/observables/temp/MC_Ising2d_L'
#        outputName += str(args.L)
#        outputName += '_T'
#        if (args.T<10):
#            outputName += '0'
#        outputName += str(args.T)
#        outputName += '.txt'
# 
#        inputFile  = open(inputName, 'r')
#        outputFile = open(outputName,'w')
#       
#        if (args.T == 0) :
#            STAT.write_header(outputFile)
#
#        temperature = STAT.load_temperature(args.T)
#        
#        [obs,data] = STAT.load_data(inputFile)
#       
#        [avg,err] = STAT.analyze(args.targ,data,obs,n_v,temperature)
#        
#        STAT.write_output(outputFile,avg,err,temperature)
#





if __name__ == "__main__":
    main()
