import Restricted_Boltzmann_Machine as RBM
import utilities.network as NET
import utilities.dataset as DAT
import utilities.statistics as STAT
import argparse
import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np
from pylab import loadtxt

#-------------------------------------------------

def main():
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
    parser.add_argument('--n_meas', help='Number of samples measurements',type=int) 

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
                            
    #""" SAMPLE FROM NETWORK"""
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif args.command == 'sample':
        
        pathToNetwork = NET.build_networkPath(Network)
        
        Trained_Network = cPickle.load(open(pathToNetwork))
                
        Network.load(Trained_Network)

        rbm = RBM.RBM(X,Network)
        
        rbm.sample(args.l,args.n_meas)
   
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
    #""" DATA ANALYSIS """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif args.command == 'statistics':
    
        print ('Analyzing temperature %d\n' % args.T)

        inputName  = STAT.build_inputName(Network,args.targ)
        outputName = STAT.build_outputName(Network,args.targ)
        #inputName  = STAT.build_inputName(Network,'+-1_Ising_measures')
        #outputName = STAT.build_outputName(Network,'+-1_Ising_measures')
 
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
    
    



if __name__ == "__main__":
    main()
