import Restricted_Boltzmann_Machine as RBM
import argparse
import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np
import Tools

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command',type=str)
    parser.add_argument('dataset',type=str)
    parser.add_argument('--hid',type=int)
    parser.add_argument('--net',type=str)
    parser.add_argument('--T',type = int)
    parser.add_argument('--ep',default = 100,type=int)
    parser.add_argument('--bs',default = 100,type=int)
    parser.add_argument('--lr',default = 0.001,type=float)
    parser.add_argument('--CD',default = 1,type=int)
    parser.add_argument('--L2',default = 0.001, type=float)

    args = parser.parse_args()

    X = T.matrix()
    
    f = gzip.open(args.dataset,'rb')
    header, dic = cPickle.load(f)
    f.close()
    print header

    temp = dic['Temperatures'][args.T]

    train_X = theano.shared(np.asarray(dic[temp],
        dtype = theano.config.floatX), borrow = True)
   
    n_v = len(train_X.get_value()[0])
    
    AIS_c = 10000
    AIS_s = 1000

    SimPar = Tools.SimPar(args.ep,
                           args.bs,
                           args.CD,
                           args.lr,
                           args.L2,
                           AIS_c,
                           AIS_s)

    if args.command == 'train' :

        n_h = args.hid

        rbm = RBM.RBM(X,args.dataset,n_v,n_h,SimPar)
        
        rbm.Train(train_X,X,args.T)

    elif args.command == 'sample':

        pathToNetwork = args.net
        
        Network = cPickle.load(open(pathToNetwork))
        
        n_h = Network['Informations']['Hidden Units']

        rbm = RBM.RBM(X,args.dataset,n_v,n_h,SimPar,
                Network['Parameters'][0], Network['Parameters'][1],
                Network['Parameters'][2])
        
        rbm.sample(Network,args.T)
        

if __name__ == "__main__":
    main()
