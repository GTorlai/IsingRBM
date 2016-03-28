import gzip
import cPickle
import numpy as np
import theano
import argparse
import glob
import network as NET
import math as m

#-------------------------------------------------

def load_data(dataFile):
    
    """ Return the dimensions of the data matrix """
    
    # Observables names
    header = dataFile.readline().lstrip('#').split()
    # Data
    data = np.loadtxt(dataFile)
    return [header,data]

#-------------------------------------------------

def get_average(data):

    """ Compute averages of data """
    
    n_meas = len(data)
    
    if len(data.shape) != 1:
        n_obs  = len(data[0])
        avg = np.zeros((n_obs))
        for i in range(n_meas):
            for j in range(n_obs):
                avg[j] += data[i,j]
 
    else:
        avg = 0.0
        for i in range(n_meas):
            avg += data[i]
        
    avg /= 1.0*n_meas
    return avg

#-------------------------------------------------

def binning_error(data):

    """ Compute the binning analysis"""
 
    B = data[:]     # copy the dataset

    # Determine  number of binning levels
    min_bin = 25    # minimum bin size
    if B.shape[0]<min_bin: Nl = 1 
    else:                  Nl = int(m.floor(m.log(B.shape[0]/min_bin,2))+1)
    
    
    if B.ndim == 1: B.resize(B.shape[0],1)  # reshape if 1D array
    D = np.zeros((Nl, B.shape[1]))          # initialize binning variable
   
    # First level of binning is raw data
    D[0,:] = np.std(data,0)/m.sqrt(B.shape[0]-1)
    #print MC[:10,0] 
    
    TruncN = 0
    # Binning loop over levels l
    for l in range(1,Nl):
        # Bin pairs of bins: if odd # of bins, truncate first bin
        if ((B.shape[0] % 2) == 0):
            B = (B[::2,:]+ B[1::2,:])/2.0
        else:
            TruncIndex = TruncN * (2**(Nl-l))    
            Averaged = (B[TruncIndex]+B[TruncIndex+1]+B[TruncIndex+2])/3.0
            if TruncIndex == 0:
                SecondHalf = (B[TruncIndex+3::2,:]+ B[TruncIndex+4::2,:])/2.0
                B = np.concatenate(([Averaged],SecondHalf),axis=0)
            else:
                FirstHalf  = (B[0:TruncIndex-2:2,:]+ B[1:TruncIndex-1:2,:])/2.0 
                SecondHalf = (B[TruncIndex+3::2,:]+ B[TruncIndex+4::2,:])/2.0
                B = np.concatenate((FirstHalf,[Averaged],SecondHalf),axis=0)
            TruncN = TruncN + 1 
        
        D[l,:] = np.std(B,0)/m.sqrt(B.shape[0]-1)
    
    return D

#-------------------------------------------------

def logPartitionFunction(data):

    """ Compute mean and error of partition function"""

    average = get_average(data)
    Bin = binning_error(data)
    error = np.amax(Bin)

    return [average,error]

#-------------------------------------------------

def observables(data,obs,N,T):

    """ Compute mean and error of observables """

    avg = get_average(data)
    Bin = binning_error(data)

    err = np.zeros(avg.shape)
    o   = {}

    for j in range(len(avg)):
        o[obs[j]] = j
        err[j] = np.amax(Bin[j])
    
    average = {}
    errors = {}

    average['E'] = 1.0*avg[o['E']] / N
    average['M'] = 1.0*avg[o['M']] / N
    average['C'] = (1.0*avg[o['E2']]-1.0*avg[o['E']]**2) / (N*T**2)
    average['S'] = (1.0*avg[o['M2']]-1.0*avg[o['M']]**2) / (N*T)
    
    errors['E'] = 1.0*err[o['E']] / (1.0*N)
    errors['M'] = 1.0*err[o['E']] / N 
    errors['C'] = m.sqrt((err[o['E2']] / N)**2 + (err[o['E']] / N)**2)  
    errors['S'] = m.sqrt((err[o['M2']] / N)**2 + (err[o['M']] / N)**2)  
 
    #errors['C'] = err[o['E2']] / N + 2.0*err[o['E']] / N  
    #errors['S'] = 1.0*err[o['M2']] / N + 2.0*err[o['M']] / N
    
    return [average,errors]

#-------------------------------------------------

def analyze(target,data,obs,N,T):

    """ Analyze measurements """

    if target == 'Ising_measures':

        [avg,err] = observables(data,obs,N,T)

    if target == 'logZ':

        [avg,err] = logPartitionFunction(data)

    return [avg,err]

#-------------------------------------------------

def print_output(avg,err):
    
    print ('Energy         : %.5f +- %f\n' % (avg['E'],err['E']))
    print ('Magnetization  : %.5f +- %f\n' % (avg['M'],err['M']))
    print ('Specific Heat  : %.5f +- %f\n' % (avg['C'],err['C']))
    print ('Susceptibility : %.5f +- %f\n' % (avg['S'],err['S']))

#-------------------------------------------------

def write_output(outputFile,avg,err,T):
   

    outputFile.write('%.3f  ' % T)
    outputFile.write('%.5f  ' % avg['E']) 
    outputFile.write('%.5f  ' % avg['M']) 
    outputFile.write('%.5f  ' % avg['C']) 
    outputFile.write('%.5f  ' % avg['S']) 

    outputFile.write('%f  ' % err['E']) 
    outputFile.write('%f  ' % err['M']) 
    outputFile.write('%f  ' % err['C']) 
    outputFile.write('%f  ' % err['S']) 
   
 
    outputFile.write('\n')

#-------------------------------------------------

def write_header(outputFile):
    
    outputFile.write('#   T\t      E')
    outputFile.write('        M')
    outputFile.write('\tC')
    outputFile.write('\t S')
    outputFile.write('\t  dE')
    outputFile.write('\t    dM')
    outputFile.write('        dC')
    outputFile.write('\tdS\n')

#-------------------------------------------------

def build_inputName(Network,target):
    
    L = int(np.sqrt(Network.infos['Visible Units']))
    dataFileName = 'data/measurements/L'
    dataFileName += str(L)
    dataFileName += '/'
    dataFileName += NET.build_fileName(Network,target)

    return dataFileName

#-------------------------------------------------

def load_temperature(T_index):

    temperatureFile = open('data/datasets/Ising2d_Temperatures.txt','r')
    T_array = np.loadtxt(temperatureFile)
    T = T_array[T_index]

    return T
 
#-------------------------------------------------

def build_outputName(Network,target):

    outputName = 'data/observables/temp/'
    outputName += NET.build_fileName(Network,'observables')
 
    return outputName

#-------------------------------------------------

def significant(x):
    return round(x,-int(m.floor(m.log10(abs(x)))))

#------------------------------------------------

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('command',type=str)
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
    parser.add_argument('--l', help='Binning Level', type=str)
    parser.add_argument('--targ', help='Data',  type=str)

    args = parser.parse_args()

    n_v = args.L*args.L

    Network = NET.Network(n_v,
                          args.model,
                          args.hid,
                          args.ep,
                          args.bs,
                          args.CD,
                          args.lr,
                          args.L2,
                          args.T)
    
    dataFileName = '../data/measurements/L'
    dataFileName += str(args.L)
    dataFileName += '/'
    dataFileName += NET.build_fileName(Network,args.targ)
    
    outputName = '../data/observables/temp/'
    outputName += NET.build_fileName(Network,'Ising_observables')
    outputFile = open(outputName,'w')
 
    temperatureFile = open('../data/datasets/Ising2d_Temperatures.txt','r')
    T_array = np.loadtxt(temperatureFile)
    T = T_array[args.T]
 
    dataFile = open(dataFileName,'r')
    [obs,data] = load_data(dataFile)
    
    [avg,err] = observables(obs,data,n_v,T)
     
    write_output(outputFile,avg,err,T)    

