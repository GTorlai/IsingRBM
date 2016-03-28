import numpy as np
import matplotlib
import argparse
import matplotlib
from pylab import *

def over_H(L,hiddens,estimator):

    basePath = '../data/observables/L'
    basePath += str(L)
    basePath += '/RBM_CD20_hid'

    # Set colors
    colors = ["#FF5858","#3425FF","#43FF7C","#FFF922",
              "#FF68FF","#68C3FF","#FFC368","#AA68FF",
              "#FF5858","#3425FF","#43FF7C","#FFF922",
              "#FF68FF","#68C3FF","#FFC368","#AA68FF"]

    fig = figure(1,figsize=(8,6))
    # Loop through files and plot data

    fileMC = '../data/observables/L'
    fileMC += str(L)
    fileMC += '/MC_Ising2d_L'
    fileMC += str(L)
    fileMC += '_Obs.dat'
 
    dataMC = open(fileMC,'r')
    
    headers = dataMC.readline().lstrip('#').split()
    err_header = 'd'
    err_header += estimator
    
    data_col = list([headers.index(estimator)])
    err_col  = list([headers.index(err_header)])
    x = [i for i in range(21)]

    data = loadtxt(fileMC,usecols=data_col)        
    err  = loadtxt(fileMC,usecols=err_col)
    errorbar(x, data, yerr = err)
    errorbar(x,data,yerr=err,color="k",marker='o',linewidth=0.5,linestyle='-',label='Monte Carlo')
 
    for i,nH in enumerate(hiddens):
        fileName = basePath
        fileName += str(nH)
        fileName += '_bS50_ep2000_lr0.01_L0.0_Ising2d_L'
        fileName += str(L)
        fileName +='_Ising_Obs.dat'
        
        dataFile = open(fileName,'r')
        data_col = list([headers.index(estimator)])
        err_col  = list([headers.index(err_header)])
        data = loadtxt(fileName,usecols=data_col)
        err  = loadtxt(fileName,usecols=err_col)
        print err
        Lab = 'RBM, '
        Lab += str(nH)
        Lab += ' hidden'
        errorbar(x,data,yerr=err,color=colors[i],linewidth=0.5,linestyle='-',label=Lab,marker='^',markersize=8)
    
    
    # Set Label and Tickes
    xlabel(headers[0])
    xlabel('T')
    ylabel(estimator)
    xticks([0,5,10,15,20],[1.0,1.635,2.27,2.905,3.54])
    #yscale('log')

    # Set Legend
    if (estimator == 'M'):
        location = 'upper right'
    else:
        location = 'upper left'

    lgd = legend(loc=location,prop={'size':12})
    if lgd: lgd.draggable(state=True) 

    show()


   
def main():   
    
    # Configure command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid',type=int, nargs='+')
    parser.add_argument('--e',help='Estimator',type=str)
    parser.add_argument('--L',type=int)
    
    # Read command line	
    args = parser.parse_args()	
    
    #over_H(args.L,args.hid,args.e)
    over_L(args.e)

if __name__ == "__main__":

    main()






