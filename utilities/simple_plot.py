import os,sys
import numpy as np
import math as m
from pylab import *
import matplotlib
import argparse

# PLOT.PY								  
#	
# Simple script to plot data from file		
#						   

def main():
	
	# Configure command line parser
	parser = argparse.ArgumentParser(description='Plot data from file for various estimators')
	parser.add_argument('fileNames', help='Data files', nargs='+')
	parser.add_argument('--estimator','-e',help='Estimator',type=str)
	
	# Read command line	
 	args = parser.parse_args()	
	
	# Set colors
	colors = ["ko", "bo", "ro", "go","mo","yo","co"]

	fig = figure(1,figsize=(8,6))
	Lab = ['1','2','3','4','5','6','7','8']

        for i,fileName in enumerate(fileNames):
		dataFile = open(fileName,'r')
		headers = dataFile.readline().lstrip('#').split()
		col = list([headers.index(args.estimator)])
		data = loadtxt(fileName,usecols=col)
                
		plot(data,colors[i],linewidth=0.5,linestyle='-',label=Lab[i])
	
        
	ylabel(args.estimator)
        xticks([0,5,10,15,20],[1.0,1.635,2.27,2.905,3.54])
        #yscale('log')

	# Set Legend
        lgd = legend(loc='upper right',prop={'size':12})
	if lgd: lgd.draggable(state=True) 

	show()


if __name__ == "__main__":
	main()

