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
	
	fileNames = args.fileNames
	if len(fileNames) < 1:
		parser.error("Need at least one file")
	
	# Read file's headers
	f = open(fileNames[0],'r')
	headers = f.readline().lstrip('#').split()
	
	# If no estimator is provided in the argument, write them as list
	if (not args.estimator) or (args.estimator not in headers):
		errorString = "Need to specify on of:\n"
		for head in headers:
			errorString += "\"%s\"" % head + "	"
		parser.error(errorString)	
	
	# Set colors
	colors = ["go", "bo", "ro", "mo","yo","ko","co"]

	fig = figure(1,figsize=(8,6))
	# Loop through files and plot data
	Lab = ['MC','RBM, 16  hidden','RBM, 32 hidden','RBM 64 hidden','RBM 128 hidden','RBM 256 hidden','RBM 512 hidden']


        for i,fileName in enumerate(fileNames):
		dataFile = open(fileName,'r')
		headers = dataFile.readline().lstrip('#').split()
		col = list([headers.index(args.estimator)])
		data = loadtxt(fileName,usecols=col)
                
		plot(data,colors[i],linewidth=0.5,linestyle='-',label=Lab[i])
	
        
        # Set Label and Tickes
	#xlabel(headers[0])
        #ylim((50,101))
	xlabel('T')
	ylabel(args.estimator)
        #xticks([0,5,10,15,20],[1.5,2.0,2.5,3.0,3.5])
	#xticks([0,4,8,12,16,19],[0.06,0.08,0.10,0.12,0.14])
        xticks([0,5,10,15,20],[1.0,1.635,2.27,2.905,3.54])
        #yscale('log')

	# Set Legend
	lgd = legend(loc='upper left')
	if lgd: lgd.draggable(state=True) 

	show()

# MAIN

if __name__ == "__main__":
	main()

