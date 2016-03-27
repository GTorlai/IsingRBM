#!/bin/sh

#Number of Hidden Units
nH=4

#Temperature Index
t=0

#Batch Size
bS=50

#CD order
CD=20

#Epochs
ep=2000

#Size
L=4

#Learning Rate
lr=0.01

#Regularization
L2=0.0

#K
K=1000

#M
M=100

cd ../utilities/

for i in {1..21}
do
    sqsub -q NAP_8998 --mpp=2g -r 1d -o output$((nH)).log mpirun --mca mpi_warn_on_fork 0 python ais.py --L $((L)) --T $((t)) --beta $((K)) --runs $((M)) --CD $((CD)) --hid $((nH)) --bS $((bS)) --ep $((ep)) --lr ${lr} --L2 ${L2} 
    #python ais.py --L $((L)) --T $((t)) --K $((K)) --M $((M)) --CD $((CD)) --hid $((nH)) --bS $((bS)) --ep $((ep)) --lr ${lr} --L2 ${L2}
    t=$((t+1))
done

