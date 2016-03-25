#!/bin/sh

#Number of Hidden Units
nH=256

#Batch Size
bS=50

#CD order
CD=20

#Epochs
ep=2000

#Size
L=10

#Learning rate
lr=0.01

#Regularization
L2=0.0

#Layer to sample
layer=full

#Temperature Index
t=0

cd ..

for i in {1..21}
do
    sqsub -q NAP_8998 --mpp=2g -r 30m -o output$((nH)).log mpirun --mca mpi_warn_on_fork 0 python main.py measureIsing --L $((L)) --T $((t)) --hid $((nH)) --bs $((bS)) --ep $((ep)) --CD $((CD)) --lr ${lr} --L2 ${L2}
    t=$((t+1))
done
