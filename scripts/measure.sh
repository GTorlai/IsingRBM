#!/bin/sh

#Number of Hidden Units
nH=4

#Batch Size
bS=50

#CD order
CD=20

#Epochs
ep=2000

#Size
L=4

#Learning rate
lr=0.01

#Regularization
L2=0.0

#Temperature Index
t=0

cd ../data/samples/L$((L))/
tar -xzvf RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_samples.tar.gz
cd ../../../

for i in {1..21}
do
    python main.py measureIsing --L $((L)) --T $((t)) --hid $((nH)) --bs $((bS)) --ep $((ep)) --CD $((CD)) --lr ${lr} --L2 ${L2}
 
    #sqsub -q NAP_8998 --mpp=2g -r 30m -o output$((nH)).log mpirun --mca mpi_warn_on_fork 0 python main.py measureIsing --L $((L)) --T $((t)) --hid $((nH)) --bs $((bS)) --ep $((ep)) --CD $((CD)) --lr ${lr} --L2 ${L2}
    t=$((t+1))
done

