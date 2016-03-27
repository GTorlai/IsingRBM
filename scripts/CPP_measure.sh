#!/bin/sh

#Number of Hidden Units
nH=64

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

#Temperature Index
t=0

cd ../data/samples/L$((L))/
tar -xzvf RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_samples.tar.gz
cd ../../../ising_measurement_module

g++ measure.cpp -O2 -o measure

for i in {1..21}
do
    ./measure --D 2 --L $((L)) --T $((t)) --hid $((nH)) --bs $((bS)) --ep $((ep)) --CD $((CD)) --lr ${lr} --L2 ${L2}
 
    t=$((t+1))
done

cd ../data/measurements/L$((L))/

tar -czvf RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_Ising_measures.tar.gz *hid$((nH))*

rm *.txt

cd ../../samples/L$((L))/

rm *.txt
