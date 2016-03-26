#!/bin/sh

#Number of Hidden Units
nH=2

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

#Analysis Flag
flag=average

cd ../data/measurements/L$((L))/
tar -xzvf RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_Ising_measures.tar.gz

cd ../../../utilities/

for i in {1..21}
do
    python data_analysis.py obs --flag ${flag} --L $((L)) --T $((t)) --hid $((nH)) --bs $((bS)) --ep $((ep)) --CD $((CD)) --lr ${lr} --L2 ${L2}

 
    t=$((t+1))
done

cd ../data/measurements/L$((L))
rm *.txt

cd ../../observables/temp

cat *.txt >> RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_observables.dat

mv *.dat ../L$((L))

rm *.txt

