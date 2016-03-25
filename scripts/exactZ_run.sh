#!/bin/sh

#Number of Hidden Units
nH=2

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

cd ../utilities/

for i in {1..21}
do
    python exact_Z.py --L $((L)) --T $((t)) --CD $((CD)) --hid $((nH)) --bS $((bS)) --ep $((ep)) --lr ${lr} --L2 ${L2}
    t=$((t+1))
done

