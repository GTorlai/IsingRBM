#/bin/sh

#Number of Hidden Units
nH=24

#Temperature Index
t=0

#Batch Size
bS=50

#CD order
CD=20

#Epochs
ep=2000

L=4

lr=0.01

L2=0.0

cd ../data/samples/L$((L))

mkdir compression

for i in {1..10}
do
    mv RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_T0$((t))_visible_samples.txt compression/
    mv RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_T0$((t))_hidden_samples.txt compression/
    t=$((t+1))
done
for i in {1..11}
do
    mv RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_T$((t))_visible_samples.txt compression/
    mv RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_T$((t))_hidden_samples.txt compression/
    t=$((t+1))
done

cd compression/

tar -czvf RBM_CD$((CD))_hid$((nH))_bS$((bS))_ep$((ep))_lr0.01_L0.0_Ising2d_L$((L))_samples.tar.gz *.txt

mv *.gz ../
cd ../
rm -rf compression/

