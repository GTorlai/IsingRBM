#ifndef SPINS_H
#define SPINS_H

// spins.h
// Class describing Ising spins on a lattice

#include <vector>
#include <fstream>

using namespace std;

class Spins {
    
    public:
    
        int N;  //Number of spins

        //Vector containing the spins values
        vector<int> spin;

        //Functions
        Spins(int N_);
        Spins();
        void resize(int N_);
        void setSpins(vector<int> config); 
        void print();
        void filePrint(ofstream & file); 
};

#endif
