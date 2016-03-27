#ifndef OBSERVER_H
#define OBSERVER_H

#include "hypercube.cpp"
#include "spins.cpp"

#include <string>

//Observer.h
//A class that keeps track of observables on Ising spin configurations

class Observer {

    public:
    
        //Number of spins
        int N;
        //Dimension of the lattice
        int D;
        //Number of configurations
        int dataSize;

        int e;
        int m;
    
        //Vector containing NN
        vector<vector<int> > NearestNeighbors;
        
        //Constructor
        Observer(Spins & sigma, Hypercube & cube, int dataSize_);
    
        //Functions
        void GetEnergy(Spins & sigma);
        void GetMagnetization(Spins & sigma);
        void output(ofstream & file);
};
#endif



