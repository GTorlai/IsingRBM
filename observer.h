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
    
        //Observables
        double Energy; 
        double Energy2;
        double Magn;
        double Magn2;
        double Magn4;
        vector<vector<double> > SpinSpinCorr;
        double CorrLength; //Sandvik definition
    
        //Vector containing NN
        vector<vector<int> > NearestNeighbors;
        
        //Constructor
        Observer(Spins & sigma, Hypercube & cube, int dataSize_);
    
        //Functions
        double GetEnergy(Spins & sigma);
        int GetMagnetization(Spins & sigma);
        void GetCorrelationLength(const int & L, vector<vector<int> > &coordinate);
        void record(double energy, int magn,Spins & sigma);
        void reset();
        void output(double T,ofstream & file);

};
#endif



