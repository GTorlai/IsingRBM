#ifndef HYPERCUBE_H
#define HYPERCUBE_H

//hypercube.h
//Class that build hypercube lattice in D dimensions

#include <vector>
#include <iostream>

using namespace std;

class Hypercube {
    
    public:
        
        //Linear Size
        int L;
        //Dimension
        int D;
        //Total number of sites
        int N;

        vector<vector<int> > Neighbors; //neighbors with positive offsets
        vector<vector<int> > Negatives; //neighbors with negative offsets
        vector<vector<int> > Coordinates;

        //Functions
        Hypercube(int L_, int D_);
        void print();
         
    private:
        int myPow(int, int);

};

#endif
