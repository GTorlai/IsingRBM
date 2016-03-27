#include "observer.h"
#include <vector>
#include <math.h>

//Constructor
Observer::Observer(Spins & sigma, Hypercube & cube, int dataSize_) {

    N = cube.N;
    D = cube.D;

    dataSize = dataSize_;

    sigma.resize(N);

    //Build the nearest neighbors connections
    NearestNeighbors.resize(N,vector<int>(2*D));
    
    for(int i=0; i<NearestNeighbors.size(); i++) {
    	for(int j=0; j<D; j++) {
    	    NearestNeighbors[i][j] = cube.Neighbors[i][j];
    	    NearestNeighbors[i][j+D] = cube.Negatives[i][j];
    	}//j
    }//i
     
}

//Computer the energy in the system
void Observer::GetEnergy(Spins & sigma) {
	
    e = 0;

    for(int i=0; i<sigma.N; i++) {
	for(int j=0; j<NearestNeighbors[i].size(); j++) {
	    e += - (2*sigma.spin[i]-1)*(2*sigma.spin[NearestNeighbors[i][j]]-1);
	}//j
    }//i

    e /= 2;

}//GetEnergy

//Computer the magnetization in the system
void Observer::GetMagnetization(Spins & sigma) {
    
    m = 0;
    for(int i=0; i<sigma.N; i++) {
        m += 2*sigma.spin[i]-1;
    }//i

}//GetMagnetization

//Write average on file
void Observer::output(ofstream & file){
   
    file<< 1.0*e      <<" ";
    file<< 1.0*e*e    <<" ";
    file<< 1.0*abs(m) <<" ";
    file<< 1.0*m*m    <<" ";
    file << endl;
} 
