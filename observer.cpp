#include "observer.h"
#include <vector>
#include <math.h>

#define PI 3.14159265

//Constructor
Observer::Observer(Spins & sigma, Hypercube & cube, int dataSize_) {

    N = cube.N;
    D = cube.D;

    dataSize = dataSize_;

    Energy  = 0.0;
    Energy2 = 0.0;
    Magn    = 0.0;
    Magn2   = 0.0;
    Magn4   = 0.0;

    sigma.resize(N);

    //Build the nearest neighbors connections
    NearestNeighbors.resize(N,vector<int>(2*D));
    
    for(int i=0; i<NearestNeighbors.size(); i++) {
    	for(int j=0; j<D; j++) {
    	    NearestNeighbors[i][j] = cube.Neighbors[i][j];
    	    NearestNeighbors[i][j+D] = cube.Negatives[i][j];
    	}//j
    }//i
     
    vector<double> temp;
    temp.assign(N,0);
	
    for(int i=0; i<N; i++) {
	SpinSpinCorr.push_back(temp);
    }//i


}

//Computer the energy in the system
void Observer::GetEnergy(Spins & sigma) {
	
    e = 0.0;

    for(int i=0; i<sigma.N; i++) {
	for(int j=0; j<NearestNeighbors[i].size(); j++) {
	    e += - (2*sigma.spin[i]-1)*(2*sigma.spin[NearestNeighbors[i][j]]-1);
	}//j
    }//i

    e /= 2.0;

}//GetEnergy

//Computer the magnetization in the system
void Observer::GetMagnetization(Spins & sigma) {
    
    m = 0;
    for(int i=0; i<sigma.N; i++) {
        m += 2*sigma.spin[i]-1;
    }//i

}//GetMagnetization

//Reset
void Observer::reset() {

    Energy  = 0.0;
    Energy2 = 0.0;
    Magn    = 0.0;
    Magn2   = 0.0;
    Magn4   = 0.0;

    for(int i=0; i<N; i++) {
    	for(int j=0; j<N; j++) {
    	    SpinSpinCorr[i][j] = 0.0;
        }//j
    }//i

}//reset

//Update the measurements
void Observer::record(Spins & sigma) {

    Energy += e;
    Energy2 += e*e;
    Magn += 1.0*abs(m);
    Magn2 += 1.0*m*m;
    Magn4 += 1.0*m*m*m*m;

    for(int i=0; i<N; i++) {
    	for(int j=0; j<N; j++) {
    	    SpinSpinCorr[i][j] += (2*sigma.spin[i]-1)*(2*sigma.spin[j]-1);
    	}//j
    }//i

}//update

//Computer different correlation lengths
void Observer::GetCorrelationLength(const int & L, vector<vector<int> > &coordinate) {

    double q1 = 2*PI/L;
    double suscept0 =0.0;
    double suscept1 =0.0;

    vector<vector<double> > ConnectedCorr;
    ConnectedCorr.resize(N,vector<double>(N));
	
    for(int i=0; i<N; i++) {
    	for(int j=0; j<N; j++) {
    	    ConnectedCorr[i][j] = SpinSpinCorr[i][j]/(1.0*dataSize);
            suscept0 += ConnectedCorr[i][j];
    	    suscept1 += ConnectedCorr[i][j]*cos(q1*(coordinate[i][0]-coordinate[j][0])); 	
    	}//i
    }//j
    
    CorrLength = (1.0/q1)*sqrt(suscept0/suscept1 - 1.0)/L;
 
}


//Write average on file
void Observer::output(double T, ofstream & file){
    
    file << T << " ";

    file<< Energy/(1.0*dataSize * N) <<" ";
    //file<< Energy2/(1.0*MCS * N * N) <<" ";
    double Cv = Energy2/(1.0*dataSize) - Energy*Energy/(1.0*dataSize*dataSize);
    file<< Cv/(T*T*1.0*N) <<" ";
    file<< Magn/(1.0*dataSize * N) <<" ";
    //file<< Magn2/(1.0*MCS * N*N) <<" ";
    double susc = Magn2/(1.0*dataSize) - Magn*Magn/(1.0*dataSize*dataSize);
    file<< susc/(T*1.0*N) <<" ";
    file<< CorrLength << " ";
    double BinderCumulant = 1.5*(1.0-(1.0/3.0)*(Magn4/(Magn2*Magn2))*dataSize);
    file<< BinderCumulant;
    file << endl;
} 
