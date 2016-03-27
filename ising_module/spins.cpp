#include "spins.h"
//#include "MersenneTwister.h"

//Constructor 1
Spins::Spins(){

    spin.clear(); 

}

//Constructor 2
Spins::Spins(int N_){

    N = N_;

    spin.resize(N,1); //assign every spin as 1

}

//Resize system and set spins to 1
void Spins::resize(int N_){

    N = N_;

    spin.resize(N,1); //assign every spin as 1

}

void Spins::setSpins(vector<int> config) {
    
    spin.clear(); 
    for (int i=0; i<N; i++) {
        spin.push_back(config[i]);
    }
}


//Print function
void Spins::print(){

    for (int i=0;i<spin.size();i++){
        cout<<(spin[i]+1)/2<<" ";
    }//i
    cout<<endl;

}//print

//Print configuration on file
void Spins::filePrint(ofstream & file){

   for (int i=0;i<spin.size();i++){
        file<<(spin[i]+1)/2<<" ";
    }//i
    file<<endl; 
}
