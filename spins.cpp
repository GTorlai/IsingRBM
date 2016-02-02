#include "spins.h"
#include "MersenneTwister.h"

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

//Randomize the spin values
void Spins::randomize(MTRand & random){

    int ising_spin;
    
    for (int i = 0; i<spin.size(); i++){
        ising_spin = 2*random.randInt(1)-1;

        spin.at(i) = ising_spin;
    }//i


}//randomize

//Perform a single-spin flip
void Spins::flip(int index){

    spin.at(index) *= -1;

}//flip


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
