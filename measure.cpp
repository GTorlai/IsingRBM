#include "Observer.cpp"
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <vector>

int main(int argc, char *argv[]) {

    
    int MCS = 100000;
    int D;
    int L;
    int T;
    int n_H;
    int CD = 15;
    int bS = 100;
    int n_temp = 21;
        
    vector<double> temperatures;    
    vector <vector<int> > dataSet;

    for (int i=1;i<argc;i++) {
        
        if (strcmp(argv[i],"--D") == 0) {
           D = atoi(argv[i+1]);
        }
        
        if (strcmp(argv[i],"--L") == 0) {
           L = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--T") == 0) {
           T = atoi(argv[i+1]);
        } 

        if (strcmp(argv[i],"--hid") == 0) {
           n_H = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--bS") == 0) {
           bS = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--CD") == 0) {
           CD = atoi(argv[i+1]);
        } 
         
    }

    cout << "Initializing...";

    Hypercube cube(L,D);

    Spins sigma;
    Observer obs(sigma,cube,MCS);


    string inputFileTemp = "datasets/temperatures/Ising";
    inputFileTemp += to_string(D);
    inputFileTemp += "d_Temps.txt";

    string inputFileName = "samples/RBM_CD";
    inputFileName += to_string(CD);
    inputFileName += "_hid";
    inputFileName += to_string(n_H);
    inputFileName += "_Ising";
    inputFileName += to_string(D);
    inputFileName += "d_L";
    inputFileName += to_string(L);
    inputFileName += "_T";
    inputFileName += to_string(T);
    inputFileName += "_samples.txt";
    
    string outputFileName = "measurements/raw/RBM_CD";
    outputFileName += to_string(CD);
    outputFileName += "_hid";
    outputFileName += to_string(n_H);
    outputFileName += "_Ising";
    outputFileName += to_string(D);
    outputFileName += "d_L";
    outputFileName += to_string(L);
    outputFileName += "_T";
    outputFileName += to_string(T);
    outputFileName += "_measure.txt";
    
    ifstream inputTemp(inputFileTemp, std::ios_base::in);
    ifstream inputFile(inputFileName, std::ios_base::in);
    ofstream outputFile(outputFileName);
    

    temperatures.assign(n_temp,0.0);

    vector<int> temp1;
    temp1.assign(sigma.N,0);

    for (int i=0; i<MCS; i++) {
        dataSet.push_back(temp1);
    }

    cout << "done" << endl;

    //Inport Temperatures
    for(int i=0; i<n_temp; i++) {
        inputTemp >> temperatures[i];
    }
    
    //Import dataset from file
    for (int i=0; i<MCS; i++) {
        for (int j=0; j<sigma.N; j++) {
            inputFile >> dataSet[i][j];
        }
    }

    
    cout << "Running...";
    
    for (int k = 0; k<MCS; k++) {
            
        sigma.setSpins(dataSet[k]);
        obs.GetEnergy(sigma);
        obs.GetMagnetization(sigma);
        obs.record(sigma);
    }

    cout << "done" << endl;

    obs.GetCorrelationLength(L,cube.Coordinates);
    obs.output(temperatures[T],outputFile);
}
