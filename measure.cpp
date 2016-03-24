#include "observer.cpp"
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <vector>

int main(int argc, char *argv[]) {

    // Variables 
    int MCS = 100000;
    int D = 2;
    int L;
    int T;
    int hid;
    int CD;
    int bS;
    int ep;
    int n_temp = 21;
    string model;
    string L2;
    string lr;
    
    vector<double> temperatures;    
    vector <vector<int> > dataSet;

    // Read inputs from command line

    for (int i=1;i<argc;i++) {
        
        if (strcmp(argv[i],"--model") == 0) {
           model = argv[i+1];
        }
  
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
           hid = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--bS") == 0) {
           bS = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--ep") == 0) {
           ep = atoi(argv[i+1]);
        } 

        if (strcmp(argv[i],"--CD") == 0) {
           CD = atoi(argv[i+1]);
        } 
        
        if (strcmp(argv[i],"--L2") == 0) {
           L2 = argv[i+1];
        } 
        
        if (strcmp(argv[i],"--lr") == 0) {
           lr = argv[i+1];
        }
    }

    cout << endl << "Performing measurement at temperature " << T << endl;
    
    //Initializing classes
    Hypercube cube(L,D);

    Spins sigma;
    Observer obs(sigma,cube,MCS);

    string inputFileTemp = "data/datasets/Ising";
    inputFileTemp += to_string(D);
    inputFileTemp += "d_Temperatures.txt";
    
    if (model == "MC") {

        string inputFileName = "data/samples/L";
        inputFileName += to_string(L);
        inputFileName += "/MC_Ising";
        inputFileName += to_string(D);
        inputFileName += "d_L";
        inputFileName += to_string(L);
        inputFileName += "_T"
        inputFileName += to_string(T);
        inputFileName += ".txt";
        
        string outputFileName = "data/measurements/temp/MC";
        outputFileName += "_Ising";
        outputFileName += to_string(D);
        outputFileName += "d_L";
        outputFileName += to_string(L);
        outputFileName += "_T";
        if (T < 10) {
            outputFileName += "0";
        }
        outputFileName += to_string(T);
        outputFileName += "_measure.txt";
    }

    else if (model =="RBM") {
        string inputFileName = "data/samples/L";
        inputFileName += to_string(L);
        inputFileName += "/RBM_CD";
        inputFileName += to_string(CD);
        inputFileName += "_hid";
        inputFileName += to_string(hid);
        inputFileName += "_bS";
        inputFileName += to_string(bS);
        inputFileName += "_ep";
        inputFileName += to_string(ep);
        inputFileName += "_lr";
        inputFileName += lr;
        inputFileName += "_L";
        inputFileName += L2;
        inputFileName += "_Ising";
        inputFileName += to_string(D);
        inputFileName += "d_L";
        inputFileName += to_string(L);
        inputFileName += "_T";
        inputFileName += to_string(T);
        inputFileName += "_visible_samples.txt";
        
        string outputFileName = "data/measurements/temp/RBM_CD";
        outputFileName += to_string(CD);
        outputFileName += "_hid";
        outputFileName += to_string(hid);
        outputFileName += "_bS";
        outputFileName += to_string(bS);
        outputFileName += "_ep";
        outputFileName += to_string(ep);
        outputFileName += "_Ising";
        outputFileName += to_string(D);
        outputFileName += "d_L";
        outputFileName += to_string(L);
        outputFileName += "_T";
        if (T < 10) {
            outputFileName += "0";
        }
        outputFileName += to_string(T);
        outputFileName += "_measure.txt";
    }
    
    //Open files
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
    
    for (int k = 0; k<MCS; k++) {
            
        sigma.setSpins(dataSet[k]);
        obs.GetEnergy(sigma);
        obs.GetMagnetization(sigma);
        obs.record(sigma);
    }

    obs.GetCorrelationLength(L,cube.Coordinates);
    obs.output(temperatures[T],outputFile);
}
