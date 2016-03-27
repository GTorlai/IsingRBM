
#include "hypercube.h"

#define PRINT_RED(x) std::cout << "\033[1;31m" << x << "\033[0m" << " "
#define PRINT_BLUE(x) std::cout << "\033[1;34m" << x << "\033[0m" << " "
#define PRINT_GREEN(x) std::cout << "\033[1;32m" << x << "\033[0m" << " "
#define PRINT_YELLOW(x) std::cout << "\033[1;33m" << x << "\033[0m" << " "

//Constructor
Hypercube::Hypercube(int L_, int D_){

    L = L_;
    D = D_;
    N = myPow(L,D);

    //build the nearest-neighbor connections
    vector<int> temp;  //the "inner" vector
    int pair; //the bond pair index
    for (int i=0;i<N;i++){

        temp.clear();
        for (int j=0;j<D;j++){
            if ( ( (i+myPow(L,j))%(myPow(L,(j+1))) >= 0)  //TODO: this line not needed
                    && ( (i+myPow(L,j))%(myPow(L,(j+1))) < myPow(L,j) ) )
                pair = i+myPow(L,j) - myPow(L,(j+1));
            else
                pair = i+myPow(L,j);

            temp.push_back(pair);
        }//j

        Neighbors.push_back(temp);
    }//i

    //Negative neighbors
	Negatives = Neighbors;

    for (int i=0;i<N;i++){
        temp.clear();
        for (int j=0;j<D;j++){
            if ( ( (i+myPow(L,j))%(myPow(L,(j+1))) >= 0)  //TODO: this line not needed
                    && ( (i+myPow(L,j))%(myPow(L,(j+1))) < myPow(L,j) ) )
                pair = i+myPow(L,j) - myPow(L,(j+1));
            else
                pair = i+myPow(L,j);

            Negatives[pair][j] = i;
        }//j

    }//i


    //build the (x,y,z,...) coordinates of each lattice site
    temp.clear();
    temp.assign(D,0);  //D integers with value 0

    for (int i=0;i<N;i++){
        Coordinates.push_back(temp);

        if ( (temp[0]+1) % L == 0){ //end of x-row

            temp[0] = 0; //reset

            for (int j=1;j<D;j++){
                if ( (temp[j]+1) % L == 0)
                    temp[j] = 0; //reset
                else{
                    temp[j]++;
                    break;
                }//else
            }//j
        }//if
        else
            temp[0]++;
    }//i

}//constructor

//Print function
void Hypercube::print(){

    cout<<"L D N \n";
    cout<<L<<" "<<D<<" "<<N<<endl;

    cout<<"Neighbor list:"<<endl;
    for (int i=0;i<Neighbors.size();i++){
        cout<<i<<" ";
        for (int j=0;j<Neighbors[i].size();j++){
            PRINT_RED(Neighbors[i][j]);
        }//j
        cout<<endl;
    }//i

	cout << endl;

    cout<<"Negative Neighbor list:"<<endl;
    for (int i=0;i<Negatives.size();i++){
        cout<<i<<" ";
        for (int j=0;j<Negatives[i].size();j++){
            PRINT_GREEN(Negatives[i][j]);
        }//j
        cout<<endl;
    }//i

	cout << endl;

    cout<<"Coordinates:"<<endl;
	for (int i=0;i<Coordinates.size();i++){
        cout<<i<<" ";
        for (int j=0;j<Coordinates[i].size();j++){
            cout<<j<<" ";
            cout<<Coordinates[i][j]<<" ";
			PRINT_YELLOW(Coordinates[i][j]);

        }//j
        cout<<endl;
    }//i

	cout << endl;

}//print

//a simple function to calculate powers of an integer: not for general use
int Hypercube::myPow (int x, int p) {
  int i = 1;
  for (int j = 1; j <= p; j++)  i *= x;
  return i;
}
