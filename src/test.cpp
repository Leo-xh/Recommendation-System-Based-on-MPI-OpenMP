#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {
    int  numtasks, rank, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    // initialize MPI
    MPI_Init(&argc, &argv);

    // get number of tasks
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // get my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // this one is obvious
    MPI_Get_processor_name(hostname, &len);
    printf ("Number of tasks= %d My rank= %d Running on %s\n", numtasks, rank, hostname);

    string movieMappingFile = "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/movies.csv";
    stringstream ss;
    string fileLine;
    ifstream movieMapFile(movieMappingFile);
    string movieName;
    getline(movieMapFile, fileLine); // header
    // clock_t begin = clock();
    // cout << "\treading movieMap...";
    while (getline(movieMapFile, fileLine)) {
        cout << fileLine << endl;
    }
    MPI_Finalize();
}
