#include <cerrno>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "sparse_matrix.h"
#include "utility_functions.h"


using namespace std;




int main(int argc, char * argv[])
{
    SparseMatrix adjM;
    ifstream nodeFile;

    if(argc < 3){
        cerr << "Invalid options." << endl << "<program> <input_file> <nodes> [-t <num_threads>]" << endl;
        exit(1);
    }
    string fileName(argv[1]);
    uns32 N = atoi(argv[2]);
    uns32 nthreads = 1;
    if(argc == 5 && strcasecmp(argv[3], "-t") == 0){
        nthreads = atoi(argv[4]);

#ifdef OPEN_MP_PROJECT
        omp_set_num_threads(nthreads);
#endif
    }
    std::cout << "file: " << fileName << endl;
    std::cout << "number of nodes: " << N << endl;
    std::cout << "nthreads: " << nthreads << endl;   

    auto start = chrono::high_resolution_clock::now();


    // Get NodeInfo for each node
    NodeInfo *nodes = new NodeInfo[N];
    uns32 *oLinks = (uns32 *) malloc(sizeof(uns32) * N);
    memset(oLinks, 0x0, sizeof(uns32) * N);
    nodeFile.open(fileName);

    uns32 totalNnz = getNnzPerNode(nodes, oLinks, N, nodeFile);
    flt32 sparcity = (flt32)totalNnz / (N*N);
    cout << "Total number of non-zeros = " << totalNnz << endl;
    cout << "Sparcity = " << sparcity << endl;


    // Create the adjacency matrix from the NodeInfo array
    adjM.reserve(N, N, totalNnz);
    createNodeMatrix(&adjM, nodes, oLinks, N);
    // adjM.printMatrix();
    // test_matrix(&adjM);


    adjM.tearDown();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Duration: " << duration.count() << " ms" << endl;

    return EXIT_SUCCESS;
}



