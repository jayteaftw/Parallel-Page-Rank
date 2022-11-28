#ifndef UTILITY_FUNCTIONS
#define UTILITY_FUNCTIONS

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "global_types.h"
#include "sparse_matrix.h"

using namespace std;

#define NodeInfo vector<uns32>


void test_matrix(SparseMatrix * mat);


int getToFrom(string *line, uns32 &from, uns32 &to);
uns32 getNnzPerNode(NodeInfo *nodes, uns32 *oLinks, uns32 N, ifstream &fileS);

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N);



// #define OPEN_MP_PROJECT

// #define OPEN_ACC_PROJECT



#ifdef OPEN_MP_PROJECT
#include "omp.h"


#elif OPEN_ACC_PROJECT




#else
// single threaded




#endif


#endif


