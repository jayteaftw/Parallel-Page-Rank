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
#include "cuda_page_rank.h"

using namespace std;

#define NodeInfo vector<uns32>

#include "omp.h"  

#ifdef OPEN_MP_PROJECT
    #include "omp.h"

#elif defined(CUDA_PROJECT)
      

#else
// single threaded

#endif


void test_matrix(SparseMatrix * mat);


int getToFrom(string *line, uns32 &from, uns32 &to);
uns32 getNnzPerNode(NodeInfo *nodes, uns32 *oLinks, uns32 N, ifstream &fileS);


void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N);


void initializePageRankVector(flt32 *pgRnkV, uns32 N);



void smmOp(SparseMatrix *mA, SparseMatrix *mB, SparseMatrix *mC);
void smvOp(SparseMatrix *mA, flt32 *dVB, flt32 *dVC);
void hasConverged(flt32 *dVA, flt32 *dVB);
void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);
void calculatePageRank2(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);




#endif


