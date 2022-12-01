#ifndef CUDA_PAGE_RANK
#define CUDA_PAGE_RANK

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

void calculatePageRankCuda(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);
#endif