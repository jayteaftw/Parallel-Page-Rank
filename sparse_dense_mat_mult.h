#ifndef SPARSE_DENSE_MAT_MULT
#define SPARSE_DENSE_MAT_MULT

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


#include "sparse_matrix.h"

#ifdef OPEN_MP_PROJECT
#include "omp.h"

#elif OPEN_ACC_PROJECT

#else
// single threaded
#endif


void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N);
#endif