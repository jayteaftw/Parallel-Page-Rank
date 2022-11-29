#include "sparse_dense_mat_mult.h"



#ifdef OPEN_MP_PROJECT

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    
    //Idx M: Loop throgh adjM->ptrs array
    #pragma omp parallel for
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }        
    }
}

#elif OPEN_ACC_PROJECT

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    
    
    //Idx M: Loop throgh adjM->ptrs array
    #pragma acc kernels loop copy(adjM, initPgRnkV, finPgRnkV)
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }        
    }
    
}

#else
// single threaded

void SparseDenseMatMult(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N)
{
    
    //Idx M: Loop throgh adjM->ptrs array
    for(uns32 i = 0; i < N; i++){
        //Idx V
        for(uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; j++){
            finPgRnkV[i] += adjM->vals[j] *initPgRnkV[adjM->inds[j]];
        }        
    }
}


#endif
