#include "utility_functions.h"


void test_matrix(SparseMatrix * mat) {
    uns32 nrows = mat->nrows;
    uns32 ncols = mat->ncols;
    assert(mat->ptrs);
    auto nnz = mat->ptrs[nrows];
    for(uns32 i = 0; i < nrows; ++i){
        assert(mat->ptrs[i] <= nnz);
    }
    for(uns32 j=0; j < nnz; ++j){
        assert(mat->inds[j] < ncols);
    }
}




int getToFrom(string line, uns32 &from, uns32 &to) {
    size_t splitIdx = line.find("\t");
    if (splitIdx == string::npos) return FAILURE;

    from = atoi(line.substr(0, splitIdx).c_str())-1;    // input is not zero-indexed
    to = atoi(line.substr(splitIdx).c_str())-1;

    return SUCCESS;
}



uns32 getNnzPerNode(NodeInfo *nodes, uns32 *oLinks, uns32 N, ifstream &fileS) {

    if (!fileS.is_open()) return 0;

    string currLine, from, to;
    uns32 fromNode, toNode, total;
    total = 0;
    while (fileS.good()) {
        // if (total % 1000000 == 0) {
        //     cout << "Line " << total << endl;
        // }

        fromNode = 0;
        toNode = 0;
        getline(fileS, currLine);

        // get from and to indices
        if (getToFrom(currLine, fromNode, toNode) == FAILURE || 
            fromNode >= N || toNode >= N) continue;

        // set NodeInfo[to]
        ++total;
        nodes[toNode].push_back(fromNode);
        ++oLinks[fromNode];
    }
    return total;
}





#ifdef OPEN_MP_PROJECT

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N) {
    // master thread to figure out how to split up C based on nnzPerRow
    for (uns32 z = 1; z <= N; ++z) {
        adjM->ptrs[z] = adjM->ptrs[z-1] + nodes[z-1].size();
    }


    #pragma omp parallel
    {
        #pragma omp for
        for (uns32 i = 0; i < N; ++i) {
            uns32 currIdx = 0;
            for (uns32 j = adjM->ptrs[i]; j < adjM->ptrs[i+1]; ++j) {
                uns32 colID = nodes[i][currIdx++];
                adjM->inds[j] = colID;
                adjM->vals[j] = 1.0 / oLinks[colID];
            }
        }
    }
}

void computeInverseIndex(SparseMatrix *A) {
   val_t* oldVal = A->vals;
   idx_t* oldInd = A->inds;
   ptr_t* oldPtr = A->ptrs;
  
   A->vals = (val_t*)malloc(oldPtr[A->nrows]*sizeof(val_t));
   A->inds = (idx_t*)malloc(oldPtr[A->nrows]*sizeof(idx_t));
   A->ptrs = (ptr_t*)calloc(A->ncols+1, sizeof(ptr_t));
  
   for(idx_t i=0; i<oldPtr[A->nrows]; ++(A->ptrs[oldInd[i]]), ++i);
   for(idx_t i=1; i<=A->ncols; ++i)
       A->ptrs[i] += A->ptrs[i-1];
   for(idx_t i=0; i<A->nrows; ++i) {
       for(idx_t j=oldPtr[i]; j<oldPtr[i+1]; ++j) {
           A->vals[--(A->ptrs[oldInd[j]])] = oldVal[j];
           A->inds[A->ptrs[oldInd[j]]] = i;
       }
   }
   int temp = A->nrows;
   A->nrows = A->ncols;
   A->ncols = temp;
   if(oldInd)
       free(oldInd);
   if(oldVal)
       free(oldVal);
}
 
void sparsematmult(SparseMatrix * A, SparseMatrix * B, SparseMatrix *C) {
   // computeInverseIndex(B);
   idx_t numThreads = omp_get_max_threads();
   accumulatorPerThread accumulator[numThreads];
   for(idx_t i=0; i<numThreads; ++i) {
       accumulator[i].accum = (val_t*)calloc(B->ncols, sizeof(val_t));
       accumulator[i].index = (idx_t*)calloc(B->ncols, sizeof(idx_t));
   }
   ansPerRow rowAns[A->nrows];
   #pragma omp parallel for schedule(dynamic)
   for(idx_t i=0; i<A->nrows; ++i) {
       int tid = omp_get_thread_num();
       for(idx_t j=A->ptrs[i]; j<A->ptrs[i+1]; ++j) {
           for(idx_t k=B->ptrs[A->inds[j]]; k<B->ptrs[A->inds[j]+1]; ++k) {
               if(accumulator[tid].accum[B->inds[k]] == 0.0) {
                   accumulator[tid].index[accumulator[tid].indexFilled] = B->inds[k];
                   (accumulator[tid].indexFilled)++;
               }
               accumulator[tid].accum[B->inds[k]] += A->vals[j]*B->vals[k];
           }
       }
       rowAns[i].size = accumulator[tid].indexFilled;
       if(rowAns[i].size != 0) {
           rowAns[i].val = (val_t*)malloc((rowAns[i].size)*sizeof(val_t));
           rowAns[i].idx = (idx_t*)malloc((rowAns[i].size)*sizeof(idx_t));
       }
       for(idx_t j=0; j<rowAns[i].size; ++j) {
           rowAns[i].val[j] = accumulator[tid].accum[accumulator[tid].index[j]];
           accumulator[tid].accum[accumulator[tid].index[j]] = 0.0;
           rowAns[i].idx[j] = accumulator[tid].index[j];
           accumulator[tid].index[j] = 0;
       }
       accumulator[tid].indexFilled = 0;
   }
  
   C->reserve(A->nrows, B->ncols, 0);  
   for(idx_t i=1; i<C->nrows; ++i)
       C->ptrs[i] = C->ptrs[i-1]+rowAns[i-1].size;
   C->ptrs[C->nrows] = 0;
   C->reserve(A->nrows, B->ncols, C->ptrs[C->nrows-1]+rowAns[C->nrows-1].size);
   C->ptrs[C->nrows] = C->ptrs[C->nrows-1]+rowAns[C->nrows-1].size;
  
   #pragma omp parallel for schedule(dynamic)
   for(idx_t i=0; i<C->nrows; ++i) {
       for(idx_t j=0; j<rowAns[i].size; j++) {
           C->vals[C->ptrs[i]+j] = rowAns[i].val[j];
           C->inds[C->ptrs[i]+j] = rowAns[i].idx[j];
       }
       if(rowAns[i].val)
           free(rowAns[i].val);
       rowAns[i].val = nullptr;
       if(rowAns[i].idx)
           free(rowAns[i].idx);
       rowAns[i].idx = nullptr;
   }
}

void initializePageRankVector(flt32 *pgRnkV, uns32 N) {
    flt32 baseValue = 1.0 / N;

    #pragma omp parallel
    {
        #pragma omp for
        for (uns32 i = 0; i < N; ++i)
            pgRnkV[i] = baseValue;
    }
}




void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {





}



#elif OPEN_ACC_PROJECT







#else

void createNodeMatrix(SparseMatrix *adjM, NodeInfo *nodes, uns32 *oLinks, uns32 N) {
    uns32 currNnz = 0;
    for (uns32 i = 0; i < N; ++i) {
        adjM->ptrs[i] = currNnz;
        uns32 nnz = nodes[i].size();
        for (uns32 j = 0; j < nnz; ++j) {
            uns32 colID = nodes[i][j];
            adjM->inds[currNnz] = colID;
            adjM->vals[currNnz++] = 1.0 / oLinks[colID];
        }
    }
    adjM->ptrs[N] = currNnz;
}



void initializePageRankVector(flt32 *pgRnkV, uns32 N) {
    flt32 baseValue = 1.0 / N;
    for (uns32 i = 0; i < N; ++i)
        pgRnkV[i] = baseValue;
}







void calculatePageRank(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {
    




}




#endif

