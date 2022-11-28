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




#endif

