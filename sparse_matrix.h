#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX

#include <cerrno>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string.h>

#include "global_types.h"

using namespace std;


class SparseMatrix {
private:

public:
    uns32 nrows;    // number of rows
    uns32 ncols;    // number of columns
    uns32 * inds;   // column indices
    flt32 * vals;   // values
    uns32 * ptrs;   // pointers (start of row in ind/val)

    SparseMatrix(){
        nrows = ncols = 0;
        inds = NULL;
        vals = NULL;
        ptrs = NULL;
    }
    
    void allocate(const uns32 r, const uns32 c); // full size for now
    void reserve(const uns32 r, const uns32 c, const uns32 nnz);
    string info(const string name="") const
    {
        return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
            (ptrs ? to_string(ptrs[nrows]) : "0") + ">";
    }
    void printMatrix();
    void tearDown() {
        if (inds) free(inds);
        if (vals) free(vals);
        if (ptrs) free(ptrs);
    }

};


#endif

