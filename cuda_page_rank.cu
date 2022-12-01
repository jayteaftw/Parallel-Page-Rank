
#include "cuda_page_rank.h"

__global__
void cudaCalculate(int n, uns32*ptrs, uns32 *inds, flt32 *vals, flt32 *x, flt32 *y){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        for(uns32 j = ptrs[i]; j < ptrs[i+1]; j++){
            y[i] += vals[j] *x[inds[j]];
        }  
    }
}

__global__
void cudaSwap(int n, flt32 *x, flt32 *y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        x[i] = y[i];
        y[i] = 0;
    }

}

void calculatePageRankCuda(SparseMatrix *adjM, flt32 *initPgRnkV, flt32 *finPgRnkV, uns32 N) {

    flt32 min_error = 0.0001;
    flt32 cur_error = min_error;
    cout<<"Cuda Check"<<endl;
    flt32 *d_x, *d_y, *vals;
    uns32 *inds, *ptrs;
    uns32 size = adjM->ptrs[N];
    cudaMalloc(&d_x, N*sizeof(flt32)); 
    cudaMalloc(&d_y, N*sizeof(flt32));
    cudaMalloc(&ptrs, (N+1)*sizeof(uns32));
    cudaMalloc(&vals, (size)*sizeof(flt32));
    cudaMalloc(&inds, (size)*sizeof(uns32));

    cout<<"before cuda calc"<<endl;
    cout<<initPgRnkV[0]<<" "<<initPgRnkV[10]<<" "<<initPgRnkV[32]<<" "<<initPgRnkV[1000]<<" "<<endl; 
    cout<<finPgRnkV[0]<<" "<<finPgRnkV[10]<<" "<<finPgRnkV[32]<<" "<<finPgRnkV[1000]<<" "<<endl;   

    
    cudaMemcpy(ptrs, adjM->ptrs, (N+1)*sizeof(uns32), cudaMemcpyHostToDevice);
    cudaMemcpy(inds, adjM->inds, size*sizeof(uns32), cudaMemcpyHostToDevice);
    cudaMemcpy(vals, adjM->vals, size*sizeof(flt32), cudaMemcpyHostToDevice);

    for(int iter =0; iter < 10000; iter++){      
        
        cudaMemcpy(d_x, initPgRnkV, N*sizeof(flt32), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, finPgRnkV, N*sizeof(flt32), cudaMemcpyHostToDevice);
        uns32 block_size = 1024;

        for(int i = 0; i < 1000; i++)
        {
            cudaCalculate<<<(N+(block_size-1))/block_size, block_size>>>(N, ptrs, inds, vals, d_x, d_y);
            cudaSwap<<<N+(block_size-1)/block_size, block_size>>>(N, d_x, d_y);
        }

        cudaCalculate<<<(N+(block_size-1))/block_size, block_size>>>(N, ptrs, inds, vals, d_x, d_y);
        cudaMemcpy(finPgRnkV, d_y, N*sizeof(flt32), cudaMemcpyDeviceToHost);
        cudaMemcpy(initPgRnkV, d_x, N*sizeof(flt32), cudaMemcpyDeviceToHost);

        cout<<"After cuda calc"<<endl;
        cout<<"init "<<initPgRnkV[0]<<" "<<initPgRnkV[10]<<" "<<initPgRnkV[32]<<" "<<initPgRnkV[1000]<<" "<<endl; 
        cout<<"fin  "<<finPgRnkV[0]<<" "<<finPgRnkV[10]<<" "<<finPgRnkV[32]<<" "<<finPgRnkV[1000]<<" "<<endl<<endl;

        
        cur_error = 0;
        #pragma omp parallel for reduction(+:cur_error)
        for(uns32 idx = 0; idx < N; idx++){
            cur_error += abs(finPgRnkV[idx] - initPgRnkV[idx]);
            initPgRnkV[idx] = finPgRnkV[idx];
            finPgRnkV[idx] = 0;
        }
        cout<<"iter: "<<iter<<" error: "<<cur_error<<endl;
        if (cur_error < min_error){
            cout<<"iter: "<<iter<<" Final Error: "<< cur_error <<endl;
            break;
        }
        
    } 
    cout<<initPgRnkV[0]<<" "<<initPgRnkV[10]<<" "<<initPgRnkV[32]<<" "<<initPgRnkV[1000]<<" "<<endl; 

}