CC = g++
NVCC = nvcc

default: openmpver

singlethread: main.cpp utility_functions.cpp sparse_matrix.cpp
	${CC} -std=c++11 -O0 -g -Wall -Wextra -Wno-unused-parameter -o $@ main.cpp utility_functions.cpp sparse_matrix.cpp
openmpver: main.cpp utility_functions.cpp sparse_matrix.cpp
	${CC} -std=c++11 -O0 -g -Wall -Wextra -Wno-unused-parameter -fopenmp -DOPEN_MP_PROJECT -o $@ main.cpp utility_functions.cpp sparse_matrix.cpp
cudaver: main.cpp utility_functions.cpp sparse_matrix.cpp cuda_page_rank.cu
	${NVCC} -std=c++11 -O0 -g -Xcompiler -fopenmp -DCUDA_PROJECT  -o $@ main.cpp utility_functions.cpp sparse_matrix.cpp cuda_page_rank.cu
clean:
	-rm -f singlethread
	-rm -f openmpver
	-rm -f cudaver
