CC = g++

default: openmpver

singlethread: main.cpp utility_functions.cpp sparse_matrix.cpp
	${CC} -std=c++11 -O0 -g -Wall -Wextra -Wno-unused-parameter -o $@ main.cpp utility_functions.cpp sparse_matrix.cpp sparse_dense_mat_mult.cpp
openmpver: main.cpp utility_functions.cpp sparse_matrix.cpp
	${CC} -std=c++11 -O0 -g -Wall -Wextra -Wno-unused-parameter -fopenmp -DOPEN_MP_PROJECT -o $@ main.cpp utility_functions.cpp sparse_matrix.cpp sparse_dense_mat_mult.cpp
clean:
	-rm -f singlethread
	-rm -f openmpver
