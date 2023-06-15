//
// Created by andre on 6/7/23.
//

#ifndef SVD_OMP_ONLY_LIB_GLOBAL_H_
#define SVD_OMP_ONLY_LIB_GLOBAL_H_

#define NT 16
//#define DEBUG
//#define SEQUENTIAL
//#define REPORT
#define OMP
//#define LAPACK
//#define IMKL
//#define TESTS

#define iteratorR(i,j,ld)(((i)*(ld))+(j))
#define iteratorC(i,j,ld)(((j)*(ld))+(i))

// For double precision accuracy in the eigenvalues and eigenvectors, a tolerance of order 10−16 will suffice. Erricos
#define tolerance 1e-16

#endif //SVD_OMP_ONLY_LIB_GLOBAL_H_
