//
// Created by andre on 6/7/23.
//

#ifndef SVD_OMP_ONLY_LIB_JACOBIMETHOD_H_
#define SVD_OMP_ONLY_LIB_JACOBIMETHOD_H_

#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include "Utils.h"
#include "Matrix.h"
#include "global.h"

namespace Thesis {
enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};
  void omp_dgesvd(SVD_OPTIONS jobu,
                  SVD_OPTIONS jobv,
                  size_t m,
                  size_t n,
                  MATRIX_LAYOUT matrix_layout_A,
                  Matrix &A,
                  size_t lda,
                  Matrix &s,
                  Matrix &U,
                  size_t ldu,
                  Matrix &V,
                  size_t ldv);
}

#endif //SVD_OMP_ONLY_LIB_JACOBIMETHOD_H_
