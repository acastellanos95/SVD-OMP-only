//
// Created by andre on 6/7/23.
//

#ifndef SVD_OMP_ONLY_LIB_MATRIX_H_
#define SVD_OMP_ONLY_LIB_MATRIX_H_

struct Matrix{
  unsigned long width{};
  unsigned long height{};
  double *elements{};

  Matrix()= default;

  Matrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  ~Matrix(){
    delete []elements;
  }
};

#endif //SVD_OMP_ONLY_LIB_MATRIX_H_
