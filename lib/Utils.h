//
// Created by andre on 6/7/23.
//

#ifndef SVD_OMP_ONLY_LIB_UTILS_H_
#define SVD_OMP_ONLY_LIB_UTILS_H_

namespace Thesis{

  enum MATRIX_LAYOUT{
    ROW_MAJOR,
    COL_MAJOR
  };

  int omp_thread_count();
}

#endif //SVD_OMP_ONLY_LIB_UTILS_H_
