//
// Created by andre on 6/7/23.
//

#include "Utils.h"

int Thesis::omp_thread_count() {
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}
