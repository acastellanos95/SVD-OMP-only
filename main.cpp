#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <random>
#include "lib/Utils.h"
#include "lib/global.h"
#include "lib/JacobiMethod.h"

int main() {
  // SEED!!!
  const unsigned seed = 1000000;

  size_t begin = 20;
  size_t end = 20;
  size_t delta = 1000;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  auto now_time = oss.str();

  std::stringstream file_output;
  file_output << "CPU: AMD Ryzen 3700x CPU  @ 3.6GHz\n";
  std::cout << "CPU: AMD Ryzen 3700x CPU  @ 3.6GHz\n";
  file_output << "Number of threads: " << Thesis::omp_thread_count() << '\n';
  std::cout << "Number of threads: " << Thesis::omp_thread_count() << '\n';
  for (; begin <= end; begin += delta) {
    #ifdef OMP
    {
      double time_avg = 0.0;
      auto times_repeat = 1;
      for(auto i_repeat = 0; i_repeat < times_repeat; ++i_repeat){
        {
          // Build matrix A and R
          /* -------------------------------- Test 1 (Squared matrix SVD) OMP -------------------------------- */
          file_output
              << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";
          std::cout
              << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";

          const size_t height = begin;
          const size_t width = begin;

          file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
          std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

          Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

          const unsigned long A_height = A.height, A_width = A.width;

          std::fill_n(U.elements, U.height * U.width, 0.0);
          std::fill_n(V.elements, V.height * V.width, 0.0);
          std::fill_n(A.elements, A.height * A.width, 0.0);
          std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

          // Create R matrix
          std::random_device random_device;
          std::default_random_engine e(seed);
          std::uniform_real_distribution<double> uniform_dist(0.0, 10.0);
          for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
            for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
              double value = uniform_dist(e);
              A.elements[iteratorC(indexRow, indexCol, A_height)] = value;
              A_copy.elements[iteratorC(indexRow, indexCol, A_height)] = value;
            }
          }

          for (size_t i = 0; i < A.width; ++i) {
            V.elements[iteratorC(i, i, A_width)] = 1.0;
          }

          // Calculate SVD decomposition
          double ti = omp_get_wtime();
          Thesis::omp_dgesvd(Thesis::AllVec,
                             Thesis::AllVec,
                             A.height,
                             A.width,
                             Thesis::COL_MAJOR,
                             A,
                             A_height,
                             s,
                             U,
                             A_height,
                             V,
                             A_width);
          double tf = omp_get_wtime();
          double time = tf - ti;
          time_avg += time;

          file_output << "SVD OMP time with U,V calculation: " << time << "\n";
          std::cout << "SVD OMP time with U,V calculation: " << time << "\n";

          // A - A*
          #pragma omp parallel for
          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
              double value = 0.0;
              for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
                value += U.elements[iteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot]
                    * V.elements[iteratorC(indexCol, k_dot, A_height)];
              }
              A_copy.elements[iteratorC(indexRow, indexCol, A_height)] -= A.elements[iteratorC(indexRow, indexCol, A_height)];
            }
          }

          // Calculate frobenius norm
          double frobenius_norm = 0.0;
          #pragma omp parallel for reduction(+:frobenius_norm)
          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
              double value = A_copy.elements[iteratorC(indexRow, indexCol, A_height)];
              frobenius_norm += value*value;
            }
          }

          file_output << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
          std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
        }
      }

      std::cout << "Tiempo promedio: " << (time_avg / round((double) times_repeat)) << "\n";
    }
    #endif
  }

  std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
  file << file_output.rdbuf();
  file.close();
  return 0;
}
