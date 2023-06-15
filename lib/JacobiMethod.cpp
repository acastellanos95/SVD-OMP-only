//
// Created by andre on 6/7/23.
//

#include "JacobiMethod.h"

namespace Thesis {
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
                size_t ldv) {

  if (matrix_layout_A == COL_MAJOR) {

    size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
    // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
    // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
    size_t istop = 0;
    size_t stop_condition = n * (n - 1) / 2;
    size_t maxIterations = 1;

    for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
      // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
      // 1971
      for (size_t k = 1; k < m_ordering; ++k) {
        size_t p = 0;
        size_t p_trans = 0;
        size_t q_trans = 0;
        #pragma omp parallel for private(p, p_trans, q_trans)
        for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
          if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
            p = ((2 * m_ordering) - (2 * k) + 1) - q;
          } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
            p = ((4 * m_ordering) - (2 * k)) - q;
          } else if ((2 * m_ordering) - k - 1 < q) {
            p = n;
          }

          // Translate to (0,0)
          p_trans = p - 1;
          q_trans = q - 1;

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorC(i, p_trans, lda)];
            tmp_q = A.elements[iteratorC(i, q_trans, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // Schur
          double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

          if (std::abs(apq) > tolerance) {

//            #pragma omp critical
//            std::cout << "a_pq before jacobi: " << apq << '\n';

            double tau = (aqq - app) / (2.0 * apq);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            double tmp_A_p, tmp_A_q;
            for (size_t i = 0; i < m; ++i) {
              tmp_A_p = A.elements[iteratorC(i, p_trans, lda)];
              tmp_A_q = A.elements[iteratorC(i, q_trans, lda)];
              tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
              tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
              A.elements[iteratorC(i, p_trans, lda)] = tmp_p;
              A.elements[iteratorC(i, q_trans, lda)] = tmp_q;
            }

            /*
            double value = 0.0;
            for (size_t i = 0; i < m; ++i) {
              tmp_p = A.elements[iteratorC(i, p_trans, lda)];
              tmp_q = A.elements[iteratorC(i, q_trans, lda)];
              value += tmp_p * tmp_q;
            }

            # pragma omp critical
            std::cout << "a_pq after jacobi: " << value << '\n';
             */

            if (jobv == AllVec || jobv == SomeVec) {
              for (size_t i = 0; i < n; ++i) {
                tmp_p =
                    c_schur * V.elements[iteratorC(i, p_trans, ldv)] - s_schur * V.elements[iteratorC(i, q_trans, ldv)];
                tmp_q =
                    s_schur * V.elements[iteratorC(i, p_trans, ldv)] + c_schur * V.elements[iteratorC(i, q_trans, ldv)];
                V.elements[iteratorC(i, p_trans, ldv)] = tmp_p;
                V.elements[iteratorC(i, q_trans, ldv)] = tmp_q;
              }
            }
          }
        }
      }

      for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
        size_t p = 0;
        size_t p_trans = 0;
        size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
        for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
          if (q < (2 * m_ordering) - k + 1) {
            p = n;
          } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
            p = ((4 * m_ordering) - (2 * k)) - q;
          } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
            p = ((6 * m_ordering) - (2 * k) - 1) - q;
          }

          // Translate to (0,0)
          p_trans = p - 1;
          q_trans = q - 1;

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorC(i, p_trans, lda)];
            tmp_q = A.elements[iteratorC(i, q_trans, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // Schur
          double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

          if (std::abs(apq) > tolerance) {
            double tau = (aqq - app) / (2.0 * apq);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            double tmp_A_p, tmp_A_q;
            for (size_t i = 0; i < m; ++i) {
              tmp_A_p = A.elements[iteratorC(i, p_trans, lda)];
              tmp_A_q = A.elements[iteratorC(i, q_trans, lda)];
              tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
              tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
              A.elements[iteratorC(i, p_trans, lda)] = tmp_p;
              A.elements[iteratorC(i, q_trans, lda)] = tmp_q;
            }

            if (jobv == AllVec || jobv == SomeVec) {
              for (size_t i = 0; i < n; ++i) {
                tmp_p =
                    c_schur * V.elements[iteratorC(i, p_trans, ldv)] - s_schur * V.elements[iteratorC(i, q_trans, ldv)];
                tmp_q =
                    s_schur * V.elements[iteratorC(i, p_trans, ldv)] + c_schur * V.elements[iteratorC(i, q_trans, ldv)];
                V.elements[iteratorC(i, p_trans, ldv)] = tmp_p;
                V.elements[iteratorC(i, q_trans, ldv)] = tmp_q;
              }
            }
          }

#ifdef DEBUG
          // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
        }
      }
    }

    // Compute \Sigma
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
      }
      s.elements[k] = sqrt(s.elements[k]);
    }

    //Compute U
    if (jobu == AllVec) {
#pragma omp parallel for
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
          U.elements[iteratorC(j, i, ldu)] = A.elements[iteratorC(j, i, ldu)] / s.elements[i];
        }
      }
    } else if (jobu == SomeVec) {
#pragma omp parallel for
      for (size_t k = 0; k < std::min(m, n); ++k) {
        for (size_t i = 0; i < m; ++i) {
          U.elements[iteratorC(i, k, ldu)] = A.elements[iteratorC(i, k, ldu)] / s.elements[k];
        }
      }
    }
  } else if (matrix_layout_A == ROW_MAJOR) {

    size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
    // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
    // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
    size_t istop = 0;
    size_t stop_condition = n * (n - 1) / 2;
    uint16_t reps = 0;
    uint16_t maxIterations = 1;

    do {
      istop = 0;
      // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
      // 1971
      for (size_t k = 1; k < m_ordering; ++k) {
        size_t p = 0;
        size_t p_trans = 0;
        size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
        for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
          if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
            p = ((2 * m_ordering) - (2 * k) + 1) - q;
          } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
            p = ((4 * m_ordering) - (2 * k)) - q;
          } else if ((2 * m_ordering) - k - 1 < q) {
            p = n;
          }

          // Translate to (0,0)
          p_trans = p - 1;
          q_trans = q - 1;

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorR(i, p_trans, lda)];
            tmp_q = A.elements[iteratorR(i, q_trans, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);

          if (convergence_value > tolerance) {

            // Schur
            double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

            // Calculate a_{pp}, a_{qq}, a_{pq}
            for (size_t i = 0; i < m; ++i) {
              double value_p = A.elements[iteratorR(i, p, lda)];
              double value_q = A.elements[iteratorR(i, q, lda)];
              app += value_p * value_p;
              aqq += value_q * value_q;
            }

            if (std::abs(apq) > tolerance) {
              double tau = (aqq - app) / (2.0 * apq);
              double t = 0.0;

              if (tau >= 0) {
                t = 1.0 / (tau + sqrt(1 + (tau * tau)));
              } else {
                t = 1.0 / (tau - sqrt(1 + (tau * tau)));
              }

              c_schur = 1.0 / sqrt(1 + (t * t));
              s_schur = t * c_schur;

              double tmp_A_p, tmp_A_q;
              for (size_t i = 0; i < m; ++i) {
                tmp_A_p = A.elements[iteratorR(i, p_trans, lda)];
                tmp_A_q = A.elements[iteratorR(i, q_trans, lda)];
                tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
                tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
                A.elements[iteratorR(i, p_trans, lda)] = tmp_p;
                A.elements[iteratorR(i, q_trans, lda)] = tmp_q;
              }

              if (jobv == AllVec || jobv == SomeVec) {
                for (size_t i = 0; i < n; ++i) {
                  tmp_p =
                      c_schur * V.elements[iteratorR(i, p_trans, ldv)]
                          - s_schur * V.elements[iteratorR(i, q_trans, ldv)];
                  tmp_q =
                      s_schur * V.elements[iteratorR(i, p_trans, ldv)]
                          + c_schur * V.elements[iteratorR(i, q_trans, ldv)];
                  V.elements[iteratorR(i, p_trans, ldv)] = tmp_p;
                  V.elements[iteratorR(i, q_trans, ldv)] = tmp_q;
                }
              }
            }
          }

#ifdef DEBUG
          // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
        }
      }

      for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
        size_t p = 0;
        size_t p_trans = 0;
        size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
        for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
          if (q < (2 * m_ordering) - k + 1) {
            p = n;
          } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
            p = ((4 * m_ordering) - (2 * k)) - q;
          } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
            p = ((6 * m_ordering) - (2 * k) - 1) - q;
          }

          // Translate to (0,0)
          p_trans = p - 1;
          q_trans = q - 1;

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorR(i, p_trans, lda)];
            tmp_q = A.elements[iteratorR(i, q_trans, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);

          if (convergence_value > tolerance) {

            // Schur
            double c_schur = 1.0, s_schur = 0.0, aqq = 0.0, app = 0.0, apq = alpha;

            // Calculate a_{pp}, a_{qq}, a_{pq}
            for (size_t i = 0; i < m; ++i) {
              double value_p = A.elements[iteratorR(i, p, lda)];
              double value_q = A.elements[iteratorR(i, q, lda)];
              app += value_p * value_p;
              aqq += value_q * value_q;
            }

            if (std::abs(apq) > tolerance) {
              double tau = (aqq - app) / (2.0 * apq);
              double t = 0.0;

              if (tau >= 0) {
                t = 1.0 / (tau + sqrt(1 + (tau * tau)));
              } else {
                t = 1.0 / (tau - sqrt(1 + (tau * tau)));
              }

              c_schur = 1.0 / sqrt(1 + (t * t));
              s_schur = t * c_schur;

              double tmp_A_p, tmp_A_q;
              for (size_t i = 0; i < m; ++i) {
                tmp_A_p = A.elements[iteratorR(i, p_trans, lda)];
                tmp_A_q = A.elements[iteratorR(i, q_trans, lda)];
                tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
                tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
                A.elements[iteratorR(i, p_trans, lda)] = tmp_p;
                A.elements[iteratorR(i, q_trans, lda)] = tmp_q;
              }

              if (jobv == AllVec || jobv == SomeVec) {
                for (size_t i = 0; i < n; ++i) {
                  tmp_p =
                      c_schur * V.elements[iteratorR(i, p_trans, ldv)]
                          - s_schur * V.elements[iteratorR(i, q_trans, ldv)];
                  tmp_q =
                      s_schur * V.elements[iteratorR(i, p_trans, ldv)]
                          + c_schur * V.elements[iteratorR(i, q_trans, ldv)];
                  V.elements[iteratorR(i, p_trans, ldv)] = tmp_p;
                  V.elements[iteratorR(i, q_trans, ldv)] = tmp_q;
                }
              }
            }
          }

#ifdef DEBUG
          // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
        }
      }

#ifdef DEBUG
      // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
    } while (++reps < maxIterations);

    std::cout << "How many repetitions?: " << reps << "\n";

    // Compute \Sigma
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        s.elements[k] += A.elements[iteratorR(i, k, lda)] * A.elements[iteratorR(i, k, lda)];
      }
      s.elements[k] = sqrt(s.elements[k]);
    }

    //Compute U
    if (jobu == AllVec) {
#pragma omp parallel for
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
          U.elements[iteratorR(j, i, ldu)] = A.elements[iteratorR(j, i, ldu)] / s.elements[i];
        }
      }
    } else if (jobu == SomeVec) {
#pragma omp parallel for
      for (size_t k = 0; k < std::min(m, n); ++k) {
        for (size_t i = 0; i < m; ++i) {
          U.elements[iteratorR(i, k, ldu)] = A.elements[iteratorR(i, k, ldu)] / s.elements[k];
        }
      }
    }
  }
}
}
