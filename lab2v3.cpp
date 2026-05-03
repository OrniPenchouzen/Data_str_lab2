#define NOMINMAX

#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <cstring>
#include <locale>
#include <windows.h>

struct Matrix {
    int n;
    float* real;
    float* imag;

    Matrix(int n_, bool zero = false) : n(n_) {
        real = (float*)_aligned_malloc(n * n * sizeof(float), 32);
        imag = (float*)_aligned_malloc(n * n * sizeof(float), 32);

        if (zero) {
            memset(real, 0, n * n * sizeof(float));
            memset(imag, 0, n * n * sizeof(float));
        }
        else {
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int i = 0; i < n * n; i++) {
                real[i] = dist(gen);
                imag[i] = dist(gen);
            }
        }
    }

    inline int idx(int i, int j) const { return i * n + j; }

    ~Matrix() {
        _aligned_free(real);
        _aligned_free(imag);
    }
};

// транспонирование
Matrix transpose(const Matrix& B) {
    Matrix BT(B.n, true);

#pragma omp parallel for
    for (int i = 0; i < B.n; i++)
        for (int j = 0; j < B.n; j++) {
            int s = B.idx(i, j);
            int d = BT.idx(j, i);
            BT.real[d] = B.real[s];
            BT.imag[d] = B.imag[s];
        }

    return BT;
}

// AVX2 + OpenMP умножение
Matrix multiply(const Matrix& A, const Matrix& BT) {
    int n = A.n;
    Matrix C(n, true);
    const int BLOCK = 32;

#pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < n; i0 += BLOCK)
        for (int j0 = 0; j0 < n; j0 += BLOCK)
        {
            for (int k0 = 0; k0 < n; k0 += BLOCK) {

                int i_max = std::min(i0 + BLOCK, n);
                int j_max = std::min(j0 + BLOCK, n);
                int k_max = std::min(k0 + BLOCK, n);

                for (int i = i0; i < i_max; i++) {
                    for (int j = j0; j < j_max; j += 4) {

                        __m256 sum_r = _mm256_setzero_ps();
                        __m256 sum_i = _mm256_setzero_ps();

                        for (int k = k0; k < k_max; k++) {

                            float ar = A.real[i * n + k];
                            float ai = A.imag[i * n + k];

                            // FIX: loadu вместо load
                            __m256 brbi = _mm256_loadu_ps(&BT.real[j * n + k]);

                            __m256 br = _mm256_permute_ps(brbi, 0b11011000);
                            __m256 bi = _mm256_permute_ps(brbi, 0b11111101);

                            __m256 ar_v = _mm256_set1_ps(ar);
                            __m256 ai_v = _mm256_set1_ps(ai);

                            sum_r = _mm256_add_ps(sum_r,
                                _mm256_sub_ps(_mm256_mul_ps(ar_v, br),
                                    _mm256_mul_ps(ai_v, bi)));

                            sum_i = _mm256_add_ps(sum_i,
                                _mm256_add_ps(_mm256_mul_ps(ar_v, bi),
                                    _mm256_mul_ps(ai_v, br)));
                        }

                        _mm256_storeu_ps(&C.real[i * n + j], sum_r);
                        _mm256_storeu_ps(&C.imag[i * n + j], sum_i);
                    }
                }
            }
        }

    return C;
}

int main() {
    const int n = 2048;
    const double TARGET = 30000.0;

    // локализация
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::setlocale(LC_ALL, "Russian");

    omp_set_dynamic(0);
    omp_set_num_threads(12);

    std::cout << "\n==========================================================\n";
    std::cout << "   AVX2 + OpenMP ОПТИМИЗАЦИЯ\n";
    std::cout << "   Умножение матриц " << n << "x" << n << "\n";
    std::cout << "==========================================================\n";

#pragma omp parallel
    {
#pragma omp single
        std::cout << "Потоки: " << omp_get_num_threads() << "\n";
    }

    std::cout << "\nГенерация матриц... ";
    Matrix A(n);
    Matrix B(n);
    std::cout << "ГОТОВО\n";

    std::cout << "Транспонирование... ";
    Matrix BT = transpose(B);
    std::cout << "ГОТОВО\n";

    double flops = 2.0 * n * n * n;

    std::cout << "Умножение... ";
    auto start = std::chrono::high_resolution_clock::now();

    Matrix C = multiply(A, BT);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "ГОТОВО\n";

    double time = std::chrono::duration<double>(end - start).count();
    double mflops = flops / time / 1e6;
    double percent = (mflops / TARGET) * 100.0;

    std::cout << "\n==========================================================\n";
    std::cout << "РЕЗУЛЬТАТЫ:\n";
    std::cout << "  Время: " << time << " сек\n";
    std::cout << "  Производительность: " << mflops << " MFLOPS\n";
    std::cout << "Зыков Алексей Александрович 020303-АИСа-о25";



    return 0;
}