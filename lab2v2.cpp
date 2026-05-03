#include <iostream>
#include <cstring>
#include <random>
#include <chrono>
#include <locale>
#include <complex>

// Подключаем CBLAS из OpenBLAS
#include <cblas.h>

class Matrix
{
private:
    int ctrok;
    int stolb;
    std::complex<float>* m;  
    std::complex<float>** arr;

    void mat() {
        m = new std::complex<float>[ctrok * stolb];
        arr = new std::complex<float>*[ctrok];
        for (int i = 0; i < ctrok; i++) {
            arr[i] = m + i * stolb;
        }
    }

public:
    // Конструктор с случайным заполнением
    Matrix(int cc, int ss) : ctrok(cc), stolb(ss) {
        mat();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> randomFLOAT(-10.0f, 10.0f);
        for (int i = 0; i < ctrok; i++) {
            for (int j = 0; j < stolb; j++) {
                arr[i][j] = std::complex<float>(randomFLOAT(gen), randomFLOAT(gen));
            }
        }
    }

    // Конструктор для создания пустой матрицы
    Matrix(int cc, int ss, bool) : ctrok(cc), stolb(ss) {
        mat();
        std::memset(m, 0, ctrok * stolb * sizeof(std::complex<float>));
    }

    // Конструктор копирования
    Matrix(const Matrix& other) : ctrok(other.ctrok), stolb(other.stolb) {
        mat();
        std::memcpy(m, other.m, ctrok * stolb * sizeof(std::complex<float>));
    }

    // Оператор присваивания
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] m;
            delete[] arr;
            ctrok = other.ctrok;
            stolb = other.stolb;
            mat();
            std::memcpy(m, other.m, ctrok * stolb * sizeof(std::complex<float>));
        }
        return *this;
    }

    // УМНОЖЕНИЕ ЧЕРЕЗ BLAS (cblas_cgemm) - для комплексных чисел float
    Matrix operator*(const Matrix& matrix) const {
        if (stolb != matrix.ctrok)
            throw std::invalid_argument("Ошибка: размеры матриц не совпадают!");

        Matrix result(ctrok, matrix.stolb, true);

        std::complex<float> alpha = 1.0f;
        std::complex<float> beta = 0.0f;

        // Вызов BLAS для комплексных чисел float
        cblas_cgemm(
            CblasRowMajor,      // расположение по строкам
            CblasNoTrans,       // не транспонируем A
            CblasNoTrans,       // не транспонируем B
            ctrok,              // количество строк A и C (m)
            matrix.stolb,       // количество столбцов B и C (n)
            stolb,              // количество столбцов A / строк B (k)
            &alpha,             // α
            reinterpret_cast<const float*>(m),      // A
            stolb,              // ведущая размерность A (lda)
            reinterpret_cast<const float*>(matrix.m), // B
            matrix.stolb,       // ведущая размерность B (ldb)
            &beta,              // β
            reinterpret_cast<float*>(result.m),     // C (результат)
            matrix.stolb        // ведущая размерность C (ldc)
        );

        return result;
    }

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

    ~Matrix() {
        delete[] m;
        delete[] arr;
    }
};

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
    for (int i = 0; i < matrix.ctrok; i++) {
        for (int j = 0; j < matrix.stolb; j++) {
            stream << matrix.arr[i][j] << " ";
        }
        stream << std::endl;
    }
    return stream;
}

int main() {
    setlocale(LC_ALL, "Russian");

    const int n = 2048;  // Размер матрицы

    std::cout << "\n============================================================" << std::endl;
    std::cout << "   ВАРИАНТ 2: BLAS (cblas_cgemm)" << std::endl;
    std::cout << "   Умножение комплексных матриц " << n << "x" << n << std::endl;
    std::cout << "   Тип данных: std::complex<float> (одинарная точность)" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::cout << "\nГенерация случайных матриц... " << std::flush;
    Matrix A(n, n);
    Matrix B(n, n);
    std::cout << "ГОТОВО" << std::endl;

    double c = 2.0 * n * n * n;
    std::cout << "Вычислительная сложность: " << (c / 1e9) << " млрд операций" << std::endl;

    std::cout << "\nУмножение матриц (BLAS)... " << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    double t = std::chrono::duration<double>(end - start).count();
    double p = c / t * 1e-6;

    std::cout << "ГОТОВО" << std::endl;
    std::cout << "\n============================================================" << std::endl;
    std::cout << "РЕЗУЛЬТАТЫ:" << std::endl;
    std::cout << "  Время: " << t << " секунд" << std::endl;
    std::cout << "  Производительность: " << p << " MFLOPS" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "\nЗыков Алексей Александрович 020303-АИСа-о25" << std::endl;

    std::cerr << "\nНажмите Enter...";
    std::cin.get();
    return 0;
}
