#include <iostream>
#include <cstring>
#include <random>
#include <chrono>
#include <locale>  

class Matrix
{
private:
    int ctrok;
    int stolb;
    double* m;
    double** arr;

    void mat() {
        m = new double[ctrok * stolb];
        arr = new double* [ctrok];
        for (int i = 0; i < ctrok; i++) {
            arr[i] = m + i * stolb;
        }
    }

public:
    
    Matrix(int cc, int ss) : ctrok(cc), stolb(ss) {
        mat();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> randomDOUBLE(-10.0, 10.0);
        for (int i = 0; i < ctrok; i++) {
            for (int j = 0; j < stolb; j++) {
                arr[i][j] = randomDOUBLE(gen);
            }
        }
    }

    
    Matrix(int cc, int ss, bool) : ctrok(cc), stolb(ss) {
        mat();
        std::memset(m, 0, ctrok * stolb * sizeof(double));
    }

    
    Matrix(const Matrix& other) : ctrok(other.ctrok), stolb(other.stolb) {
        mat();
        std::memcpy(m, other.m, ctrok * stolb * sizeof(double));
    }

    
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] m;
            delete[] arr;
            ctrok = other.ctrok;
            stolb = other.stolb;
            mat();
            std::memcpy(m, other.m, ctrok * stolb * sizeof(double));
        }
        return *this;
    }

    
    Matrix operator*(const Matrix& matrix) const {
        if (stolb != matrix.ctrok)
            throw std::invalid_argument("Ошибка: размеры матриц не совпадают!");

        Matrix result(ctrok, matrix.stolb, true);

        for (int i = 0; i < ctrok; i++) {
            for (int j = 0; j < matrix.stolb; j++) {
                double sum = 0.0;
                for (int k = 0; k < stolb; k++) {
                    sum += arr[i][k] * matrix.arr[k][j];
                }
                result.arr[i][j] = sum;
            }
        }
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

    const int n = 2048;  

    std::cout << "\n============================================================" << std::endl;
    std::cout << "   ВАРИАНТ 1: КЛАССИЧЕСКИЙ АЛГОРИТМ" << std::endl;
    std::cout << "   Умножение матриц " << n << "x" << n << std::endl;
    std::cout << "============================================================" << std::endl;

    std::cout << "\nГенерация случайных матриц... " << std::flush;
    Matrix A(n, n);
    Matrix B(n, n);
    std::cout << "ГОТОВО" << std::endl;

    double c = 2.0 * n * n * n;
    std::cout << "Вычислительная сложность: " << (c / 1e9) << " млрд операций" << std::endl;

    std::cout << "\nУмножение матриц... " << std::flush;
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
    std::cout << "\nЗыков Алексей Александрович  020303-АИСа-о25" << std::endl;

    std::cerr << "\nНажмите Enter...";
    std::cin.get();
    return 0;
}