#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>


using vector_row = std::vector<double>;
using matrix = std::vector<vector_row>;

double integral_by_line_horizontal(double x, double y, double h, double epsilon)
{
    double x_inter = 3 * (1 - y / 4);
    double l = std::min(std::max(x_inter - x + h / 2, 0.0), h);
    return (l + (h - l) / epsilon) / h;
}

double integral_by_line_vertical(double x, double y, double h, double epsilon)
{
    double y_inter = 4 * (1 - x / 3);
    double l = std::min(std::max(y_inter - y + h / 2, 0.0), h);
    return (l + (h - l) / epsilon) / h;
}

double integral_by_triangle(double x, double y)
{
    double x_inter = 3 * (1 - y / 4);
    double y_inter = 4 * (1 - x / 3);
    return double(x / 3 + y / 4 <= 1) * std::abs(x - x_inter) * std::abs(y - y_inter) / 2;
}

double integral_by_square(double x, double y, double hx, double hy)
{
    double S = integral_by_triangle(x + hx / 2, y + hy / 2);
    S += integral_by_triangle(x - hx / 2, y - hy / 2);
    S -= integral_by_triangle(x - hx / 2, y + hy / 2);
    S -= integral_by_triangle(x + hx / 2, y - hy / 2);
    return S / (hx * hy);
}

void construct_sparse_matrix(
    matrix& mat, 
    matrix& F, 
    int M, int N, 
    double hx, double hy, 
    double epsilon)
{
    matrix A(M + 1, vector_row(N + 1));
    matrix B(M + 1, vector_row(N + 1));

    #pragma omp parallel for collapse(2)
    for(int i = 1; i <= M; i++) {
        for(int j = 1; j <= N; j++) {
            A[i][j] = integral_by_line_vertical(hx * i - hx / 2, hy * j - hy / 2, hy, epsilon);
            B[i][j] = integral_by_line_horizontal(hx * i - hx / 2, hy * j - hy / 2, hx, epsilon);
            F[i][j] = integral_by_square(hx * i - hx / 2, hy * j - hy / 2, hx, hy);
        }
    }

    hx *= hx;
    hy *= hy;

    #pragma omp parallel for collapse(2)
    for(int i = 1; i < M; ++i) {
        for(int j = 1; j < N; ++j) {
            mat[i * (N + 1) + j][0] = -A[i][j] / hx;
            mat[i * (N + 1) + j][1] = -B[i][j] / hy;
            mat[i * (N + 1) + j][2] = (A[i][j] + A[i + 1][j]) / hx + (B[i][j] + B[i][j + 1]) / hy;
            mat[i * (N + 1) + j][3] = -B[i][j + 1] / hy;
            mat[i * (N + 1) + j][4] = -A[i + 1][j] / hx;
        }
    }
}

void matvec(
    matrix &res, 
    const matrix &mat, const matrix &w,
    int M, int N
) {
    #pragma omp parallel for collapse(2)
    for(int i = 1; i < M; ++i) {
        for(int j = 1; j < N; ++j) {
            res[i][j] = mat[i * (N + 1) + j][0] * w[i - 1][j] + 
                        mat[i * (N + 1) + j][1] * w[i][j - 1] + 
                        mat[i * (N + 1) + j][2] * w[i][j] + 
                        mat[i * (N + 1) + j][3] * w[i][j + 1] + 
                        mat[i * (N + 1) + j][4] * w[i + 1][j];
        }
    }
};

double dot_prod(
    const matrix &u, const matrix &v, 
    double hx, double hy, 
    int M, int N) 
{
    double sum = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for(int i = 1; i < M; ++i) {
        for(int j = 1; j < N; ++j) {
            sum += u[i][j] * v[i][j];
        }
    }
    return sum * hx * hy;
}

void CG(
    matrix &res,
    const matrix &mat, const matrix &F, 
    int M, int N, 
    double max_error, double max_iter,
    double hx, double hy,
    std::ofstream& outfile) 
{
    matrix work(M + 1, vector_row(N + 1));
    matrix rk(M + 1, vector_row(N + 1));
    matrix zk(M + 1, vector_row(N + 1));
    matrix pk(M + 1, vector_row(N + 1));

    matvec(work, mat, res, M, N);

    #pragma omp parallel for collapse(2)
    for(int i = 1; i < M; ++i) {
        for(int j = 1; j < N; ++j) {
            rk[i][j] = F[i][j] - work[i][j];
            zk[i][j] = rk[i][j] / (mat[i * (N + 1) + j][2] + 1e-20);
            pk[i][j] = zk[i][j];
        }
    }

    double znam = dot_prod(zk, rk, hx, hy, M, N);
    double alpha, max_val, chisl, beta;

    for(int iter = 0; iter < max_iter; ++iter) {
        matvec(work, mat, pk, M, N);
        alpha = znam / dot_prod(pk, work, hx, hy, M, N);

        max_val = std::abs(alpha * pk[1][1]);
        #pragma omp parallel for collapse(2)
        for(int i = 1; i < M; ++i) {
            for(int j = 1; j < N; ++j) {
                res[i][j] += alpha * pk[i][j];
                rk[i][j] -= alpha * work[i][j];
                zk[i][j] = rk[i][j] / (mat[i * (N + 1) + j][2] + 1e-20);
                max_val = std::max(max_val, std::abs(alpha * pk[i][j]));
            }
        }

        if(max_val < max_error){
            outfile << "End iter = " << iter + 1 << ", Max_val = " << max_val << std::endl; 
            break;
        }

        chisl = dot_prod(zk, rk, hx, hy, M, N);
        beta = chisl / znam;
        znam = chisl;

        #pragma omp parallel for collapse(2)
        for(int i = 1; i < M; ++i) {
            for(int j = 1; j <= N; ++j) {
                pk[i][j] = zk[i][j] + beta * pk[i][j];
            }
        }
    }
};

int main(int argc, char* argv[])
{
    std::ofstream outfile("res.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open res.txt for writing!" << std::endl;
        return 1;
    }

    std::streambuf* coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(outfile.rdbuf());

    if (argc < 4) {
        std::cout << "Arguments not enough!" << std::endl;
        return 1;
    }
    
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    std::cout << "M: " << M << std::endl << "N: " << N << std::endl;

    double A1 = 0.0;
    double B1 = 3.0;
    double A2 = 0.0;
    double B2 = 4.0;

    double hx = (B1 - A1) / M;
    double hy = (B2 - A2) / N;

    double epsilon = std::max(hx, hy) * std::max(hx, hy);
    double max_error = 1e-11;
    int max_iter = 5000;

    matrix mat((M + 1) * (N + 1), vector_row(5));
    matrix F(M + 1, vector_row(N + 1));
    matrix res(M + 1, vector_row(N + 1));

    auto start_time_c = std::chrono::steady_clock::now(); 

    construct_sparse_matrix(mat, F, M, N, hx, hy, epsilon);
    CG(res, mat, F, M, N, max_error, max_iter, hx, hy, outfile);

    auto end_time_c = std::chrono::steady_clock::now(); 

    std::chrono::duration<double> elapsed = end_time_c - start_time_c;
    std::cout << "Total time: " << elapsed.count() << " seconds." << std::endl;

    double err = 0.0;
    double max_err = 0.0;
    double norm = 0.0;
    matrix work = std::vector<vector_row>(M + 1, vector_row(N + 1));
    matvec(work, mat, res, M, N);
    for(int i = 1; i < M; ++i) {
        for(int j = 1; j < N; ++j) {
            max_err = std::abs(F[i][j] - work[i][j]);
            err += max_err * max_err;
            norm += F[i][j] * F[i][j];
        }
    }
    std::cout << "Frobenius error: " << std::sqrt(err) / std::sqrt(norm) << std::endl;
    std::cout << "Cheb error: " << max_err << std::endl;

    std::cout.rdbuf(coutbuf);
    outfile.close();

    return 0;
}
