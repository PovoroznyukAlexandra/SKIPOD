#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

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

void construct_sparse_matrix_local(
    matrix& mat_local, 
    matrix& F_local, 
    int M, int N, 
    double hx, double hy, 
    double epsilon,
    int start_row, int end_row)
{
    int local_rows = end_row - start_row + 1;
    
    for(int i = start_row; i <= end_row; i++) {
        for(int j = 1; j <= N; j++) {
            int local_i = i - start_row;
            double A_ij = integral_by_line_vertical(hx * i - hx / 2, hy * j - hy / 2, hy, epsilon);
            double B_ij = integral_by_line_horizontal(hx * i - hx / 2, hy * j - hy / 2, hx, epsilon);
            F_local[local_i][j] = integral_by_square(hx * i - hx / 2, hy * j - hy / 2, hx, hy);
            
            if (i < M && j < N) {
                double A_ip1j = integral_by_line_vertical(hx * (i + 1) - hx / 2, hy * j - hy / 2, hy, epsilon);
                double B_ijp1 = integral_by_line_horizontal(hx * i - hx / 2, hy * (j + 1) - hy / 2, hx, epsilon);
                
                int idx = local_i * (N + 1) + j;
                mat_local[idx][0] = -A_ij / (hx * hx);
                mat_local[idx][1] = -B_ij / (hy * hy);
                mat_local[idx][2] = (A_ij + A_ip1j) / (hx * hx) + (B_ij + B_ijp1) / (hy * hy);
                mat_local[idx][3] = -B_ijp1 / (hy * hy);
                mat_local[idx][4] = -A_ip1j / (hx * hx);
            }
        }
    }
}

void matvec_local(
    matrix &res_local,
    const matrix &mat_local,
    const matrix &w_local,
    int M, int N,
    int start_row, int end_row,
    MPI_Comm comm)
{
    int rank, world;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);

    int local_rows = end_row - start_row + 1;
    const int stride = N + 1;

    matrix wbuf(local_rows + 2, vector_row(stride));
    
    for (int i = 0; i < local_rows; ++i) {
        wbuf[i + 1] = w_local[i];
    }

    int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int dn_rank = (rank < world - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Request requests[4];
    int req_count = 0;

    if (up_rank != MPI_PROC_NULL) {
        MPI_Isend(wbuf[1].data(), stride, MPI_DOUBLE, up_rank, 0, comm, &requests[req_count++]);
    }
    if (dn_rank != MPI_PROC_NULL) {
        MPI_Irecv(wbuf[local_rows + 1].data(), stride, MPI_DOUBLE, dn_rank, 0, comm, &requests[req_count++]);
    }

    if (dn_rank != MPI_PROC_NULL) {
        MPI_Isend(wbuf[local_rows].data(), stride, MPI_DOUBLE, dn_rank, 1, comm, &requests[req_count++]);
    }
    if (up_rank != MPI_PROC_NULL) {
        MPI_Irecv(wbuf[0].data(), stride, MPI_DOUBLE, up_rank, 1, comm, &requests[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    for (int local_i = 0; local_i < local_rows; ++local_i) {
        int global_i = start_row + local_i;
        
        if (global_i >= M) continue;
        
        for (int j = 1; j < N; ++j) {
            int idx = local_i * stride + j;
            
            res_local[local_i][j] =
                mat_local[idx][0] * wbuf[local_i][j] + 
                mat_local[idx][1] * wbuf[local_i + 1][j - 1] + 
                mat_local[idx][2] * wbuf[local_i + 1][j] + 
                mat_local[idx][3] * wbuf[local_i + 1][j + 1] +
                mat_local[idx][4] * wbuf[local_i + 2][j]; 
        }
    }
}

double dot_prod_local(
    const matrix &u_local, const matrix &v_local, 
    double hx, double hy, 
    int start_row, int end_row, int M, int N,
    MPI_Comm comm) 
{
    double local_sum = 0.0;
    int local_rows = end_row - start_row + 1;

    for(int local_i = 0; local_i < local_rows; ++local_i) {
        int global_i = start_row + local_i;
        if (global_i >= M) continue;
        
        for(int j = 1; j < N; ++j) {
            local_sum += u_local[local_i][j] * v_local[local_i][j];
        }
    }
    
    local_sum *= hx * hy;
    
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

void CG_distributed(
    matrix &res_local,
    const matrix &mat_local, const matrix &F_local, 
    int M, int N, 
    double max_error, double max_iter,
    double hx, double hy,
    int start_row, int end_row,
    std::ofstream& outfile,
    MPI_Comm comm) 
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int local_rows = end_row - start_row + 1;
    
    matrix work_local(local_rows, vector_row(N + 1, 0.0));
    matrix rk_local(local_rows, vector_row(N + 1, 0.0));
    matrix zk_local(local_rows, vector_row(N + 1, 0.0));
    matrix pk_local(local_rows, vector_row(N + 1, 0.0));

    matvec_local(work_local, mat_local, res_local, M, N, start_row, end_row, comm);
    
    for(int local_i = 0; local_i < local_rows; ++local_i) {
        int global_i = start_row + local_i;
        if (global_i >= M) continue;
        
        for(int j = 1; j < N; ++j) {
            rk_local[local_i][j] = F_local[local_i][j] - work_local[local_i][j];
            
            int idx = local_i * (N + 1) + j;
            double diag = mat_local[idx][2];
            zk_local[local_i][j] = rk_local[local_i][j] / (diag + 1e-20);
            
            pk_local[local_i][j] = zk_local[local_i][j];
        }
    }

    double znam = dot_prod_local(zk_local, rk_local, hx, hy, start_row, end_row, M, N, comm);
    double alpha, max_val, chisl, beta;

    for(int iter = 0; iter < max_iter; ++iter) {

        matvec_local(work_local, mat_local, pk_local, M, N, start_row, end_row, comm);
        
        alpha = znam / dot_prod_local(pk_local, work_local, hx, hy, start_row, end_row, M, N, comm);

        max_val = 0.0;
        for(int local_i = 0; local_i < local_rows; ++local_i) {
            int global_i = start_row + local_i;
            if (global_i >= M) continue;
            
            for(int j = 1; j < N; ++j) {
                res_local[local_i][j] += alpha * pk_local[local_i][j];
                rk_local[local_i][j] -= alpha * work_local[local_i][j];
                
                int idx = local_i * (N + 1) + j;
                double diag = mat_local[idx][2];
                zk_local[local_i][j] = rk_local[local_i][j] / (diag + 1e-20);
                
                max_val = std::max(max_val, std::abs(alpha * pk_local[local_i][j]));
            }
        }

        double global_max_val;
        MPI_Allreduce(&max_val, &global_max_val, 1, MPI_DOUBLE, MPI_MAX, comm);

        if(rank == 0 && iter % 100 == 0) {
            outfile << "Iter " << iter << ", Max update = " << global_max_val << std::endl;
        }

        if(global_max_val < max_error){
            if(rank == 0) {
                outfile << "Converged at iteration " << iter + 1 << ", Max_val = " << global_max_val << std::endl; 
            }
            break;
        }

        chisl = dot_prod_local(zk_local, rk_local, hx, hy, start_row, end_row, M, N, comm);
        beta = chisl / znam;
        znam = chisl;

        for(int local_i = 0; local_i < local_rows; ++local_i) {
            int global_i = start_row + local_i;
            if (global_i >= M) continue;
            
            for(int j = 1; j < N; ++j) {
                pk_local[local_i][j] = zk_local[local_i][j] + beta * pk_local[local_i][j];
            }
        }
    }
}

void gather_solution(
    matrix &global_res,
    const matrix &local_res,
    int M, int N,
    int start_row, int end_row,
    MPI_Comm comm)
{
    int rank, world;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);

    int local_rows = end_row - start_row + 1;
    const int stride = N + 1;

    std::vector<double> send_buffer(local_rows * stride);
    for (int i = 0; i < local_rows; ++i) {
        std::copy(local_res[i].begin(), local_res[i].end(), 
                  send_buffer.begin() + i * stride);
    }

    std::vector<int> recv_counts(world);
    std::vector<int> displs(world);

    std::vector<int> all_start_rows(world);
    std::vector<int> all_local_rows(world);
    
    MPI_Allgather(&start_row, 1, MPI_INT, all_start_rows.data(), 1, MPI_INT, comm);
    MPI_Allgather(&local_rows, 1, MPI_INT, all_local_rows.data(), 1, MPI_INT, comm);

    int total_size = 0;
    for (int p = 0; p < world; ++p) {
        recv_counts[p] = all_local_rows[p] * stride;
        displs[p] = total_size;
        total_size += recv_counts[p];
    }

    std::vector<double> recv_buffer;
    if (rank == 0) {
        recv_buffer.resize(total_size);
    }

    MPI_Gatherv(
        send_buffer.data(), send_buffer.size(), MPI_DOUBLE,
        recv_buffer.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
        0, comm
    );

    if (rank == 0) {
        for (int p = 0; p < world; ++p) {
            int p_start = all_start_rows[p];
            int p_rows = all_local_rows[p];
            
            for (int i = 0; i < p_rows; ++i) {
                int global_i = p_start + i;
                if (global_i <= M) {
                    std::copy(recv_buffer.begin() + displs[p] + i * stride,
                             recv_buffer.begin() + displs[p] + (i + 1) * stride,
                             global_res[global_i].begin());
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    std::ofstream outfile;
    if (rank == 0) {
        outfile.open("fres.txt");
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open res.txt for writing!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (argc < 3) {
        if (rank == 0) {
            std::cout << "Arguments not enough! Usage: " << argv[0] << " M N" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    
    if (rank == 0) {
        outfile << "M: " << M << std::endl << "N: " << N << std::endl;
        outfile << "MPI processes: " << world << std::endl;
    }

    double A1 = 0.0;
    double B1 = 3.0;
    double A2 = 0.0;
    double B2 = 4.0;

    double hx = (B1 - A1) / M;
    double hy = (B2 - A2) / N;

    double epsilon = std::max(hx, hy) * std::max(hx, hy);
    double max_error = 1e-11;
    int max_iter = 5000;

    int rows_per_proc = M / world;
    int remainder = M % world;
    
    int start_row = rank * rows_per_proc + std::min(rank, remainder);
    int end_row = start_row + rows_per_proc - 1;
    if (rank < remainder) {
        end_row += 1;
    }
    
    if (rank == world - 1) {
        end_row = M;
    }
    
    int local_rows = end_row - start_row + 1;

    if (rank == 0) {
        outfile << "Domain decomposition:" << std::endl;
        for (int p = 0; p < world; ++p) {
            int p_start = p * rows_per_proc + std::min(p, remainder);
            int p_end = p_start + rows_per_proc - 1;
            if (p < remainder) p_end += 1;
            if (p == world - 1) p_end = M;
            outfile << "  Process " << p << ": rows " << p_start << " to " << p_end 
                   << " (" << (p_end - p_start + 1) << " rows)" << std::endl;
        }
    }

    matrix mat_local(local_rows * (N + 1), vector_row(5, 0.0));
    matrix F_local(local_rows, vector_row(N + 1, 0.0));
    matrix res_local(local_rows, vector_row(N + 1, 0.0));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time_c = std::chrono::steady_clock::now(); 

    construct_sparse_matrix_local(mat_local, F_local, M, N, hx, hy, epsilon, start_row, end_row);
    
    CG_distributed(res_local, mat_local, F_local, M, N, max_error, max_iter, hx, hy, 
                  start_row, end_row, outfile, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time_c = std::chrono::steady_clock::now(); 
    std::chrono::duration<double> elapsed = end_time_c - start_time_c;

    if (rank == 0) {
        outfile << "Total time: " << elapsed.count() << " seconds." << std::endl;
    }

    matrix global_res;
    if (rank == 0) {
        global_res.resize(M + 1, vector_row(N + 1, 0.0));
    }
    
    gather_solution(global_res, res_local, M, N, start_row, end_row, MPI_COMM_WORLD);

    if (rank == 0) {
        matrix mat_full((M + 1) * (N + 1), vector_row(5, 0.0));
        matrix F_full(M + 1, vector_row(N + 1, 0.0));
        
        for (int i = 0; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                double A_ij = integral_by_line_vertical(hx * i - hx / 2, hy * j - hy / 2, hy, epsilon);
                double B_ij = integral_by_line_horizontal(hx * i - hx / 2, hy * j - hy / 2, hx, epsilon);
                F_full[i][j] = integral_by_square(hx * i - hx / 2, hy * j - hy / 2, hx, hy);
                
                if (i < M && j < N) {
                    double A_ip1j = integral_by_line_vertical(hx * (i + 1) - hx / 2, hy * j - hy / 2, hy, epsilon);
                    double B_ijp1 = integral_by_line_horizontal(hx * i - hx / 2, hy * (j + 1) - hy / 2, hx, epsilon);
                    
                    int idx = i * (N + 1) + j;
                    mat_full[idx][0] = -A_ij / (hx * hx);
                    mat_full[idx][1] = -B_ij / (hy * hy);
                    mat_full[idx][2] = (A_ij + A_ip1j) / (hx * hx) + (B_ij + B_ijp1) / (hy * hy);
                    mat_full[idx][3] = -B_ijp1 / (hy * hy);
                    mat_full[idx][4] = -A_ip1j / (hx * hx);
                }
            }
        }
        
        matrix work_full(M + 1, vector_row(N + 1, 0.0));
        
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                int idx = i * (N + 1) + j;
                work_full[i][j] =
                    mat_full[idx][0] * global_res[i-1][j] +
                    mat_full[idx][1] * global_res[i][j-1] +
                    mat_full[idx][2] * global_res[i][j] +
                    mat_full[idx][3] * global_res[i][j+1] +
                    mat_full[idx][4] * global_res[i+1][j];
            }
        }
        
        double err = 0.0;
        double max_err = 0.0;
        double norm = 0.0;
        for(int i = 1; i < M; ++i) {
            for(int j = 1; j < N; ++j) {
                double current_err = std::abs(F_full[i][j] - work_full[i][j]);
                max_err = std::max(max_err, current_err);
                err += current_err * current_err;
                norm += F_full[i][j] * F_full[i][j];
            }
        }
        outfile << "Frobenius error: " << std::sqrt(err) / std::sqrt(norm) << std::endl;
        outfile << "Max error: " << max_err << std::endl;
    }

    if (rank == 0) {
        outfile.close();
    }

    MPI_Finalize();
    return 0;
}