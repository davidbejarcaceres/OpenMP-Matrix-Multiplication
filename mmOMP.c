// To compile for debugging:
// gcc-9 mmopenmp.c -o mmopenmp -fopenmp -O3 -g

// To Compile for the best performance:
// gcc-9 mmopenmp.c -o mmopenmp -fopenmp -O3

// To run
// ./mmopenmp 500

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[])
{

    int N; //size of columns and rows (matrices)
    int tid, nthreads, i, j, k;
    double **a, **b, **c;
    double *a_block, *b_block, *c_block;
    double **res;
    double *res_block;

    if (argc < 2)
    {
        printf("Usage: mm matrix_size\n");
        exit(-1);
    }
    if (argc == 3)
    {
        nthreads = atoi(argv[2]);
    }
    else
    {
        nthreads = 4;
    }

    // clock_t t;
    // t = clock();

    // To measure time with OpenMP, uses only real processing time in seconds
    double start, end;
    start = omp_get_wtime();

    N = atoi(argv[1]);

    a = (double **)malloc(N * sizeof(double *)); /* matrix a to be multiplied */
    b = (double **)malloc(N * sizeof(double *)); /* matrix b to be multiplied */
    c = (double **)malloc(N * sizeof(double *)); /* result matrix c */

    a_block = (double *)malloc(N * N * sizeof(double)); /* Storage for matrices */
    b_block = (double *)malloc(N * N * sizeof(double));
    c_block = (double *)malloc(N * N * sizeof(double));

    /* Result matrix for the sequential algorithm */
    res = (double **)malloc(N * sizeof(double *));
    res_block = (double *)malloc(N * N * sizeof(double));

    for (i = 0; i < N; i++) /* Initialize pointers to a */
        a[i] = a_block + i * N;

    for (i = 0; i < N; i++) /* Initialize pointers to b */
        b[i] = b_block + i * N;

    for (i = 0; i < N; i++) /* Initialize pointers to c */
        c[i] = c_block + i * N;

    for (i = 0; i < N; i++) /* Initialize pointers to res */
        res[i] = res_block + i * N;

#pragma omp for nowait
    for (i = 0; i < N; i++) /* last matrix has been initialized */
        for (j = 0; j < N; j++)
            a[i][j] = (i + j) * ((double)rand());

#pragma omp for nowait
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            b[i][j] = i * j * ((double)rand());

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0.0;

    omp_set_dynamic(0);            // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads); // Use 8 threads for all consecutive parallel regions
    printf("using amount of threads:  %d \n", nthreads);
#pragma omp parallel
    {
        printf("working on thread:  %d ... \n", omp_get_thread_num());

#pragma omp for nowait
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                    c[i][j] += a[i][k] * b[k][j];
    }

    end = omp_get_wtime();
    printf("%f\n", end - start);
    exit(0);
}