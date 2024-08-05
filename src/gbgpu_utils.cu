// Code by Michael Katz. Based on code by Travis Robson, Neil Cornish, Tyson Littenberg, Stas Babak

// imports
#include "stdio.h"

#include "gbgpu_utils.hh"
#include "global.h"
#include "LISA.h"
#include "cuda_complex.hpp"

#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"
#else
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_cblas.h>
#endif

#define NUM_THREADS 256

// Add functionality for proper summation in the kernel
#ifdef __CUDACC__
CUDA_DEVICE
double atomicAddDouble(double *address, double val)
{
    unsigned long long *address_as_ull =
        (unsigned long long *)address;
    unsigned long long old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// Add functionality for proper summation in the kernel
CUDA_DEVICE
void atomicAddComplex(cmplx *a, cmplx b)
{
    // transform the addresses of real and imag. parts to double pointers
    double *x = (double *)a;
    double *y = x + 1;
    // use atomicAdd for double variables

#ifdef __CUDACC__
    atomicAddDouble(x, b.real());
    atomicAddDouble(y, b.imag());
#else
    *x += b.real();
    *y += b.imag();
#endif
}

// calculate batched log likelihood
CUDA_KERNEL
void fill_global(cmplx *A_glob, cmplx *E_glob, cmplx *A_template, cmplx *E_template, int *start_ind_all, int M, int num_bin, int *group_index, int data_length)
{
    // prepare loop based on CPU/GPU
    int start, end, increment;
#ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

#else

    start = 0;
    end = num_bin;
    increment = 1;

#endif
    for (int bin_i = start;
         bin_i < end;
         bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];
        int group_i = group_index[bin_i];

        for (int i = 0;
             i < M;
             i += 1)
        {
            int j = start_ind + i;

            if ((j >= data_length) || (j < 0))
            {
                continue;
            }
            cmplx temp_A = A_template[i * num_bin + bin_i];
            cmplx temp_E = E_template[i * num_bin + bin_i];

            int ind_out = group_i * data_length + j;
            atomicAddComplex(&A_glob[ind_out], temp_A);
            atomicAddComplex(&E_glob[ind_out], temp_E);
            // printf("CHECK: %d %e %e %d %d %d %d %d %d\n", bin_i, A_template[i * num_bin + bin_i], temp_A, group_i, data_length, j, num_groups, per_group, i);
        }
    }
}

// wrapper for log likelihood
void fill_global_wrap(cmplx *A_glob, cmplx *E_glob, cmplx *A_template, cmplx *E_template, int *start_ind_all, int M, int num_bin, int *group_index, int data_length)
{
// GPU / CPU difference
#ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS - 1) / NUM_THREADS);

    fill_global<<<num_blocks, NUM_THREADS>>>(
        A_glob, E_glob, A_template, E_template, start_ind_all, M, num_bin, group_index, data_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#else

    fill_global(
        A_glob, E_glob, A_template, E_template, start_ind_all, M, num_bin, group_index, data_length);

#endif
}

// calculate batched log likelihood
CUDA_KERNEL
void get_ll(cmplx *d_h, cmplx *h_h, cmplx *A_template, cmplx *E_template, cmplx *A_data, cmplx *E_data, double *A_psd, double *E_psd, double df, int *start_ind_all, int M, int num_bin, int *data_index, int *noise_index, int data_length)
{
    // prepare loop based on CPU/GPU
    int start, end, increment;
#ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

#else

    start = 0;
    end = num_bin;
    increment = 1;

#endif
    for (int bin_i = start;
         bin_i < end;
         bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];
        int data_index_bin_i = data_index[bin_i];
        int noise_index_bin_i = noise_index[bin_i];

        // initialize likelihood
        cmplx h_h_temp(0.0, 0.0);
        cmplx d_h_temp(0.0, 0.0);
        for (int i = 0;
             i < M;
             i += 1)
        {
            int j = start_ind + i;

            double A_noise = A_psd[noise_index_bin_i * data_length + j];
            double E_noise = E_psd[noise_index_bin_i * data_length + j];

            // calculate h term
            cmplx h_A = A_template[i * num_bin + bin_i];
            cmplx h_E = E_template[i * num_bin + bin_i];

            cmplx d_A = A_data[data_index_bin_i * data_length + j];
            cmplx d_E = E_data[data_index_bin_i * data_length + j];

            // get <d|h> term
            d_h_temp += gcmplx::conj(d_A) * h_A / A_noise;
            d_h_temp += gcmplx::conj(d_E) * h_E / E_noise;

            // <h|h>
            h_h_temp += gcmplx::conj(h_A) * h_A / A_noise;
            h_h_temp += gcmplx::conj(h_E) * h_E / E_noise;
        }

        // read out
        d_h[bin_i] = 4. * df * d_h_temp;
        h_h[bin_i] = 4. * df * h_h_temp;
    }
}

// wrapper for log likelihood
void get_ll_wrap(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 double *A_psd, double *E_psd, double df,
                 int *start_ind, int M, int num_bin, int *data_index, int *noise_index, int data_length)
{
// GPU / CPU difference
#ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS - 1) / NUM_THREADS);

    get_ll<<<num_blocks, NUM_THREADS>>>(
        d_h, h_h, A_template, E_template, A_data, E_data, A_psd, E_psd, df, start_ind, M, num_bin, data_index, noise_index, data_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#else

    get_ll(
        d_h, h_h, A_template, E_template, A_data, E_data, A_psd, E_psd, df, start_ind, M, num_bin, data_index, noise_index, data_length);

#endif
}

#ifdef __CUDACC__
void direct_like(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 int data_length, int start_freq_ind, int nwalkers)
{

    cudaStream_t streams[nwalkers];
    cublasHandle_t handle;

    cuDoubleComplex result_d_h[nwalkers];
    cuDoubleComplex result_h_h[nwalkers];

    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed\n");
        exit(0);
    }

    for (int walker_i = 0; walker_i < nwalkers; walker_i += 1)
    {

        cudaStreamCreate(&streams[walker_i]);

        cublasSetStream(handle, streams[walker_i]);
        stat = cublasZdotc(handle, data_length,
                           (cuDoubleComplex *)&A_data[start_freq_ind], 1,
                           (cuDoubleComplex *)&A_template[walker_i * data_length], 1,
                           &result_d_h[walker_i]);
        cudaStreamSynchronize(streams[walker_i]);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            exit(0);
        }

        cmplx *temp_results_d_h = (cmplx *)&result_d_h[walker_i];
        d_h[walker_i] += 4.0 * (*temp_results_d_h);

        cublasSetStream(handle, streams[walker_i]);
        stat = cublasZdotc(handle, data_length,
                           (cuDoubleComplex *)&A_template[walker_i * data_length], 1,
                           (cuDoubleComplex *)&A_template[walker_i * data_length], 1,
                           &result_h_h[walker_i]);
        cudaStreamSynchronize(streams[walker_i]);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            exit(0);
        }

        cmplx *temp_results_h_h = (cmplx *)&result_h_h[walker_i];
        h_h[walker_i] += 4.0 * (*temp_results_h_h);

        cublasSetStream(handle, streams[walker_i]);
        stat = cublasZdotc(handle, data_length,
                           (cuDoubleComplex *)&E_data[start_freq_ind], 1,
                           (cuDoubleComplex *)&E_template[walker_i * data_length], 1,
                           &result_d_h[walker_i]);
        cudaStreamSynchronize(streams[walker_i]);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            exit(0);
        }

        temp_results_d_h = (cmplx *)&result_d_h[walker_i];
        d_h[walker_i] += 4.0 * (*temp_results_d_h);

        cublasSetStream(handle, streams[walker_i]);
        stat = cublasZdotc(handle, data_length,
                           (cuDoubleComplex *)&E_template[walker_i * data_length], 1,
                           (cuDoubleComplex *)&E_template[walker_i * data_length], 1,
                           &result_h_h[walker_i]);
        cudaStreamSynchronize(streams[walker_i]);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            exit(0);
        }

        temp_results_h_h = (cmplx *)&result_h_h[walker_i];
        h_h[walker_i] += 4.0 * (*temp_results_h_h);
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    for (int walker_i = 0; walker_i < nwalkers; walker_i += 1)
    {
        // destroy the streams
        cudaStreamDestroy(streams[walker_i]);
    }
    cublasDestroy(handle);
}

#else
void direct_like(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 int data_length, int start_freq_ind, int nwalkers)
{

    cmplx result_d_h[nwalkers];
    cmplx result_h_h[nwalkers];

    for (int walker_i = 0; walker_i < nwalkers; walker_i += 1)
    {

        cblas_zdotc_sub(data_length,
                        (void *)&A_data[start_freq_ind], 1,
                        (void *)&A_template[walker_i * data_length], 1,
                        (void *)&result_d_h[walker_i]);

        d_h[walker_i] += 4.0 * result_d_h[walker_i];

        cblas_zdotc_sub(data_length,
                        (void *)&A_template[walker_i * data_length], 1,
                        (void *)&A_template[walker_i * data_length], 1,
                        (void *)&result_h_h[walker_i]);

        h_h[walker_i] += 4.0 * result_h_h[walker_i];

        cblas_zdotc_sub(data_length,
                        (void *)&E_data[start_freq_ind], 1,
                        (void *)&E_template[walker_i * data_length], 1,
                        (void *)&result_d_h[walker_i]);

        d_h[walker_i] += 4.0 * result_d_h[walker_i];

        cblas_zdotc_sub(data_length,
                        (void *)&E_template[walker_i * data_length], 1,
                        (void *)&E_template[walker_i * data_length], 1,
                        (void *)&result_h_h[walker_i]);

        h_h[walker_i] += 4.0 * result_h_h[walker_i];
    }
}
#endif
