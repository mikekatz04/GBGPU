/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This is the central piece of code. This file implements a class
 *  that takes data in on the cpu side, copies
 *  it to the gpu, and exposes functions that let
 *  you perform actions with the GPU.
 *
 *  This class will get translated into python via cython.
 *
 *
 *
 *  Copyright (C) 2019 Michael Katz
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */


#include "cuComplex.h"
#include "cublas_v2.h"
#include <cuda_runtime_api.h>
//#include <cuda.h>


#include <likelihood.hh>
#include <assert.h>
#include <iostream>
#include <complex>

#include "omp.h"
//#include "cuda_complex.hpp"
// TODO: CUTOFF PHASE WHEN IT STARTS TO GO BACK UP!!!

using namespace std;

#define BATCH 1

#ifdef __CUDACC__
void print_mem_info(){
        // show memory usage of GPU

        cudaError_t cuda_status;

        size_t free_byte ;

        size_t total_byte ;

        cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

            exit(1);

        }



        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

#endif

Likelihood::Likelihood (
    int data_stream_length_,
    double *data_freqs_,
    cmplx *data_channel1_,
    cmplx *data_channel2_,
    cmplx *data_channel3_,
    long ptr_template_channel1_,
    long ptr_template_channel2_,
    long ptr_template_channel3_,
    double *channel1_ASDinv_,
    double *channel2_ASDinv_,
    double *channel3_ASDinv_,
    int nwalkers_,
    int ndevices_,
    double Tobs_,
    double dt_){

      Tobs = Tobs_;
      dt = dt_;


    #pragma omp parallel
    {
      if (omp_get_thread_num() == 1) printf("NUM OMP THREADS: %d\n", omp_get_num_threads());
    }


    data_freqs = data_freqs_;
    data_stream_length = data_stream_length_;
    channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    data_channel1 = data_channel1_;
    data_channel2 = data_channel2_;
    data_channel3 = data_channel3_;
    nwalkers = nwalkers_;
    ndevices = ndevices_;

    ndevices = ndevices_;

      gpuErrchk(cudaMalloc(&d_data_freqs, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_data_channel1, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel2, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel3, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_channel1_ASDinv, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel2_ASDinv, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel3_ASDinv, data_stream_length*sizeof(double)));

      d_template_channel1 = (agcmplx *)ptr_template_channel1_;
      d_template_channel2 = (agcmplx *)ptr_template_channel2_;
      d_template_channel3 = (agcmplx *)ptr_template_channel3_;

      // for likelihood
      // --------------
      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }

  Likelihood::input_data(data_freqs, data_channel1,
                        data_channel2, data_channel3,
                        channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, data_stream_length);

}


void Likelihood::input_data(double *data_freqs_, cmplx *data_channel1_,
                          cmplx *data_channel2_, cmplx *data_channel3_,
                          double *channel1_ASDinv_, double *channel2_ASDinv_,
                          double *channel3_ASDinv_, int data_stream_length_){

    assert(data_stream_length_ == data_stream_length);

        gpuErrchk(cudaMemcpy(d_data_freqs, data_freqs_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel1, data_channel1_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel2, data_channel2_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel3, data_channel3_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel1_ASDinv, channel1_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel2_ASDinv, channel2_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel3_ASDinv, channel3_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));
}


void Likelihood::GetLikelihood(double *likelihood){

  double d_h = 0.0;
  double h_h = 0.0;
  char * status;
  double res;
  cuDoubleComplex result;
  cublasStatus_t stat;

          // get data - template terms
           stat = cublasZdotc(handle, data_stream_length,
                   (cuDoubleComplex*)d_template_channel1, 1,
                   (cuDoubleComplex*)d_data_channel1, 1,
                   &result);

            d_h += cuCreal(result);

            stat = cublasZdotc(handle, data_stream_length,
                    (cuDoubleComplex*)d_template_channel2, 1,
                    (cuDoubleComplex*)d_data_channel2, 1,
                    &result);
             d_h += cuCreal(result);

             stat = cublasZdotc(handle, data_stream_length,
                     (cuDoubleComplex*)d_template_channel3, 1,
                     (cuDoubleComplex*)d_data_channel3, 1,
                     &result);

           d_h += cuCreal(result);

           //<h|h>
           stat = cublasZdotc(handle, data_stream_length,
                   (cuDoubleComplex*)d_template_channel1, 1,
                   (cuDoubleComplex*)d_template_channel1, 1,
                   &result);

            h_h += cuCreal(result);

            stat = cublasZdotc(handle, data_stream_length,
                    (cuDoubleComplex*)d_template_channel2, 1,
                    (cuDoubleComplex*)d_template_channel2, 1,
                    &result);

             h_h += cuCreal(result);

             stat = cublasZdotc(handle, data_stream_length,
                     (cuDoubleComplex*)d_template_channel3, 1,
                     (cuDoubleComplex*)d_template_channel3, 1,
                     &result);

           h_h += cuCreal(result);

           likelihood[0] = 4*d_h;
           likelihood[1] = 4*h_h;
}

/*
Destructor
*/
Likelihood::~Likelihood() {

  gpuErrchk(cudaFree(d_data_freqs));
  gpuErrchk(cudaFree(d_data_channel1));
  gpuErrchk(cudaFree(d_data_channel2));
  gpuErrchk(cudaFree(d_data_channel3));

  gpuErrchk(cudaFree(d_channel1_ASDinv));
  gpuErrchk(cudaFree(d_channel2_ASDinv));
  gpuErrchk(cudaFree(d_channel3_ASDinv));

  cublasDestroy(handle);

}

int GetDeviceCount(){
    int num_device_check;
    #ifdef __CUDACC__
    cudaError_t cuda_status = cudaGetDeviceCount(&num_device_check);
    if (cudaSuccess != cuda_status) num_device_check = 0;
    #else
    num_device_check = 0;
    #endif
    printf("NUMBER OF DEVICES: %d\n", num_device_check);
    return num_device_check;
}
