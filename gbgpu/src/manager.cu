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

#ifdef __CUDACC__

#include "cuComplex.h"
#include "cublas_v2.h"
#include <kernel.cu>
#include "createGPUHolders.cu"
#include <cuda_runtime_api.h>
#include <cufft.h>
//#include <cuda.h>

#else

#include <kernel.cpp>

#endif

#include <manager.hh>
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

GBGPU::GBGPU (
    int data_stream_length_,
    double *data_freqs_,
    int N_,
    int nwalkers_,
    int ndevices_,
    double Tobs_,
    double dt_,
    int NP_){

      Tobs = Tobs_;
      dt = dt_;
      NP = NP_;


    #pragma omp parallel
    {
      if (omp_get_thread_num() == 1) printf("NUM OMP THREADS: %d\n", omp_get_num_threads());
    }


    data_freqs = data_freqs_;
    N = N_;
    data_stream_length = data_stream_length_;
    /*channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    data_channel1 = data_channel1_;
    data_channel2 = data_channel2_;
    data_channel3 = data_channel3_;*/
    nwalkers = nwalkers_;
    ndevices = ndevices_;

    ndevices = ndevices_;

    df = data_freqs[2] - data_freqs[1];

      gpuErrchk(cudaMalloc(&d_data_freqs, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMemcpy(d_data_freqs, data_freqs, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

      /*gpuErrchk(cudaMalloc(&d_data_channel1, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel2, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel3, data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_channel1_ASDinv, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel2_ASDinv, data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel3_ASDinv, data_stream_length*sizeof(double)));*/

      h_wfm = new Waveform[nwalkers];

      for (int i=0; i<nwalkers; i++){
          h_wfm[i].NP = NP;
          h_wfm[i].T = Tobs;
          h_wfm[i].N = N;
          N = h_wfm[i].N;
          alloc_waveform(&h_wfm[i]);
      }
      gpuErrchk(cudaMalloc(&wfm, nwalkers*sizeof(Waveform)));



      gpuErrchk(cudaMemcpy(wfm, h_wfm, nwalkers*sizeof(Waveform), cudaMemcpyHostToDevice));

      //gpuErrchk(cudaMalloc(&X_buffer, 2*N*nwalkers*sizeof(double)));
      //gpuErrchk(cudaMalloc(&Y_buffer, 2*N*nwalkers*sizeof(double)));
      //gpuErrchk(cudaMalloc(&Z_buffer, 2*N*nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&data12, 2*N*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&data21, 2*N*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&data13, 2*N*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&data31, 2*N*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&data23, 2*N*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&data32, 2*N*nwalkers*sizeof(double)));

      //gpuErrchk(cudaMalloc(&XLS, 2*data_stream_length*sizeof(double)));
      //gpuErrchk(cudaMalloc(&YLS, 2*data_stream_length*sizeof(double)));
      //gpuErrchk(cudaMalloc(&ZLS, 2*data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_like_out, nwalkers*3*sizeof(double)));

      //gpuErrchk(cudaMalloc(&XSL, 2*data_stream_length*sizeof(double)));
      //gpuErrchk(cudaMalloc(&YSL, 2*data_stream_length*sizeof(double)));
      //gpuErrchk(cudaMalloc(&ZSL, 2*data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_params, NP*nwalkers*sizeof(double)));


      // for likelihood
      // --------------
      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }

  if (cufftPlan1d(&plan, h_wfm->N, CUFFT_Z2Z, nwalkers) != CUFFT_SUCCESS){
        	fprintf(stderr, "CUFFT error: Plan creation failed");
        	return;	}

  //GBGPU::input_data(data_freqs, data_channel1,
    //                    data_channel2, data_channel3,
      //                  channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, data_stream_length);

}


void GBGPU::input_data(
    int data_stream_length_,
    long ptr_template_channel1_,
    long ptr_template_channel2_,
    long ptr_template_channel3_,
    long ptr_data_channel1_,
    long ptr_data_channel2_,
    long ptr_data_channel3_,
    long ptr_ASD_inv1_,
    long ptr_ASD_inv2_,
    long ptr_ASD_inv3_
){

    assert(data_stream_length_ == data_stream_length);
    XLS = (double *)ptr_template_channel1_;
    YLS = (double *)ptr_template_channel2_;
    ZLS = (double *)ptr_template_channel3_;

    d_data_channel1 = (agcmplx*)ptr_data_channel1_;
    d_data_channel2 = (agcmplx*)ptr_data_channel2_;
    d_data_channel3 = (agcmplx*)ptr_data_channel3_;

    d_channel1_ASDinv = (double*) ptr_ASD_inv1_;
    d_channel2_ASDinv = (double*) ptr_ASD_inv2_;
    d_channel3_ASDinv = (double*) ptr_ASD_inv3_;

}


__global__
void fill_params(Waveform *wfm_trans, double *params, int nwalkers, int NP)
{
    Waveform *wfm;
  for (int walker_i = blockIdx.x * blockDim.x + threadIdx.x;
       walker_i < nwalkers;
       walker_i += blockDim.x * gridDim.x){
         wfm = &wfm_trans[walker_i];

         for (int i=0; i<NP; i++) wfm->params[i] = params[walker_i*NP + i];
         wfm->q  = (long)(wfm->params[0]);

  }
}


void GBGPU::Fast_GB(double *params_){//,double *XLS, double *YLS, double *ZLS,double* XSL, double* YSL, double* ZSL){

    gpuErrchk(cudaMemcpy(d_params, params_, nwalkers*NP*sizeof(double), cudaMemcpyHostToDevice));

      //gpuErrchk(cudaMemcpy(wfm, h_wfm, nwalkers*sizeof(Waveform), cudaMemcpyHostToDevice));

      int NUM_THREADS = 256;
      int num_blocks_basis_tensors = std::ceil((nwalkers + NUM_THREADS -1)/NUM_THREADS);

      fill_params<<<num_blocks_basis_tensors,NUM_THREADS>>>(wfm, d_params, nwalkers,NP);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

    	get_basis_tensors<<<num_blocks_basis_tensors,NUM_THREADS>>>(wfm, nwalkers);      //  Tensor construction for building slowly evolving LISA response
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

      int num_blocks = std::ceil((h_wfm->N + NUM_THREADS -1)/NUM_THREADS);
      dim3 gridDim(num_blocks, nwalkers);
      GenWave<<<gridDim, NUM_THREADS>>>(wfm, h_wfm->N, nwalkers,
                                         data12, data21, data13, data31, data23, data32);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

      fft_data(data12, data21, data13, data31, data23, data32, plan, nwalkers);

      unpack_data_1<<<gridDim, NUM_THREADS>>>(wfm, data12, data21, data13, data31, data23, data32, nwalkers);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());

      /*unpack_data_2<<<gridDim, NUM_THREADS>>>(wfm, nwalkers);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());*/

      XYZ_wrap<<<gridDim, NUM_THREADS>>>(wfm, nwalkers, N, dt, Tobs, XLS, YLS, ZLS, df);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
}


__global__ void calc_like(double *like_out, Waveform *wfm_trans, agcmplx *Xarr_all, agcmplx *Yarr_all, agcmplx *Zarr_all,
                          double *channel1_ASDinv, double *channel2_ASDinv, double *channel3_ASDinv,
                          agcmplx *data_channel1, agcmplx *data_channel2, agcmplx *data_channel3,
                          int nwalkers, int M)
{

    agcmplx A, E, T, X, Y, Z, Ainv, Einv, Tinv, A_data, E_data, T_data, A_temp, E_temp, T_temp;
    int add_ind;

    for (int walker_i = blockIdx.x * blockDim.x + threadIdx.x;
             walker_i < nwalkers;
             walker_i += blockDim.x * gridDim.x){

    Waveform * wfm = &wfm_trans[walker_i];
    int N = wfm->N;
    int mid_ind = wfm->q;
    int start_ind = walker_i*M;

    agcmplx *Xarr = &Xarr_all[start_ind];
    agcmplx *Yarr = &Yarr_all[start_ind];
    agcmplx *Zarr = &Zarr_all[start_ind];

    double temp_h_h = 0.0;
    double temp_d_h = 0.0;
    double temp_d_minus_h = 0.0;

    for (int i = 0;
             i < M;
             i += 1)
    {

        add_ind = (mid_ind + i - M/2);

        A = Xarr[i];
        E = Yarr[i];
        T = Zarr[i];

        Ainv = channel1_ASDinv[add_ind];
        Einv = channel2_ASDinv[add_ind];
        Tinv = channel3_ASDinv[add_ind];

        A_data = data_channel1[add_ind];
        E_data = data_channel2[add_ind];
        T_data = data_channel3[add_ind];

        A_temp = A*Ainv;
        E_temp = E*Einv;
        T_temp = T*Tinv;



        temp_h_h += gcmplx::real(gcmplx::conj(A_temp)*A_temp);
        temp_h_h += gcmplx::real(gcmplx::conj(E_temp)*E_temp);
        temp_h_h += gcmplx::real(gcmplx::conj(T_temp)*T_temp);

        temp_d_h += gcmplx::real(gcmplx::conj(A_data)*A_temp);
        temp_d_h += gcmplx::real(gcmplx::conj(E_data)*E_temp);
        temp_d_h += gcmplx::real(gcmplx::conj(T_data)*T_temp);

        agcmplx A_d_minus_h = A_data - A_temp;
        agcmplx E_d_minus_h = E_data - E_temp;
        agcmplx T_d_minus_h = T_data - T_temp;

        temp_d_minus_h += gcmplx::real(gcmplx::conj(A_d_minus_h)*A_d_minus_h);
        temp_d_minus_h += gcmplx::real(gcmplx::conj(E_d_minus_h)*E_d_minus_h);
        temp_d_minus_h += gcmplx::real(gcmplx::conj(T_d_minus_h)*T_d_minus_h);

   }

   like_out[3*walker_i] = 4.0*temp_d_h;
   like_out[3*walker_i + 1] = 4.0*temp_h_h;
   like_out[3*walker_i + 2] = 4.0*temp_d_minus_h;
 }
}


void GBGPU::Likelihood(double *likelihood){
    dim3 likeDim(nwalkers);
    int NUM_THREADS = 256;

    calc_like<<<likeDim, NUM_THREADS>>>(d_like_out, wfm, (agcmplx*)XLS, (agcmplx*)YLS, (agcmplx*)ZLS,
                              d_channel1_ASDinv, d_channel2_ASDinv, d_channel3_ASDinv,
                              d_data_channel1, d_data_channel2, d_data_channel3,
                              nwalkers, N);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(likelihood, d_like_out, nwalkers*3*sizeof(double), cudaMemcpyDeviceToHost));
}

/*
Destructor
*/
GBGPU::~GBGPU() {

  gpuErrchk(cudaFree(d_data_freqs));
/*  gpuErrchk(cudaFree(d_data_channel1));
  gpuErrchk(cudaFree(d_data_channel2));
  gpuErrchk(cudaFree(d_data_channel3));

  gpuErrchk(cudaFree(d_channel1_ASDinv));
  gpuErrchk(cudaFree(d_channel2_ASDinv));
  gpuErrchk(cudaFree(d_channel3_ASDinv));*/
  free_waveform(wfm);
  //free(h_wfm);
  delete[] h_wfm;
  cublasDestroy(handle);

//gpuErrchk(cudaFree(X_buffer));
//gpuErrchk(cudaFree(Y_buffer));
//gpuErrchk(cudaFree(Z_buffer));

gpuErrchk(cudaFree(d_params));

//gpuErrchk(cudaFree(XLS));
//gpuErrchk(cudaFree(YLS));
//gpuErrchk(cudaFree(ZLS));
gpuErrchk(cudaFree(d_like_out));

gpuErrchk(cudaFree(XSL));
gpuErrchk(cudaFree(YSL));
gpuErrchk(cudaFree(ZSL));

gpuErrchk(cudaFree(data12));
gpuErrchk(cudaFree(data21));
gpuErrchk(cudaFree(data13));
gpuErrchk(cudaFree(data31));
gpuErrchk(cudaFree(data23));
gpuErrchk(cudaFree(data32));

cufftDestroy(plan);
gpuErrchk(cudaFree(wfm));
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
