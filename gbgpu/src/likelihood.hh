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

#ifndef __MANAGER_H__
#define __MANAGER_H__

#include "cuComplex.h"
#include "cublas_v2.h"

#include "global.h"
#include <complex>

class Likelihood {
  // pointer to the GPU memory where the array is stored
  int current_status;
  int data_stream_length;
  int current_length;
  int nwalkers;
  int to_gpu;
  double Tobs;
  double dt;

  cublasHandle_t handle;
  cublasStatus_t stat;


  int ndevices;

  double *data_freqs;
  cmplx *data_channel1;
  cmplx *data_channel2;
  cmplx *data_channel3;

  double *channel1_ASDinv;
  double *channel2_ASDinv;
  double *channel3_ASDinv;

  double *d_data_freqs;
  agcmplx *d_data_channel1;
  agcmplx *d_data_channel2;
  agcmplx *d_data_channel3;

  double *d_channel1_ASDinv;
  double *d_channel2_ASDinv;
  double *d_channel3_ASDinv;

  agcmplx *d_template_channel1;
  agcmplx *d_template_channel2;
  agcmplx *d_template_channel3;

public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

   Likelihood (
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
       double dt_); // constructor (copies to GPU)

  void GetLikelihood(double *like);
  void input_data(double *data_freqs_, cmplx *data_channel1_,
                            cmplx *data_channel2_, cmplx *data_channel3_,
                            double *channel1_ASDinv_, double *channel2_ASDinv_,
                            double *channel3_ASDinv_, int data_stream_length_);

  ~Likelihood(); // destructor


};

int GetDeviceCount();

#endif //__MANAGER_H__
