/*  This code was edited by Michael Katz. The information in this file is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited and added to
 *  the code for his purposes and removed dependencies on the LAL libraries.
 *  This code is distrbuted under the same GNU license it originally came with.

 *  Copyright (C) 2017 Sebastian Khan, Francesco Pannarale, Lionel London
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


#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include "cuda_complex.hpp"
#include "Constants.h"
#include "cuda_runtime_api.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_KERNEL __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#endif


typedef std::complex<double> cmplx;
typedef gcmplx::complex<double> agcmplx;


typedef struct tagGB
{
	double T;			// observation period
	double f0;			// initial GW freq
	double theta, phi;  // sky-location (spherical-polar)
	double amp, iota;   // amplitude and inclination angle
	double psi, phi0;   // polarization angle, initial phase

	double cosiota;
	double costheta;

	double *params;		// vector to store parameters

	long q, N;  		// carrier freq bin, number of samples

	long NP;			// number of parameters
} GB;

typedef struct tagWaveform
{
	long N;
	long q; // Fgw carrier bin

	int NP;		// number of parameters in signal

	double T; 		// observation period

	double *params;

	double *eplus, *ecross;
	double *dplus, *dcross;

	double DPr, DPi, DCr, DCi;

	// direction vector of GW
	double *k;

	// separation unit vectors between S/C
	double *r12, *r21;
	double *r13, *r31;
	double *r23, *r32;

	double *kdotr;
	double *kdotx;

	double *xi, *f, *fonfs;

	// Time series of slowly evolving terms at each vertex
	// dataij corresponds to fractional arm length difference yij
	double *data12, *data21;
	double *data13, *data31;
	double *data23, *data32;

	// Fourier coefficients of slowly evolving terms (numerical)
	double *a12, *a21;
	double *a13, *a31;
	double *a23, *a32;

	// S/C position
	double *x, *y, *z;

	// Time varrying quantities (Re & Im) broken up into convenient segments
	double *TR, *TI;

	//Package cij's into proper form for TDI subroutines
	double *d;
} Waveform;

/*
Function for gpu Error checking.
//*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



#endif
