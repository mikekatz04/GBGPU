/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.

 * This code is adjusted for usage in CUDA. Refer to PhenomHM.cpp for comments.


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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_complex.hpp"
#include "Constants.h"
#include "LISA.h"
#include <assert.h>

#include "global.h"

#ifdef __CUDACC__
#else
#include "omp.h"
#endif

#include <cufft.h>

__device__
void set_const_trans(Waveform *wfm)
{
	double amp, cosiota;
	double Aplus, Across;
	double psi;
	double sinps, cosps;

	amp      = exp(wfm->params[3]);
	cosiota  = wfm->params[4];
	psi      = wfm->params[5];

	//Calculate GW polarization amplitudes
	Aplus  = amp*(1. + cosiota*cosiota);
	// Aplus  = -amp*(1. + cosiota*cosiota);
	Across = -2.0*amp*cosiota;
	//Across = 2.0*amp*cosiota;

	//Calculate cos and sin of polarization
	cosps = cos(2.*psi);
	sinps = sin(2.*psi);

	//Calculate constant pieces of transfer functions
	wfm->DPr    =  Aplus*cosps;
	wfm->DPi    = -Across*sinps;
	wfm->DCr    = -Aplus*sinps;
	wfm->DCi    = -Across*cosps;

	return;
}

__device__
void get_basis_vecs(double *params, double *u, double *v, double *k)
{
	long i;

	double phi;
	double costh, sinth, cosph, sinph;

	for (i=0; i<3; i++)
	{
		u[i] = 0.;
		v[i] = 0.;
		k[i] = 0.;
	}

	phi	  = params[2];
	costh = params[1];

	sinth = sqrt(1.0-costh*costh);

	cosph = cos(phi);
	sinph = sin(phi);

	u[0] =  costh*cosph;  u[1] =  costh*sinph;  u[2] = -sinth;
	v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
	k[0] = -sinth*cosph;  k[1] = -sinth*sinph;  k[2] = -costh;

	return;
}

__global__
void get_basis_tensors(Waveform *wfm_trans, int nwalkers)
{

	Waveform * wfm;
	long i, j;

	 // GW basis vectors

	double u[3];
	double v[3];

	for (int walker_i = blockIdx.x * blockDim.x + threadIdx.x;
			 walker_i < nwalkers;
			 walker_i += blockDim.x * gridDim.x){

	wfm = &wfm_trans[walker_i];

	set_const_trans(wfm);  // set the constant pieces of transfer function

	get_basis_vecs(wfm->params, u, v, wfm->k); //Gravitational Wave source basis vectors

	//GW polarization basis tensors
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
		{
			//wfm->eplus[i][j]  = u[i]*u[j] - v[i]*v[j];
			wfm->eplus[i*3 + j]  = v[i]*v[j] - u[i]*u[j];
			wfm->ecross[i*3 + j] = u[i]*v[j] + v[i]*u[j];
			//wfm->ecross[i][j] = -u[i]*v[j] - v[i]*u[j];
		}
	}
}

	return;
}

__device__
void spacecraft(double t, double *x, double *y, double *z, int n, int N)
{
	double alpha;
	double beta1, beta2, beta3;
	double sa, sb, ca, cb;

	alpha = 2.*M_PI*fm*t + kappa;

	beta1 = 0. + lambda;
	beta2 = 2.*M_PI/3. + lambda;
	beta3 = 4.*M_PI/3. + lambda;

	sa = sin(alpha);
	ca = cos(alpha);

	sb = sin(beta1);
	cb = cos(beta1);
	x[0] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[0] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[0] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta2);
	cb = cos(beta2);
	x[1] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[1] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[1] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta3);
	cb = cos(beta3);
	x[2] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[2] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[2] = -SQ3*AU*ec*(ca*cb + sa*sb);

	return;
}

__device__
void calc_xi_f(Waveform *wfm, double t, int n, int N, double *x, double *y, double *z, double *xi, double *fonfs)
{
	long i;

	double f0, dfdt_0, d2fdt2_0;

	double kdotx[3];
	double f[3];

	f0       = wfm->params[0]/wfm->T;
	if (wfm->NP > 7) dfdt_0   = wfm->params[7]/wfm->T/wfm->T;
	if (wfm->NP > 8) d2fdt2_0 = wfm->params[8]/wfm->T/wfm->T/wfm->T;

	spacecraft(t, x, y, z, n, N); // Calculate position of each spacecraft at time t

	for(i=0; i<3; i++)
	{
		kdotx[i] = (x[i]*wfm->k[0] + y[i]*wfm->k[1] + z[i]*wfm->k[2])/C;
		//Wave arrival time at spacecraft i
		xi[i]    = t - kdotx[i];
		//FIXME
		//xi[i]    = t + kdotx[i];
		//First order approximation to frequency at spacecraft i
		f[i]     = f0;
		if (wfm->NP > 7) f[i] += dfdt_0*xi[i];
		if (wfm->NP > 8) f[i] += 0.5*d2fdt2_0*xi[i]*xi[i];

		//Ratio of true frequency to transfer frequency
		fonfs[i] = f[i]/fstar;
	}

	return;
}

__device__
void calc_sep_vecs(Waveform *wfm, int n, int N, double *x, double *y, double *z, double *r12, double *r21, double *r13, double *r31, double *r23, double *r32)
{
	long i;

	//Unit separation vector from spacecrafts i to j
	r12[0] = (x[1] - x[0])/Larm;
	r13[0] = (x[2] - x[0])/Larm;
	r23[0] = (x[2] - x[1])/Larm;
	r12[1] = (y[1] - y[0])/Larm;
	r13[1] = (y[2] - y[0])/Larm;
	r23[1] = (y[2] - y[1])/Larm;
	r12[2] = (z[1] - z[0])/Larm;
	r13[2] = (z[2] - z[0])/Larm;
	r23[2] = (z[2] - z[1])/Larm;

	//Make use of symmetry
	for(i=0; i<3; i++)
	{
		r21[i] = -r12[i];
		r31[i] = -r13[i];
		r32[i] = -r23[i];
	}
	return;
}

__device__
void calc_d_matrices(Waveform *wfm, int n, int N, double *dcross, double *dplus, double *r12, double *r21, double *r13, double *r31, double *r23, double *r32)
{
	long i, j;

	//Zero arrays to be summed
	dplus [(0*3 + 1)] = 0.0;
	dplus [(0*3 + 2)] = 0.0;
	dplus [(1*3 + 0)] = 0.;
	dplus [(1*3 + 2)] = 0.0;
	dplus [(2*3 + 0)] = 0.0;
	dplus [(2*3 + 1)] = 0.;
	dcross[(0*3 + 1)] = 0.0;
	dcross[(0*3 + 2)] = 0.0;
	dcross[(1*3 + 0)] = 0.;
	dcross[(1*3 + 2)] = 0.0;
	dcross[(2*3 + 0)] = 0.0;
	dcross[(2*3 + 1)] = 0.;

	//Convenient quantities d+ & dx
	for(i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			dplus [(0*3 + 1)] += r12[i]*r12[j]*wfm->eplus[i*3 + j];
			dcross[(0*3 + 1)] += r12[i]*r12[j]*wfm->ecross[i*3 + j];
			dplus [(1*3 + 2)] += r23[i]*r23[j]*wfm->eplus[i*3 + j];
			dcross[(1*3 + 2)] += r23[i]*r23[j]*wfm->ecross[i*3 + j];
			dplus [(0*3 + 2)] += r13[i]*r13[j]*wfm->eplus[i*3 + j];
			dcross[(0*3 + 2)] += r13[i]*r13[j]*wfm->ecross[i*3 + j];
		}
	}
	//Makng use of symmetry
	dplus[(1*3 + 0)] = dplus[(0*3 + 1)];  dcross[(1*3 + 0)] = dcross[(0*3 + 1)];
	dplus[(2*3 + 1)] = dplus[(1*3 + 2)];  dcross[(2*3 + 1)] = dcross[(1*3 + 2)];
	dplus[(2*3 + 0)] = dplus[(0*3 + 2)];  dcross[(2*3 + 0)] = dcross[(0*3 + 2)];

	return;
}


__device__
void calc_kdotr(Waveform *wfm, int n, int N, double *kdotr, double *r12, double *r21, double *r13, double *r31, double *r23, double *r32)
{
	long i;

	//Zero arrays to be summed
	kdotr[(0*3 + 1)] = 0.0;
	kdotr[(0*3 + 2)] = 0.0;
	kdotr[(1*3 + 0)] = 0.;
	kdotr[(1*3 + 2)] = 0.0;
	kdotr[(2*3 + 0)] = 0.0;
	kdotr[(2*3 + 1)] = 0.;

	for(i=0; i<3; i++)
	{
		kdotr[(0*3 + 1)] += wfm->k[i]*r12[i];
		kdotr[(0*3 + 2)] += wfm->k[i]*r13[i];
		kdotr[(1*3 + 2)] += wfm->k[i]*r23[i];
	}

	//Making use of antisymmetry
	kdotr[(1*3 + 0)] = -kdotr[(0*3 + 1)];
	kdotr[(2*3 + 0)] = -kdotr[(0*3 + 2)];
	kdotr[(2*3 + 1)] = -kdotr[(1*3 + 2)];

	return;
}


__device__

void get_transfer(Waveform *wfm, double t, int n, int N, double *kdotr, double *TR, double *TI, double *dplus, double *dcross,
									double *xi, double *fonfs)
{
	long i, j;
	long q;

	double tran1r, tran1i;
	double tran2r, tran2i;
	double aevol;			// amplitude evolution factor
	double arg1, arg2, sinc;
	double f0, dfdt_0, d2fdt2_0;
	double df, phi0;

	f0       = wfm->params[0]/wfm->T;
	phi0     = wfm->params[6];

	if (wfm->NP > 7) dfdt_0   = wfm->params[7]/wfm->T/wfm->T;
 	if (wfm->NP > 8) d2fdt2_0 = wfm->params[8]/wfm->T/wfm->T/wfm->T;

	q  = wfm->q;
	df = PI2*(((double)q)/wfm->T);

	for(i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			if(i!=j)
			{
				//Argument of transfer function
				// FIXME
				//arg1 = 0.5*fonfs[i]*(1. - kdotr[i][j]);
				arg1 = 0.5*fonfs[i]*(1. + kdotr[(i*3 + j)]);

				//Argument of complex exponentials
				arg2 = PI2*f0*xi[i] + phi0 - df*t;

				if (wfm->NP > 7) arg2 += M_PI*dfdt_0*xi[i]*xi[i];
				if (wfm->NP > 8) arg2 += M_PI*d2fdt2_0*xi[i]*xi[i]*xi[i]/3.0 ;

				//Transfer function
				sinc = 0.25*sin(arg1)/arg1;

				//Evolution of amplitude
				aevol = 1.0;
				if (wfm->NP > 7) aevol += 0.66666666666666666666*dfdt_0/f0*xi[i];

				///Real and imaginary pieces of time series (no complex exponential)
				tran1r = aevol*(dplus[(i*3 + j)]*wfm->DPr + dcross[(i*3 + j)]*wfm->DCr);
				tran1i = aevol*(dplus[(i*3 + j)]*wfm->DPi + dcross[(i*3 + j)]*wfm->DCi);

				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);

				//Real & Imaginary part of the slowly evolving signal
				TR[(i*3 + j)] = sinc*(tran1r*tran2r - tran1i*tran2i);
				TI[(i*3 + j)] = sinc*(tran1r*tran2i + tran1i*tran2r);
			}
		}
	}

	return;
}


__device__

void fill_time_series(int walker_i, int n, int N, double *TR, double *TI,
											double *data12, double *data21, double *data13,
										  double *data31, double *data23, double *data32)
{
	data12[(walker_i*2*N) + 2*n]   = TR[(0*3 + 1)];
	data21[(walker_i*2*N) + 2*n]   = TR[(1*3 + 0)];
	data31[(walker_i*2*N) + 2*n]   = TR[(2*3 + 0)];
	data12[(walker_i*2*N) + 2*n+1] = TI[(0*3 + 1)];
	data21[(walker_i*2*N) + 2*n+1] = TI[(1*3 + 0)];
	data31[(walker_i*2*N) + 2*n+1] = TI[(2*3 + 0)];
	data13[(walker_i*2*N) + 2*n]   = TR[(0*3 + 2)];
	data23[(walker_i*2*N) + 2*n]   = TR[(1*3 + 2)];
	data32[(walker_i*2*N) + 2*n]   = TR[(2*3 + 1)];
	data13[(walker_i*2*N) + 2*n+1] = TI[(0*3 + 2)];
	data23[(walker_i*2*N) + 2*n+1] = TI[(1*3 + 2)];
	data32[(walker_i*2*N) + 2*n+1] = TI[(2*3 + 1)];

	return;
}


__global__
void GenWave(Waveform *wfm_trans, int N, int nwalkers,
						 double *data12, double *data21, double *data13,
						 double *data31, double *data23, double *data32){
	double t=0.0;
	Waveform *wfm;
	int tid = (int)threadIdx.x;

	__shared__ double kdotr[9*256];
	double TR[9];
	double TI[9];
	double dplus[9];
	double dcross[9];
	double x[3], y[3], z[3];
	double xi[3], fonfs[3];
	double r12[3], r21[3], r13[3], r31[3], r23[3], r32[3];

	for (int walker_i = blockIdx.y * blockDim.y + threadIdx.y;
			 walker_i < nwalkers;
			 walker_i += blockDim.y * gridDim.y){

	wfm = &wfm_trans[walker_i];

	for (int n = blockIdx.x * blockDim.x + threadIdx.x;
			 n < N;
			 n += blockDim.x * gridDim.x){

				 t = wfm->T*(double)(n)/(double)N;
				 calc_xi_f(wfm ,t, n, N, x, y, z, xi, fonfs);		  // calc frequency and time variables
				 calc_sep_vecs(wfm, n, N, x, y, z, r12, r21, r13, r31, r23, r32);       // calculate the S/C separation vectors
				 calc_d_matrices(wfm, n, N, dplus, dcross, r12, r21, r13, r31, r23, r32);     // calculate pieces of waveform
				 calc_kdotr(wfm, n, N, &kdotr[tid*9], r12, r21, r13, r31, r23, r32);		  // calculate dot product
				 get_transfer(wfm, t, n, N, &kdotr[tid*9], TR, TI, dplus, dcross, xi, fonfs);     // Calculating Transfer function
				 fill_time_series(walker_i, n, N, TR, TI, data12, data21, data13, data31, data23, data32); // Fill  time series data arrays with slowly evolving signal.
		}
}

}

// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
    fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err,
                                _cudaGetErrorEnum(err));
    cudaDeviceReset(); assert(0);
	}
}


void fft_data(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32, cufftHandle plan, int nwalkers)
{

	cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex*)data12, (cufftDoubleComplex*)data12, -1));
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data21, (cufftDoubleComplex*)data21, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data31, (cufftDoubleComplex*)data31, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data13, (cufftDoubleComplex*)data13, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data23, (cufftDoubleComplex*)data23, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data32, (cufftDoubleComplex*)data32, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	return;
}

__global__
void unpack_data_1(Waveform *wfm_trans, double *data12, double *data21, double *data13,
double *data31, double *data23, double *data32, int nwalkers)
{

	Waveform *wfm;
	for (int walker_i = blockIdx.y * blockDim.y + threadIdx.y;
			 walker_i < nwalkers;
			 walker_i += blockDim.y * gridDim.y){

		wfm = &wfm_trans[walker_i];
		int N = wfm->N;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			 i < N;
			 i += blockDim.x * gridDim.x){
		// populate from most negative (Nyquist) to most positive (Nyquist-1)
		wfm->a12[i]   = 0.5*data12[(walker_i*2*N) + N+i]/(double)N;  // moved the 0.5
		wfm->a21[i]   = 0.5*data21[(walker_i*2*N) + N+i]/(double)N;
		wfm->a31[i]   = 0.5*data31[(walker_i*2*N) + N+i]/(double)N;
		wfm->a12[i+N] = 0.5*data12[(walker_i*2*N) + i]/(double)N;
		wfm->a21[i+N] = 0.5*data21[(walker_i*2*N) + i]/(double)N;
		wfm->a31[i+N] = 0.5*data31[(walker_i*2*N) + i]/(double)N;
		wfm->a13[i]   = 0.5*data13[(walker_i*2*N) + N+i]/(double)N;
		wfm->a23[i]   = 0.5*data23[(walker_i*2*N) + N+i]/(double)N;
		wfm->a32[i]   = 0.5*data32[(walker_i*2*N) + N+i]/(double)N;
		wfm->a13[i+N] = 0.5*data13[(walker_i*2*N) + i]/(double)N;
		wfm->a23[i+N] = 0.5*data23[(walker_i*2*N) + i]/(double)N;
		wfm->a32[i+N] = 0.5*data32[(walker_i*2*N) + i]/(double)N;
	}
}
}

/*
__global__
void unpack_data_2(Waveform *wfm_trans, int nwalkers)
{
	//   Renormalize so that the resulting time series is real

	Waveform *wfm;
	for (int walker_i = blockIdx.y * blockDim.y + threadIdx.y;
			 walker_i < nwalkers;
			 walker_i += blockDim.y * gridDim.y){

		wfm = &wfm_trans[walker_i];
		int N = wfm->N;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			 i < 2*N;
			 i += blockDim.x * gridDim.x)
	{
		wfm->d[0*3*2*N + 1*2*N + i] = 0.5*wfm->a12[i];
		wfm->d[1 *3*2*N + 0 *2*N + i] = 0.5*wfm->a21[i];
		wfm->d[2 *3*2*N + 0 *2*N + i] = 0.5*wfm->a31[i];
		wfm->d[0*3*2*N + 2*2*N + i] = 0.5*wfm->a13[i];
		wfm->d[1 *3*2*N + 2 *2*N + i] = 0.5*wfm->a23[i];
		wfm->d[2 *3*2*N + 1 *2*N + i] = 0.5*wfm->a32[i];
	}
}
}
*/


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}


__device__
void XYZ(int i, double *a12, double *a21, double *a13, double *a31, double *a23, double *a32, double f0, long q, long M, double dt, double Tobs, double *XLS_r, double *YLS_r, double *ZLS_r,
					double* XSL_r, double* YSL_r, double* ZSL_r, double *XLS_i, double *YLS_i, double *ZLS_i, double *XSL_i, double *YSL_i, double *ZSL_i)
{
	double fonfs;
	double c3, s3, c2, s2, c1, s1;
	double f;
	double phiLS, cLS, sLS, phiSL, cSL, sSL;

	double X_1, X_2, Y_1, Y_2, Z_1, Z_2;

	// YLS = malloc(2*M*sizeof(double));
	// ZLS = malloc(2*M*sizeof(double));

	phiLS = PI2*f0*(dt/2.0-Larm/C);

	cLS = cos(phiLS);
	sLS = sin(phiLS);

	//double phiLS = 2.0*pi*f0*(dt/2.0-L/clight);
	//double cLS = cos(phiLS); double sLS = sin(phiLS);

	phiSL = M_PI/2.0-2.0*M_PI*f0*(Larm/C);
	cSL = cos(phiSL);
	sSL = sin(phiSL);

  //printf("Stas, q=%ld, f0=%f, check: %f, %f \n", q, f0, q/Tobs, Tobs);

		f = ((double)(q + i - M/2))/Tobs;
		//if (i == 0){
		//		double f1 = ((double)(q + i -1 - M/2))/Tobs;
		//		double f2 = ((double)(q + i - M/2))/Tobs;
				//printf("%e, %e, %ld, %ld, %ld\n", f, f2 - f1, q, i, M/2);
		//}
		fonfs = f/fstar;
		//printf("Stas fonfs = %f, %f, %f, %f \n", fonfs, f, fstar, Tobs);
		c3 = cos(3.*fonfs);  c2 = cos(2.*fonfs);  c1 = cos(1.*fonfs);
		s3 = sin(3.*fonfs);  s2 = sin(2.*fonfs);  s1 = sin(1.*fonfs);

		X_1   = (a12[2*i]-a13[2*i])*c3 + (a12[2*i+1]-a13[2*i+1])*s3 +
		           (a21[2*i]-a31[2*i])*c2 + (a21[2*i+1]-a31[2*i+1])*s2 +
		           (a13[2*i]-a12[2*i])*c1 + (a13[2*i+1]-a12[2*i+1])*s1 +
		           (a31[2*i]-a21[2*i]);

		X_2 = (a12[2*i+1]-a13[2*i+1])*c3 - (a12[2*i]-a13[2*i])*s3 +
		           (a21[2*i+1]-a31[2*i+1])*c2 - (a21[2*i]-a31[2*i])*s2 +
		           (a13[2*i+1]-a12[2*i+1])*c1 - (a13[2*i]-a12[2*i])*s1 +
		           (a31[2*i+1]-a21[2*i+1]);

		Y_1   = (a23[2*i]-a21[2*i])*c3 + (a23[2*i+1]-a21[2*i+1])*s3 +
		           (a32[2*i]-a12[2*i])*c2 + (a32[2*i+1]-a12[2*i+1])*s2+
		           (a21[2*i]-a23[2*i])*c1 + (a21[2*i+1]-a23[2*i+1])*s1+
		           (a12[2*i]-a32[2*i]);

		Y_2 = (a23[2*i+1]-a21[2*i+1])*c3 - (a23[2*i]-a21[2*i])*s3+
		           (a32[2*i+1]-a12[2*i+1])*c2 - (a32[2*i]-a12[2*i])*s2+
		           (a21[2*i+1]-a23[2*i+1])*c1 - (a21[2*i]-a23[2*i])*s1+
		           (a12[2*i+1]-a32[2*i+1]);

		Z_1   = (a31[2*i]-a32[2*i])*c3 + (a31[2*i+1]-a32[2*i+1])*s3+
		           (a13[2*i]-a23[2*i])*c2 + (a13[2*i+1]-a23[2*i+1])*s2+
		           (a32[2*i]-a31[2*i])*c1 + (a32[2*i+1]-a31[2*i+1])*s1+
		           (a23[2*i]-a13[2*i]);

		Z_2 = (a31[2*i+1]-a32[2*i+1])*c3 - (a31[2*i]-a32[2*i])*s3+
		           (a13[2*i+1]-a23[2*i+1])*c2 - (a13[2*i]-a23[2*i])*s2+
		           (a32[2*i+1]-a31[2*i+1])*c1 - (a32[2*i]-a31[2*i])*s1+
		           (a23[2*i+1]-a13[2*i+1]);

		// XLS_r   =  (X_1*cLS - X_2*sLS);
		// XLS_i = -(X_1*sLS + X_2*cLS);
		// YLS_r   =  (Y_1*cLS - Y_2*sLS);
		// YLS_i = -(Y_1*sLS + Y_2*cLS);
		// ZLS_r   =  (Z_1*cLS - Z_2*sLS);
		// ZLS_i = -(Z_1*sLS + Z_2*cLS);
    //
		// XSL_r   =  2.0*fonfs*(X_1*cSL - X_2*sSL);
		// XSL_i = -2.0*fonfs*(X_1*sSL + X_2*cSL);
		// YSL_r   =  2.0*fonfs*(Y_1*cSL - Y_2*sSL);
		// YSL_i = -2.0*fonfs*(Y_1*sSL + Y_2*cSL);
		// ZSL_r   =  2.0*fonfs*(Z_1*cSL - Z_2*sSL);
		// ZSL_i = -2.0*fonfs*(Z_1*sSL + Z_2*cSL);

		// Alternative polarization definition
		*XLS_r   =  (X_1*cLS - X_2*sLS);
		*XLS_i =  (X_1*sLS + X_2*cLS);
		*YLS_r   =  (Y_1*cLS - Y_2*sLS);
		*YLS_i =  (Y_1*sLS + Y_2*cLS);
		*ZLS_r   =  (Z_1*cLS - Z_2*sLS);
		*ZLS_i =  (Z_1*sLS + Z_2*cLS);

		*XSL_r   =  2.0*fonfs*(X_1*cSL - X_2*sSL);
		*XSL_i =  2.0*fonfs*(X_1*sSL + X_2*cSL);
		*YSL_r   =  2.0*fonfs*(Y_1*cSL - Y_2*sSL);
		*YSL_i =  2.0*fonfs*(Y_1*sSL + Y_2*cSL);
		*ZSL_r   =  2.0*fonfs*(Z_1*cSL - Z_2*sSL);
		*ZSL_i =  2.0*fonfs*(Z_1*sSL + Z_2*cSL);

	// for(i=0; i<2*M; i++)
	// {
	// 	// A channel
	// 	ALS[i] = (2.0*XLS[i] - YLS[i] - ZLS[i])/3.0;
	// 	// E channel
	// 	ELS[i] = (ZLS[i]-YLS[i])/SQ3;
	// }


	//free(YLS);
	//free(ZLS);

	return;
}

__global__
void XYZ_wrap(Waveform *wfm_trans, int nwalkers, long M, double dt, double Tobs, double *XLS, double *YLS, double *ZLS){

		int N;
			long add_ind;
		double asd1, asd2, asd3;

		Waveform *wfm;
		for (int walker_i = blockIdx.y * blockDim.y + threadIdx.y;
				 walker_i < nwalkers;
				 walker_i += blockDim.y * gridDim.y){

		wfm = &wfm_trans[walker_i];
		N = wfm->N;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
				 i < M;
				 i += blockDim.x * gridDim.x)
		{


		double XLS_r, YLS_r, ZLS_r, XSL_r, YSL_r, ZSL_r, XLS_i, YLS_i, ZLS_i, XSL_i, YSL_i, ZSL_i;

		XYZ(i, wfm->a12, wfm->a21, wfm->a13, wfm->a31, wfm->a23, wfm->a32, wfm->params[0]/wfm->T, wfm->q, N, dt, Tobs,
				&XLS_r, &YLS_r, &ZLS_r, &XSL_r, &YSL_r, &ZSL_r, &XLS_i, &YLS_i, &ZLS_i, &XSL_i, &YSL_i, &ZSL_i);

		add_ind = (wfm->q + i - M/2);

		atomicAddDouble(&XLS[2*add_ind], XLS_r/asd1);
		atomicAddDouble(&XLS[2*add_ind+1], XLS_i/asd1);

		atomicAddDouble(&YLS[2*add_ind], YLS_r/asd2);
		atomicAddDouble(&YLS[2*add_ind+1], YLS_i/asd2);

		atomicAddDouble(&ZLS[2*add_ind], ZLS_r/asd3);
		atomicAddDouble(&ZLS[2*add_ind+1], ZLS_i/asd3);

		/*atomicAddDouble(&XSL[2*add_ind], XSL_r/asd1);
		atomicAddDouble(&XSL[2*add_ind+1], XSL_i)/asd1;

		atomicAddDouble(&YSL[2*add_ind], YSL_r/asd2);
		atomicAddDouble(&YSL[2*add_ind+1], YSL_i/asd2);

		atomicAddDouble(&YSL[2*add_ind], ZSL_r/asd3);
		atomicAddDouble(&ZSL[2*add_ind+1], ZSL_i/asd3);*/


}
}
}

/*
__global__
void prep_with_ASD(double *XLS, double *YLS, double *ZLS, double *channel1_ASDinv, double *channel2_ASDinv, double *channel3_ASDinv, int data_stream_length){
	double asd1, asd2, asd3;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			 i < data_stream_length;
			 i += blockDim.x * gridDim.x)
	{
			asd1 = channel1_ASDinv[i];
			asd2 = channel2_ASDinv[i];
			asd3 = channel3_ASDinv[i];

			XLS[2*i] = XLS[2*i]/asd1;
			XLS[2*i+1] = XLS[2*i+1]/asd1;

			YLS[2*i] = YLS[2*i]/asd2;
			YLS[2*i+1] = YLS[2*i+1]/asd2;

			ZLS[2*i] = ZLS[2*i]/asd3;
			ZLS[2*i+1] = ZLS[2*i+1]/asd3;
	}
}
*/
