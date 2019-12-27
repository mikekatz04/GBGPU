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

	double * u = new double[3];
	double * v = new double[3];

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
	delete[] u;
	delete[] v;

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
	x[0*N + n] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[0*N + n] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[0*N + n] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta2);
	cb = cos(beta2);
	x[1*N + n] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[1*N + n] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[1*N + n] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta3);
	cb = cos(beta3);
	x[2*N + n] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[2*N + n] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[2*N + n] = -SQ3*AU*ec*(ca*cb + sa*sb);

	return;
}

__device__
void calc_xi_f(Waveform *wfm, double t, int n, int N)
{
	long i;

	double f0, dfdt_0, d2fdt2_0;

	f0       = wfm->params[0]/wfm->T;
	if (wfm->NP > 7) dfdt_0   = wfm->params[7]/wfm->T/wfm->T;
	if (wfm->NP > 8) d2fdt2_0 = wfm->params[8]/wfm->T/wfm->T/wfm->T;

	spacecraft(t, wfm->x, wfm->y, wfm->z, n, N); // Calculate position of each spacecraft at time t

	for(i=0; i<3; i++)
	{
		wfm->kdotx[i*N + n] = (wfm->x[i*N + n]*wfm->k[0*N + n] + wfm->y[i*N + n]*wfm->k[1*N + n] + wfm->z[i*N + n]*wfm->k[2*N + n])/C;
		//Wave arrival time at spacecraft i
		wfm->xi[i*N + n]    = t - wfm->kdotx[i*N + n];
		//FIXME
		//wfm->xi[i]    = t + wfm->kdotx[i];
		//First order approximation to frequency at spacecraft i
		wfm->f[i*N + n]     = f0;
		if (wfm->NP > 7) wfm->f[i*N + n] += dfdt_0*wfm->xi[i*N + n];
		if (wfm->NP > 8) wfm->f[i*N + n] += 0.5*d2fdt2_0*wfm->xi[i*N + n]*wfm->xi[i*N + n];

		//Ratio of true frequency to transfer frequency
		wfm->fonfs[i*N + n] = wfm->f[i*N + n]/fstar;
	}

	return;
}

__device__
void calc_sep_vecs(Waveform *wfm, int n, int N)
{
	long i;

	//Unit separation vector from spacecrafts i to j
	wfm->r12[0*N + n] = (wfm->x[1*N + n] - wfm->x[0*N + n])/Larm;
	wfm->r13[0*N + n] = (wfm->x[2*N + n] - wfm->x[0*N + n])/Larm;
	wfm->r23[0*N + n] = (wfm->x[2*N + n] - wfm->x[1*N + n])/Larm;
	wfm->r12[1*N + n] = (wfm->y[1*N + n] - wfm->y[0*N + n])/Larm;
	wfm->r13[1*N + n] = (wfm->y[2*N + n] - wfm->y[0*N + n])/Larm;
	wfm->r23[1*N + n] = (wfm->y[2*N + n] - wfm->y[1*N + n])/Larm;
	wfm->r12[2*N + n] = (wfm->z[1*N + n] - wfm->z[0*N + n])/Larm;
	wfm->r13[2*N + n] = (wfm->z[2*N + n] - wfm->z[0*N + n])/Larm;
	wfm->r23[2*N + n] = (wfm->z[2*N + n] - wfm->z[1*N + n])/Larm;

	//Make use of symmetry
	for(i=0; i<3; i++)
	{
		wfm->r21[i*N + n] = -wfm->r12[i*N + n];
		wfm->r31[i*N + n] = -wfm->r13[i*N + n];
		wfm->r32[i*N + n] = -wfm->r23[i*N + n];
	}
	return;
}

__device__
void calc_d_matrices(Waveform *wfm, int n, int N)
{
	long i, j;

	//Zero arrays to be summed
	wfm->dplus [(0*3 + 1)*N + n] = 0.0;
	wfm->dplus [(0*3 + 2)*N + n] = 0.0;
	wfm->dplus [(1*3 + 0)*N + n] = 0.;
	wfm->dplus [(1*3 + 2)*N + n] = 0.0;
	wfm->dplus [(2*3 + 0)*N + n] = 0.0;
	wfm->dplus [(2*3 + 1)*N + n] = 0.;
	wfm->dcross[(0*3 + 1)*N + n] = 0.0;
	wfm->dcross[(0*3 + 2)*N + n] = 0.0;
	wfm->dcross[(1*3 + 0)*N + n] = 0.;
	wfm->dcross[(1*3 + 2)*N + n] = 0.0;
	wfm->dcross[(2*3 + 0)*N + n] = 0.0;
	wfm->dcross[(2*3 + 1)*N + n] = 0.;

	//Convenient quantities d+ & dx
	for(i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			wfm->dplus [(0*3 + 1)*N + n] += wfm->r12[i*N + n]*wfm->r12[j*N + n]*wfm->eplus[i*3 + j];
			wfm->dcross[(0*3 + 1)*N + n] += wfm->r12[i*N + n]*wfm->r12[j*N + n]*wfm->ecross[i*3 + j];
			wfm->dplus [(1*3 + 2)*N + n] += wfm->r23[i*N + n]*wfm->r23[j*N + n]*wfm->eplus[i*3 + j];
			wfm->dcross[(1*3 + 2)*N + n] += wfm->r23[i*N + n]*wfm->r23[j*N + n]*wfm->ecross[i*3 + j];
			wfm->dplus [(0*3 + 2)*N + n] += wfm->r13[i*N + n]*wfm->r13[j*N + n]*wfm->eplus[i*3 + j];
			wfm->dcross[(0*3 + 2)*N + n] += wfm->r13[i*N + n]*wfm->r13[j*N + n]*wfm->ecross[i*3 + j];
		}
	}
	//Makng use of symmetry
	wfm->dplus[(1*3 + 0)*N + n] = wfm->dplus[(0*3 + 1)*N + n];  wfm->dcross[(1*3 + 0)*N + n] = wfm->dcross[(0*3 + 1)*N + n];
	wfm->dplus[(2*3 + 1)*N + n] = wfm->dplus[(1*3 + 2)*N + n];  wfm->dcross[(2*3 + 1)*N + n] = wfm->dcross[(1*3 + 2)*N + n];
	wfm->dplus[(2*3 + 0)*N + n] = wfm->dplus[(0*3 + 2)*N + n];  wfm->dcross[(2*3 + 0)*N + n] = wfm->dcross[(0*3 + 2)*N + n];

	return;
}


__device__
void calc_kdotr(Waveform *wfm, int n, int N)
{
	long i;

	//Zero arrays to be summed
	wfm->kdotr[(0*3 + 1)*N + n] = wfm->kdotr[(0*3 + 2)*N + n] = wfm->kdotr[(1*3 + 0)*N + n] = 0.;
	wfm->kdotr[(1*3 + 2)*N + n] = wfm->kdotr[(2*3 + 0)*N + n] = wfm->kdotr[(2*3 + 1)*N + n] = 0.;

	for(i=0; i<3; i++)
	{
		wfm->kdotr[(0*3 + 1)*N + n] += wfm->k[i*N + n]*wfm->r12[i*N + n];
		wfm->kdotr[(0*3 + 2)*N + n] += wfm->k[i*N + n]*wfm->r13[i*N + n];
		wfm->kdotr[(1*3 + 2)*N + n] += wfm->k[i*N + n]*wfm->r23[i*N + n];
	}

	//Making use of antisymmetry
	wfm->kdotr[(1*3 + 0)*N + n] = -wfm->kdotr[(0*3 + 1)*N + n];
	wfm->kdotr[(2*3 + 0)*N + n] = -wfm->kdotr[(0*3 + 2)*N + n];
	wfm->kdotr[(2*3 + 1)*N + n] = -wfm->kdotr[(1*3 + 2)*N + n];

	return;
}


__device__

void get_transfer(Waveform *wfm, double t, int n, int N)
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
				//arg1 = 0.5*wfm->fonfs[i]*(1. - wfm->kdotr[i][j]);
				arg1 = 0.5*wfm->fonfs[i*N + n]*(1. + wfm->kdotr[(i*3 + j)*N + n]);

				//Argument of complex exponentials
				arg2 = PI2*f0*wfm->xi[i*N + n] + phi0 - df*t;

				if (wfm->NP > 7) arg2 += M_PI*dfdt_0*wfm->xi[i*N + n]*wfm->xi[i*N + n];
				if (wfm->NP > 8) arg2 += M_PI*d2fdt2_0*wfm->xi[i*N + n]*wfm->xi[i*N + n]*wfm->xi[i*N + n]/3.0 ;

				//Transfer function
				sinc = 0.25*sin(arg1)/arg1;

				//Evolution of amplitude
				aevol = 1.0;
				if (wfm->NP > 7) aevol += 0.66666666666666666666*dfdt_0/f0*wfm->xi[i*N + n];

				///Real and imaginary pieces of time series (no complex exponential)
				tran1r = aevol*(wfm->dplus[(i*3 + j)*N + n]*wfm->DPr + wfm->dcross[(i*3 + j)*N + n]*wfm->DCr);
				tran1i = aevol*(wfm->dplus[(i*3 + j)*N + n]*wfm->DPi + wfm->dcross[(i*3 + j)*N + n]*wfm->DCi);

				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);

				//Real & Imaginary part of the slowly evolving signal
				wfm->TR[(i*3 + j)*N + n] = sinc*(tran1r*tran2r - tran1i*tran2i);
				wfm->TI[(i*3 + j)*N + n] = sinc*(tran1r*tran2i + tran1i*tran2r);
			}
		}
	}

	return;
}


__device__

void fill_time_series(Waveform *wfm, int n, int N)
{
	wfm->data12[2*n]   = wfm->TR[(0*3 + 1)*N + n];
	wfm->data21[2*n]   = wfm->TR[(1*3 + 0)*N + n];
	wfm->data31[2*n]   = wfm->TR[(2*3 + 0)*N + n];
	wfm->data12[2*n+1] = wfm->TI[(0*3 + 1)*N + n];
	wfm->data21[2*n+1] = wfm->TI[(1*3 + 0)*N + n];
	wfm->data31[2*n+1] = wfm->TI[(2*3 + 0)*N + n];
	wfm->data13[2*n]   = wfm->TR[(0*3 + 2)*N + n];
	wfm->data23[2*n]   = wfm->TR[(1*3 + 2)*N + n];
	wfm->data32[2*n]   = wfm->TR[(2*3 + 1)*N + n];
	wfm->data13[2*n+1] = wfm->TI[(0*3 + 2)*N + n];
	wfm->data23[2*n+1] = wfm->TI[(1*3 + 2)*N + n];
	wfm->data32[2*n+1] = wfm->TI[(2*3 + 1)*N + n];

	return;
}


__global__
void GenWave(Waveform *wfm_trans, int N, int nwalkers){
	double t=0.0;
	Waveform *wfm;
	for (int walker_i = blockIdx.y * blockDim.y + threadIdx.y;
			 walker_i < nwalkers;
			 walker_i += blockDim.y * gridDim.y){

	wfm = &wfm_trans[walker_i];

	for (int n = blockIdx.x * blockDim.x + threadIdx.x;
			 n < N;
			 n += blockDim.x * gridDim.x){

				 t = wfm->T*(double)(n)/(double)N;
				 calc_xi_f(wfm ,t, n, N);		  // calc frequency and time variables
				 calc_sep_vecs(wfm, n, N);       // calculate the S/C separation vectors
				 calc_d_matrices(wfm, n, N);     // calculate pieces of waveform
				 calc_kdotr(wfm, n, N);		  // calculate dot product
				 get_transfer(wfm, t, n, N);     // Calculating Transfer function
				 fill_time_series(wfm, n, N); // Fill  time series data arrays with slowly evolving signal.
		}
}
}

void fft_data(Waveform *wfm_trans, cufftHandle plan, int nwalkers)
{

	Waveform *wfm;

	for (int walker_i=0; walker_i<nwalkers; walker_i++){

		wfm = &wfm_trans[walker_i];
		long N = wfm->N;
	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data12, (cufftDoubleComplex*)wfm->data12, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data21, (cufftDoubleComplex*)wfm->data21, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data31, (cufftDoubleComplex*)wfm->data31, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data13, (cufftDoubleComplex*)wfm->data13, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data23, (cufftDoubleComplex*)wfm->data23, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	if (cufftExecZ2Z(plan, (cufftDoubleComplex*)wfm->data32, (cufftDoubleComplex*)wfm->data32, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	}
	return;
}

__global__
void unpack_data_1(Waveform *wfm_trans, int nwalkers)
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
		wfm->a12[i]   = wfm->data12[N+i]/(double)N;
		wfm->a21[i]   = wfm->data21[N+i]/(double)N;
		wfm->a31[i]   = wfm->data31[N+i]/(double)N;
		wfm->a12[i+N] = wfm->data12[i]/(double)N;
		wfm->a21[i+N] = wfm->data21[i]/(double)N;
		wfm->a31[i+N] = wfm->data31[i]/(double)N;
		wfm->a13[i]   = wfm->data13[N+i]/(double)N;
		wfm->a23[i]   = wfm->data23[N+i]/(double)N;
		wfm->a32[i]   = wfm->data32[N+i]/(double)N;
		wfm->a13[i+N] = wfm->data13[i]/(double)N;
		wfm->a23[i+N] = wfm->data23[i]/(double)N;
		wfm->a32[i+N] = wfm->data32[i]/(double)N;
	}
}
}

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
		wfm->d[0 *3*2*N + 1 *2*N + i] = 0.5*wfm->a12[i];
		wfm->d[1 *3*2*N + 0 *2*N + i] = 0.5*wfm->a21[i];
		wfm->d[2 *3*2*N + 0 *2*N + i] = 0.5*wfm->a31[i];
		wfm->d[0 *3*2*N + 2 *2*N + i] = 0.5*wfm->a13[i];
		wfm->d[1 *3*2*N + 2 *2*N + i] = 0.5*wfm->a23[i];
		wfm->d[2 *3*2*N + 1 *2*N + i] = 0.5*wfm->a32[i];
	}
}
}

__device__
void XYZ(int i, double *d, double f0, long q, long M, double dt, double Tobs, double *XLS, double *YLS, double *ZLS,
					double* XSL, double* YSL, double* ZSL, double *X, double *Y, double *Z)
{
	double fonfs;
	double c3, s3, c2, s2, c1, s1;
	double f;
	double phiLS, cLS, sLS, phiSL, cSL, sSL;

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
		fonfs = f/fstar;
		//printf("Stas fonfs = %f, %f, %f, %f \n", fonfs, f, fstar, Tobs);
		c3 = cos(3.*fonfs);  c2 = cos(2.*fonfs);  c1 = cos(1.*fonfs);
		s3 = sin(3.*fonfs);  s2 = sin(2.*fonfs);  s1 = sin(1.*fonfs);

		X[2*i]   = (d[0*3*2*M + 1*2*M + 2*i]-d[0*3*2*M + 2*2*M + 2*i])*c3 + (d[0*3*2*M + 1*2*M + 2*i+1]-d[0*3*2*M + 2*2*M + 2*i+1])*s3 +
		           (d[1*3*2*M + 0*2*M + 2*i]-d[2*3*2*M + 0*2*M + 2*i])*c2 + (d[1*3*2*M + 0*2*M + 2*i+1]-d[2*3*2*M + 0*2*M + 2*i+1])*s2 +
		           (d[0*3*2*M + 2*2*M + 2*i]-d[0*3*2*M + 1*2*M + 2*i])*c1 + (d[0*3*2*M + 2*2*M + 2*i+1]-d[0*3*2*M + 1*2*M + 2*i+1])*s1 +
		           (d[2*3*2*M + 0*2*M + 2*i]-d[1*3*2*M + 0*2*M + 2*i]);

		X[2*i+1] = (d[0*3*2*M + 1*2*M + 2*i+1]-d[0*3*2*M + 2*2*M + 2*i+1])*c3 - (d[0*3*2*M + 1*2*M + 2*i]-d[0*3*2*M + 2*2*M + 2*i])*s3 +
		           (d[1*3*2*M + 0*2*M + 2*i+1]-d[2*3*2*M + 0*2*M + 2*i+1])*c2 - (d[1*3*2*M + 0*2*M + 2*i]-d[2*3*2*M + 0*2*M + 2*i])*s2 +
		           (d[0*3*2*M + 2*2*M + 2*i+1]-d[0*3*2*M + 1*2*M + 2*i+1])*c1 - (d[0*3*2*M + 2*2*M + 2*i]-d[0*3*2*M + 1*2*M + 2*i])*s1 +
		           (d[2*3*2*M + 0*2*M + 2*i+1]-d[1*3*2*M + 0*2*M + 2*i+1]);

		Y[2*i]   = (d[1*3*2*M + 2*2*M + 2*i]-d[1*3*2*M + 0*2*M + 2*i])*c3 + (d[1*3*2*M + 2*2*M + 2*i+1]-d[1*3*2*M + 0*2*M + 2*i+1])*s3 +
		           (d[2*3*2*M + 1*2*M + 2*i]-d[0*3*2*M + 1*2*M + 2*i])*c2 + (d[2*3*2*M + 1*2*M + 2*i+1]-d[0*3*2*M + 1*2*M + 2*i+1])*s2+
		           (d[1*3*2*M + 0*2*M + 2*i]-d[1*3*2*M + 2*2*M + 2*i])*c1 + (d[1*3*2*M + 0*2*M + 2*i+1]-d[1*3*2*M + 2*2*M + 2*i+1])*s1+
		           (d[0*3*2*M + 1*2*M + 2*i]-d[2*3*2*M + 1*2*M + 2*i]);

		Y[2*i+1] = (d[1*3*2*M + 2*2*M + 2*i+1]-d[1*3*2*M + 0*2*M + 2*i+1])*c3 - (d[1*3*2*M + 2*2*M + 2*i]-d[1*3*2*M + 0*2*M + 2*i])*s3+
		           (d[2*3*2*M + 1*2*M + 2*i+1]-d[0*3*2*M + 1*2*M + 2*i+1])*c2 - (d[2*3*2*M + 1*2*M + 2*i]-d[0*3*2*M + 1*2*M + 2*i])*s2+
		           (d[1*3*2*M + 0*2*M + 2*i+1]-d[1*3*2*M + 2*2*M + 2*i+1])*c1 - (d[1*3*2*M + 0*2*M + 2*i]-d[1*3*2*M + 2*2*M + 2*i])*s1+
		           (d[0*3*2*M + 1*2*M + 2*i+1]-d[2*3*2*M + 1*2*M + 2*i+1]);

		Z[2*i]   = (d[2*3*2*M + 0*2*M + 2*i]-d[2*3*2*M + 1*2*M + 2*i])*c3 + (d[2*3*2*M + 0*2*M + 2*i+1]-d[2*3*2*M + 1*2*M + 2*i+1])*s3+
		           (d[0*3*2*M + 2*2*M + 2*i]-d[1*3*2*M + 2*2*M + 2*i])*c2 + (d[0*3*2*M + 2*2*M + 2*i+1]-d[1*3*2*M + 2*2*M + 2*i+1])*s2+
		           (d[2*3*2*M + 1*2*M + 2*i]-d[2*3*2*M + 0*2*M + 2*i])*c1 + (d[2*3*2*M + 1*2*M + 2*i+1]-d[2*3*2*M + 0*2*M + 2*i+1])*s1+
		           (d[1*3*2*M + 2*2*M + 2*i]-d[0*3*2*M + 2*2*M + 2*i]);

		Z[2*i+1] = (d[2*3*2*M + 0*2*M + 2*i+1]-d[2*3*2*M + 1*2*M + 2*i+1])*c3 - (d[2*3*2*M + 0*2*M + 2*i]-d[2*3*2*M + 1*2*M + 2*i])*s3+
		           (d[0*3*2*M + 2*2*M + 2*i+1]-d[1*3*2*M + 2*2*M + 2*i+1])*c2 - (d[0*3*2*M + 2*2*M + 2*i]-d[1*3*2*M + 2*2*M + 2*i])*s2+
		           (d[2*3*2*M + 1*2*M + 2*i+1]-d[2*3*2*M + 0*2*M + 2*i+1])*c1 - (d[2*3*2*M + 1*2*M + 2*i]-d[2*3*2*M + 0*2*M + 2*i])*s1+
		           (d[1*3*2*M + 2*2*M + 2*i+1]-d[0*3*2*M + 2*2*M + 2*i+1]);

		// XLS[2*i]   =  (X[2*i]*cLS - X[2*i+1]*sLS);
		// XLS[2*i+1] = -(X[2*i]*sLS + X[2*i+1]*cLS);
		// YLS[2*i]   =  (Y[2*i]*cLS - Y[2*i+1]*sLS);
		// YLS[2*i+1] = -(Y[2*i]*sLS + Y[2*i+1]*cLS);
		// ZLS[2*i]   =  (Z[2*i]*cLS - Z[2*i+1]*sLS);
		// ZLS[2*i+1] = -(Z[2*i]*sLS + Z[2*i+1]*cLS);
    //
		// XSL[2*i]   =  2.0*fonfs*(X[2*i]*cSL - X[2*i+1]*sSL);
		// XSL[2*i+1] = -2.0*fonfs*(X[2*i]*sSL + X[2*i+1]*cSL);
		// YSL[2*i]   =  2.0*fonfs*(Y[2*i]*cSL - Y[2*i+1]*sSL);
		// YSL[2*i+1] = -2.0*fonfs*(Y[2*i]*sSL + Y[2*i+1]*cSL);
		// ZSL[2*i]   =  2.0*fonfs*(Z[2*i]*cSL - Z[2*i+1]*sSL);
		// ZSL[2*i+1] = -2.0*fonfs*(Z[2*i]*sSL + Z[2*i+1]*cSL);

		// Alternative polarization definition
		XLS[2*i]   =  (X[2*i]*cLS - X[2*i+1]*sLS);
		XLS[2*i+1] =  (X[2*i]*sLS + X[2*i+1]*cLS);
		YLS[2*i]   =  (Y[2*i]*cLS - Y[2*i+1]*sLS);
		YLS[2*i+1] =  (Y[2*i]*sLS + Y[2*i+1]*cLS);
		ZLS[2*i]   =  (Z[2*i]*cLS - Z[2*i+1]*sLS);
		ZLS[2*i+1] =  (Z[2*i]*sLS + Z[2*i+1]*cLS);

		XSL[2*i]   =  2.0*fonfs*(X[2*i]*cSL - X[2*i+1]*sSL);
		XSL[2*i+1] =  2.0*fonfs*(X[2*i]*sSL + X[2*i+1]*cSL);
		YSL[2*i]   =  2.0*fonfs*(Y[2*i]*cSL - Y[2*i+1]*sSL);
		YSL[2*i+1] =  2.0*fonfs*(Y[2*i]*sSL + Y[2*i+1]*cSL);
		ZSL[2*i]   =  2.0*fonfs*(Z[2*i]*cSL - Z[2*i+1]*sSL);
		ZSL[2*i+1] =  2.0*fonfs*(Z[2*i]*sSL + Z[2*i+1]*cSL);

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
void XYZ_wrap(Waveform *wfm_trans, int nwalkers, long M, double dt, double Tobs, double *XLS, double *YLS, double *ZLS,
					double* XSL, double* YSL, double* ZSL, double *X, double *Y, double *Z){

		int N;
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

		XYZ(i, wfm->d, wfm->params[0]/wfm->T, wfm->q, N, dt, Tobs,
				&XLS[M*walker_i], &YLS[M*walker_i], &ZLS[M*walker_i],
				&XSL[M*walker_i], &YSL[M*walker_i], &ZSL[M*walker_i],
				&X[M*walker_i], &Y[M*walker_i], &Z[M*walker_i]);
}
}
}
