/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  Creates the structures that hold waveform and interpolation information
 *  for the GPU version of the PhenomHM waveform.
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

#include <assert.h>
#include <iostream>
#include <complex>
#include "cuComplex.h"
#include "global.h"

/*
Function for creating ModeContainer on the gpu.
*/

void alloc_waveform(Waveform *wfm)
{
	long i, j, n, k;
	long N;

	N = wfm->N;

  gpuErrchk(cudaMalloc(&wfm->params, wfm->NP*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->k, 3*sizeof(double)));


  //gpuErrchk(cudaMalloc(&wfm->kdotx, 3*N*sizeof(double))); // vec
  //gpuErrchk(cudaMalloc(&wfm->kdotr, 3*3*N*sizeof(double))); // 3x3 mat


  //double *trans_kdotx = new double[3*N];
  //double *trans_kdotr = new double[3*3*N];

  /*for (k=0; k<N; k++){

	for (i=0; i<3; i++)
	{
		//for (j=0; j<3; j++) trans_kdotr[(i*3 + j)*N + k] = 0.;
		//trans_kdotx[i*N + k] = 0.;
	}
}*/

  //gpuErrchk(cudaMemcpy(wfm->kdotx, trans_kdotx, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->kdotr, trans_kdotr, 3*3*N*sizeof(double), cudaMemcpyHostToDevice));

  //delete[] trans_kdotr;
  //delete[] trans_kdotx;

  //gpuErrchk(cudaMalloc(&wfm->xi, 3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->f, 3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->fonfs, 3*N*sizeof(double)));

  //double *trans_xi = new double[3*N];
  //double *trans_f = new double[3*N];
  //double *trans_fonfs = new double[3*N];

  /*for (k=0; k<N; k++){
	for (i=0; i<3; i++)
	{
		//trans_xi[i*N + k]    = 0.;
		//trans_f[i*N + k]     = 0.;
		//trans_fonfs[i*N + k] = 0.;
	}
}*/

  //gpuErrchk(cudaMemcpy(wfm->xi, trans_xi, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->f, trans_f, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->fonfs, trans_fonfs, 3*N*sizeof(double), cudaMemcpyHostToDevice));

  //delete[] trans_xi;
  //delete[] trans_f;
  //delete[] trans_fonfs;

	// Polarization basis tensors
  gpuErrchk(cudaMalloc(&wfm->eplus, 3*3*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->ecross, 3*3*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->dplus, 3*3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->dcross, 3*3*N*sizeof(double)));

  /*
  gpuErrchk(cudaMalloc(&wfm->r12, 3*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->r21, 3*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->r31, 3*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->r13, 3*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->r23, 3*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->r32, 3*N*sizeof(double)));
  */

  /*
  gpuErrchk(cudaMalloc(&wfm->data12, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->data21, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->data31, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->data13, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->data23, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->data32, 2*N*sizeof(double)));
  */

  double *trans_data12 = new double[2*N];
  double *trans_data21 = new double[2*N];
  double *trans_data31 = new double[2*N];
  double *trans_data13 = new double[2*N];
  double *trans_data23 = new double[2*N];
  double *trans_data32 = new double[2*N];

	for (i=0; i<2*N; i++)
	{
		trans_data12[i] = 0.;
		trans_data21[i] = 0.;
		trans_data31[i] = 0.;
		trans_data13[i] = 0.;
		trans_data23[i] = 0.;
		trans_data32[i] = 0.;
	}

  /*
  gpuErrchk(cudaMemcpy(wfm->data12, trans_data12, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->data21, trans_data21, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->data31, trans_data31, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->data13, trans_data13, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->data23, trans_data23, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->data32, trans_data32, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  */

  delete[] trans_data12;
  delete[] trans_data21;
  delete[] trans_data31;
  delete[] trans_data13;
  delete[] trans_data23;
  delete[] trans_data32;


  gpuErrchk(cudaMalloc(&wfm->a12, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->a21, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->a31, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->a13, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->a23, 2*N*sizeof(double)));
  gpuErrchk(cudaMalloc(&wfm->a32, 2*N*sizeof(double)));

  double *trans_a12 = new double[2*N];
  double *trans_a21 = new double[2*N];
  double *trans_a31 = new double[2*N];
  double *trans_a13 = new double[2*N];
  double *trans_a23 = new double[2*N];
  double *trans_a32 = new double[2*N];

  for (i=0; i<2*N; i++)
  {
    trans_a12[i] = 0.;
    trans_a21[i] = 0.;
    trans_a31[i] = 0.;
    trans_a13[i] = 0.;
    trans_a23[i] = 0.;
    trans_a32[i] = 0.;
  }

  gpuErrchk(cudaMemcpy(wfm->a12, trans_a12, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->a21, trans_a21, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->a31, trans_a31, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->a13, trans_a13, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->a23, trans_a23, 2*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->a32, trans_a32, 2*N*sizeof(double), cudaMemcpyHostToDevice));

  delete[] trans_a12;
  delete[] trans_a21;
  delete[] trans_a31;
  delete[] trans_a13;
  delete[] trans_a23;
  delete[] trans_a32;

  //gpuErrchk(cudaMalloc(&wfm->TR, 3*3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->TI, 3*3*N*sizeof(double)));

  //gpuErrchk(cudaMalloc(&wfm->x, 3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->y, 3*N*sizeof(double)));
  //gpuErrchk(cudaMalloc(&wfm->z, 3*N*sizeof(double)));

  double *trans_eplus = new double[3*3];
  double *trans_ecross = new double[3*3];
  //double *trans_dplus = new double[3*3*N];
  //double *trans_dcross = new double[3*3*N];
  //double *trans_TR = new double[3*3*N];
  //double *trans_TI = new double[3*3*N];
  //double *trans_x = new double[3*N];
  //double *trans_y = new double[3*N];
  //double *trans_z = new double[3*N];
  /*
  double *trans_r12 = new double[3*N];
  double *trans_r21 = new double[3*N];
  double *trans_r31 = new double[3*N];
  double *trans_r13 = new double[3*N];
  double *trans_r23 = new double[3*N];
  double *trans_r32 = new double[3*N];
  */
  for (k=0; k<N; k++){
	for (i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			trans_eplus[(i*3 + j)]  = 0.;
			trans_ecross[(i*3 + j)] = 0.;
			//trans_dplus[(i*3 + j)*N + k]  = 0.;
			//trans_dcross[(i*3 + j)*N + k] = 0.;
			//trans_TR[(i*3 + j)*N + k]     = 0.;
			//trans_TI[(i*3 + j)*N + k]     = 0.;
		}
		//trans_x[i*N + k]   = 0.;
		//trans_y[i*N + k]   = 0.;
		//trans_z[i*N + k]   = 0.;
    /*
		trans_r12[i*N + k] = 0.;
		trans_r21[i*N + k] = 0.;
		trans_r31[i*N + k] = 0.;
		trans_r13[i*N + k] = 0.;
		trans_r23[i*N + k] = 0.;
		trans_r32[i*N + k] = 0.;
    */
	}
  }

  gpuErrchk(cudaMemcpy(wfm->eplus, trans_eplus, 3*3*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->ecross, trans_ecross, 3*3*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->dplus, trans_dplus, 3*3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->dcross, trans_dcross, 3*3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->TR, trans_TR, 3*3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->TI, trans_TI, 3*3*N*sizeof(double), cudaMemcpyHostToDevice));

  //gpuErrchk(cudaMemcpy(wfm->x, trans_x, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->y, trans_y, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(wfm->z, trans_z, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  /*
  gpuErrchk(cudaMemcpy(wfm->r12, trans_r12, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->r21, trans_r21, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->r31, trans_r31, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->r13, trans_r13, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->r23, trans_r23, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(wfm->r32, trans_r32, 3*N*sizeof(double), cudaMemcpyHostToDevice));
  */

  delete[] trans_eplus;
  delete[] trans_ecross;
  //delete[] trans_dplus;
  //delete[] trans_dcross;
  //delete[] trans_TR;
  //delete[] trans_TI;

  //delete[] trans_x;
  //delete[] trans_y;
  //delete[] trans_z;
  /*
  delete[] trans_r12;
  delete[] trans_r21;
  delete[] trans_r31;
  delete[] trans_r13;
  delete[] trans_r23;
  delete[] trans_r32;
  */
/*
  gpuErrchk(cudaMalloc(&wfm->d, 3*3*2*N*sizeof(double)));

  double *trans_d = new double[3*3*2*N];

	for (i=0; i<3; i++)
	{
		for(j=0; j<3; j++)
		{
			for (n=0; n<2*N; n++)
			{
				trans_d[i*3*2*N + j*2*N + n] = 0.;
			}
		}
	}

  gpuErrchk(cudaMemcpy(wfm->d, trans_d, 3*3*2*N*sizeof(double), cudaMemcpyHostToDevice));

  delete[] trans_d;
*/
	return;
}

void free_waveform(Waveform *wfm){

    gpuErrchk(cudaFree(wfm->params));

    gpuErrchk(cudaFree(wfm->k));

    gpuErrchk(cudaFree(wfm->eplus));
    gpuErrchk(cudaFree(wfm->ecross));
    //gpuErrchk(cudaFree(wfm->dplus));
    //gpuErrchk(cudaFree(wfm->dcross));


    /*
    gpuErrchk(cudaFree(wfm->r12));
    gpuErrchk(cudaFree(wfm->r21));
    gpuErrchk(cudaFree(wfm->r31));
    gpuErrchk(cudaFree(wfm->r13));
    gpuErrchk(cudaFree(wfm->r23));
    gpuErrchk(cudaFree(wfm->r32));
    */

    //gpuErrchk(cudaFree(wfm->kdotr));
    //gpuErrchk(cudaFree(wfm->kdotx));

    //gpuErrchk(cudaFree(wfm->xi));
    //gpuErrchk(cudaFree(wfm->f));
    //gpuErrchk(cudaFree(wfm->fonfs));

    /*
    gpuErrchk(cudaFree(wfm->data12));
    gpuErrchk(cudaFree(wfm->data21));
    gpuErrchk(cudaFree(wfm->data31));
    gpuErrchk(cudaFree(wfm->data13));
    gpuErrchk(cudaFree(wfm->data23));
    gpuErrchk(cudaFree(wfm->data32));
    */

    gpuErrchk(cudaFree(wfm->a12));
    gpuErrchk(cudaFree(wfm->a21));
    gpuErrchk(cudaFree(wfm->a31));
    gpuErrchk(cudaFree(wfm->a13));
    gpuErrchk(cudaFree(wfm->a23));
    gpuErrchk(cudaFree(wfm->a32));

    //gpuErrchk(cudaFree(wfm->x));
    //gpuErrchk(cudaFree(wfm->y));
    //gpuErrchk(cudaFree(wfm->z));

    //gpuErrchk(cudaFree(wfm->TR));
    //gpuErrchk(cudaFree(wfm->TI));

    //gpuErrchk(cudaFree(wfm->d));

}
