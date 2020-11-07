
#include "new_fastGB.hh"
#include "global.h"


__device__
void set_const_trans(double* DPr, double* DPi, double* DCr, double* DCi, double amp, double cosiota, double psi, int bin_i)
{
	double Aplus, Across;
	double sinps, cosps;

	//Calculate GW polarization amplitudes
	Aplus  = amp*(1. + cosiota*cosiota);
	// Aplus  = -amp*(1. + cosiota*cosiota);
	Across = -2.0*amp*cosiota;
	//Across = 2.0*amp*cosiota;

	//Calculate cos and sin of polarization
	cosps = cos(2.*psi);
	sinps = sin(2.*psi);

	//Calculate constant pieces of transfer functions
	DPr[bin_i]    =  Aplus*cosps;
	DPi[bin_i]    = -Across*sinps;
	DCr[bin_i]    = -Aplus*sinps;
	DCi[bin_i]    = -Across*cosps;
}



__device__
void get_basis_vecs(double *k, double *u, double *v, double phi, double theta, int bin_i, int num_bin)
{
	double costh, sinth, cosph, sinph;

    costh = cos(theta);
	sinth = sqrt(1.0-costh*costh);

	cosph = cos(phi);
	sinph = sin(phi);

	u[0] =  costh*cosph;  u[1] =  costh*sinph;  u[2] = -sinth;
	v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
	k[0 * num_bin + bin_i] = -sinth*cosph;  k[1 * num_bin + bin_i] = -sinth*sinph;  k[2 * num_bin + bin_i] = -costh;
}

#define  NUM_THREADS 256

__global__
void get_basis_tensors(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                       double* amp, double* cosiota, double* psi, double* lam, double* beta, int num_bin)
{
	 // GW basis vectors

    __shared__ double u_all[3 * NUM_THREADS];
	__shared__ double v_all[3 * NUM_THREADS];

    double* u = &u_all[3 * threadIdx.x];
    double* v = &v_all[3 * threadIdx.x];

    int start, end, increment;
    #ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

    #else

    start = 0;
    end = num_bin;
    increment = 1;

    #pragma omp parallel for
    #endif
	for (int bin_i = start; bin_i < num_bin; bin_i += increment)
    {

    	set_const_trans(DPr, DPi, DCr, DCi, amp[bin_i], cosiota[bin_i], psi[bin_i], bin_i);  // set the constant pieces of transfer function

        // TODO: beta vs theta?
    	get_basis_vecs(k, u, v, lam[bin_i], beta[bin_i], bin_i, num_bin); //Gravitational Wave source basis vectors

    	//GW polarization basis tensors
    	for(int i = 0; i < 3; i++)
    	{
    		for(int j = 0; j < 3; j++)
    		{
    			//wfm->eplus[i][j]  = u[i]*u[j] - v[i]*v[j];
    			eplus[(i*3 + j) * num_bin + bin_i]  = v[i]*v[j] - u[i]*u[j];
    			ecross[(i*3 + j) * num_bin + bin_i] = u[i]*v[j] + v[i]*u[j];
    			//wfm->ecross[i][j] = -u[i]*v[j] - v[i]*u[j];
    		}
    	}
    }
}

void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, int num_bin)
{
    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    get_basis_tensors<<<num_blocks, NUM_THREADS>>>(
        eplus, ecross, DPr, DPi, DCr, DCi, k,
        amp, cosiota, psi, lam, beta, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}
