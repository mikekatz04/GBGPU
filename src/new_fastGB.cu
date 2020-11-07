
#include "new_fastGB.hh"
#include "global.h"
#include "LISA.h"


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

// TODO: check if this can be upped by reusing shared memory
#define  NUM_THREADS 32

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
	for (int bin_i = start; bin_i < end; bin_i += increment)
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



__device__
void spacecraft(double t, double* x, double* y, double* z, int n, int N)
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
}

__device__
void calc_xi_f(double* x, double* y, double* z, double* k, double* xi, double* fonfs,
               double f0, double dfdt, double d2fdt2, double T, double t, int n, int N)
{
	double f0_0, dfdt_0, d2fdt2_0;

	double kdotx_temp, f_temp, xi_temp;

	f0_0       = f0/T;
	dfdt_0   = dfdt/T/T;
	d2fdt2_0 = d2fdt2/T/T/T;

	spacecraft(t, x, y, z, n, N); // Calculate position of each spacecraft at time t

	for(int i = 0; i < 3; i++)
	{
		kdotx_temp = (x[i] * k[0] + y[i] * k[1] + z[i] * k[2])/C;
		//Wave arrival time at spacecraft i
		xi_temp    = t - kdotx_temp;
        xi[i] = xi_temp;
		//FIXME
		//xi[i]    = t + kdotx[i];
		//First order approximation to frequency at spacecraft i
		f_temp     = f0_0 + dfdt_0 * xi_temp + 0.5 * d2fdt2_0 * xi_temp * xi_temp;

		//Ratio of true frequency to transfer frequency
		fonfs[i] = f_temp/fstar;
	}
}


__device__
void calc_sep_vecs(double *r12, double *r21, double *r13, double *r31, double *r23, double *r32,
                   double *x, double *y, double *z,
                   int n, int N)
{
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
	for(int i = 0; i < 3; i++)
	{
		r21[i] = -r12[i];
		r31[i] = -r13[i];
		r32[i] = -r23[i];

	}
}


__device__
void calc_kdotr(double* k, double *kdotr, double *r12, double *r21, double *r13, double *r31, double *r23, double *r32)
{

	//Zero arrays to be summed
	kdotr[(0*3 + 1)] = 0.0;
	kdotr[(0*3 + 2)] = 0.0;
	kdotr[(1*3 + 0)] = 0.;
	kdotr[(1*3 + 2)] = 0.0;
	kdotr[(2*3 + 0)] = 0.0;
	kdotr[(2*3 + 1)] = 0.;

	for(int i = 0; i < 3; i++)
	{
		kdotr[(0*3 + 1)] += k[i]*r12[i];
		kdotr[(0*3 + 2)] += k[i]*r13[i];
		kdotr[(1*3 + 2)] += k[i]*r23[i];
	}

	//Making use of antisymmetry
	kdotr[(1*3 + 0)] = -kdotr[(0*3 + 1)];
	kdotr[(2*3 + 0)] = -kdotr[(0*3 + 2)];
	kdotr[(2*3 + 1)] = -kdotr[(1*3 + 2)];

	return;
}

__device__
void calc_d_matrices(double *dplus, double *dcross, double* eplus, double* ecross,
                     double *r12, double *r21, double *r13, double *r31, double *r23, double *r32)
{
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
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			dplus [(0*3 + 1)] += r12[i]*r12[j]*eplus[i*3 + j];
			dcross[(0*3 + 1)] += r12[i]*r12[j]*ecross[i*3 + j];
			dplus [(1*3 + 2)] += r23[i]*r23[j]*eplus[i*3 + j];
			dcross[(1*3 + 2)] += r23[i]*r23[j]*ecross[i*3 + j];
			dplus [(0*3 + 2)] += r13[i]*r13[j]*eplus[i*3 + j];
			dcross[(0*3 + 2)] += r13[i]*r13[j]*ecross[i*3 + j];
		}
	}
	//Makng use of symmetry
	dplus[(1*3 + 0)] = dplus[(0*3 + 1)];  dcross[(1*3 + 0)] = dcross[(0*3 + 1)];
	dplus[(2*3 + 1)] = dplus[(1*3 + 2)];  dcross[(2*3 + 1)] = dcross[(1*3 + 2)];
	dplus[(2*3 + 0)] = dplus[(0*3 + 2)];  dcross[(2*3 + 0)] = dcross[(0*3 + 2)];

	return;
}


__device__
void get_transfer(long q, double f0, double dfdt, double d2fdt2, double phi0,
                 double T, double t, int n, int N,
                 double *kdotr, double *TR, double *TI,
                 double *dplus, double *dcross,
				 double *xi, double *fonfs,
                 double DPr, double DPi, double DCr, double DCi)
{
	double tran1r, tran1i;
	double tran2r, tran2i;
	double aevol;			// amplitude evolution factor
	double arg1, arg2, sinc;
	double f0_0, dfdt_0, d2fdt2_0;
	double df;

	f0       = f0/T;
	dfdt_0   = dfdt/T/T;
    d2fdt2_0 = d2fdt2/T/T/T;

	df = PI2*(((double)q)/T);

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			if(i!=j)
			{
				//Argument of transfer function
				// FIXME
				//arg1 = 0.5*fonfs[i]*(1. - kdotr[i][j]);
				arg1 = 0.5*fonfs[i]*(1. + kdotr[(i*3 + j)]);

				//Argument of complex exponentials
				arg2 = PI2*f0*xi[i] + phi0 - df*t + M_PI*dfdt_0*xi[i]*xi[i] + M_PI*d2fdt2_0*xi[i]*xi[i]*xi[i]/3.0 ;

				//Transfer function
				sinc = 0.25*sin(arg1)/arg1;

				//Evolution of amplitude
				aevol = 1.0 + 0.66666666666666666666*dfdt_0/f0*xi[i];

				///Real and imaginary pieces of time series (no complex exponential)
				tran1r = aevol*(dplus[(i*3 + j)]*DPr + dcross[(i*3 + j)]*DCr);
				tran1i = aevol*(dplus[(i*3 + j)]*DPi + dcross[(i*3 + j)]*DCi);

				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);

				//Real & Imaginary part of the slowly evolving signal
				TR[(i*3 + j)] = sinc*(tran1r*tran2r - tran1i*tran2i);
				TI[(i*3 + j)] = sinc*(tran1r*tran2i + tran1i*tran2r);
			}
		}
	}
}



__device__
void fill_time_series(int bin_i, int num_bin, int n, int N, double *TR, double *TI,
					  double *data12, double *data21, double *data13,
					  double *data31, double *data23, double *data32)
{
	data12[2*n * num_bin + bin_i]   = TR[(0*3 + 1)];
	data21[2*n * num_bin + bin_i]   = TR[(1*3 + 0)];
	data31[2*n * num_bin + bin_i]   = TR[(2*3 + 0)];
	data12[(2*n + 1) * num_bin + bin_i] = TI[(0*3 + 1)];
	data21[(2*n + 1) * num_bin + bin_i] = TI[(1*3 + 0)];
	data31[(2*n + 1) * num_bin + bin_i] = TI[(2*3 + 0)];
	data13[2*n * num_bin + bin_i]   = TR[(0*3 + 2)];
	data23[2*n * num_bin + bin_i]   = TR[(1*3 + 2)];
	data32[2*n * num_bin + bin_i]   = TR[(2*3 + 1)];
	data13[(2*n + 1) * num_bin + bin_i] = TI[(0*3 + 2)];
	data23[(2*n + 1) * num_bin + bin_i] = TI[(1*3 + 2)];
	data32[(2*n + 1) * num_bin + bin_i] = TI[(2*3 + 1)];

	return;
}



__global__
void GenWave(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_in, double T, int N, int num_bin)
{

    __shared__ double x_all[3 * NUM_THREADS];
    __shared__ double y_all[3 * NUM_THREADS];
    __shared__ double z_all[3 * NUM_THREADS];
    __shared__ double k_all[3 * NUM_THREADS];
    __shared__ double xi_all[3 * NUM_THREADS];
    __shared__ double fonfs_all[3 * NUM_THREADS];
    __shared__ double r12_all[3 * NUM_THREADS];
    __shared__ double r21_all[3 * NUM_THREADS];
    __shared__ double r13_all[3 * NUM_THREADS];
    __shared__ double r31_all[3 * NUM_THREADS];
    __shared__ double r23_all[3 * NUM_THREADS];
    __shared__ double r32_all[3 * NUM_THREADS];
    __shared__ double dplus_all[9 * NUM_THREADS];
    __shared__ double dcross_all[9 * NUM_THREADS];
    __shared__ double eplus_all[9 * NUM_THREADS];
    __shared__ double ecross_all[9 * NUM_THREADS];
    __shared__ double kdotr_all[9 * NUM_THREADS];
    __shared__ double TR_all[9 * NUM_THREADS];
    __shared__ double TI_all[9 * NUM_THREADS];


    double* x = &x_all[3 * threadIdx.x];
    double* y = &y_all[3 * threadIdx.x];
    double* z = &z_all[3 * threadIdx.x];
    double* k = &k_all[3 * threadIdx.x];
    double* xi= &xi_all[3 * threadIdx.x];
    double* fonfs = &fonfs_all[3 * threadIdx.x];
    double* r12 = &r12_all[3 * threadIdx.x];
    double* r21 = &r21_all[3 * threadIdx.x];
    double* r13 = &r13_all[3 * threadIdx.x];
    double* r31 = &r31_all[3 * threadIdx.x];
    double* r23 = &r23_all[3 * threadIdx.x];
    double* r32 = &r32_all[3 * threadIdx.x];
    double* dplus = &dplus_all[9 * threadIdx.x];
    double* dcross = &dcross_all[9 * threadIdx.x];
    double* eplus = &eplus_all[9 * threadIdx.x];
    double* ecross = &ecross_all[9 * threadIdx.x];
    double* kdotr = &kdotr_all[9 * threadIdx.x];
    double* TR = &TR_all[9 * threadIdx.x];
    double* TI = &TI_all[9 * threadIdx.x];

    int start, end, increment;
    double t;

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
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {

        k[0] = k_in[0 * num_bin + bin_i];
        k[1] = k_in[1 * num_bin + bin_i];
        k[2] = k_in[2 * num_bin + bin_i];

        double f0 = f0_all[bin_i];
        double dfdt = dfdt_all[bin_i];
        double d2fdt2 = d2fdt2_all[bin_i];
        double phi0 = phi0_all[bin_i];

        double DPr = DPr_all[bin_i];
        double DPi = DPi_all[bin_i];
        double DCr = DCr_all[bin_i];
        double DCi = DCi_all[bin_i];

        long q = (long) f0/T;

        for (int i = 0; i < 3; i ++)
        {
            for (int j = 0; j < 3; j++)
            {
                eplus[(i * 3 + j)] = eplus_in[(i * 3 + j) * num_bin + bin_i];
                ecross[(i * 3 + j)] = ecross_in[(i * 3 + j) * num_bin + bin_i];
            }
        }

        for (int n = 0;
    			 n < N;
    			 n += 1)
        {
        	 t = T*(double)(n)/(double)N;

        	 calc_xi_f(x, y, z, k, xi, fonfs, f0, dfdt, d2fdt2, T, t, n, N);		  // calc frequency and time variables
             calc_sep_vecs(r12, r21, r13, r31, r23, r32, x, y, z, n, N);       // calculate the S/C separation vectors
             calc_d_matrices(dplus, dcross, eplus, ecross, r12, r21, r13, r31, r23, r32);    // calculate pieces of waveform
             calc_kdotr(k, kdotr, r12, r21, r13, r31, r23, r32);    // calculate dot product
             get_transfer(q, f0, dfdt, d2fdt2, phi0,
                              T, t, n, N,
                              kdotr, TR, TI,
                              dplus, dcross,
             				  xi, fonfs,
                              DPr, DPi, DCr, DCi);     // Calculating Transfer function
        	 fill_time_series(bin_i, num_bin, n, N, TR, TI, data12, data21, data13, data31, data23, data32); // Fill  time series data arrays with slowly evolving signal.
        }
    }
}


void GenWave_wrap(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int num_bin)
{
    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    GenWave<<<num_blocks, NUM_THREADS>>>(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}
