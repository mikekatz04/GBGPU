// Code by Michael Katz. Based on code by Travis Robson, Neil Cornish, Tyson Littenberg, Stas Babak


// imports
#include "stdio.h"

#include "new_fastGB.hh"
#include "global.h"
#include "LISA.h"
#include "cuda_complex.hpp"
#include "omp.h"

#ifdef __CUDACC__
#else
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>
#endif

// Get constants that determine eccentricity weighting of each harmonic
CUDA_CALLABLE_MEMBER
void get_j_consts(double* Cpj, double* Spj, double* Ccj, double* Scj, double beta, double e, double cosiota, int j)
{

	double result;

    // bessel functions on CPU vs GPU
    #ifdef __CUDACC__
    double Jj   = jn(j,   j*e);
	double Jjp1 = jn(j+1, j*e);
	double Jjm1 = jn(j-1, j*e);
    #else
	double Jj   = gsl_sf_bessel_Jn(j,   j*e);
	double Jjp1 = gsl_sf_bessel_Jn(j+1, j*e);
	double Jjm1 = gsl_sf_bessel_Jn(j-1, j*e);
    #endif

	result  = cos(2*beta)*(1 + cosiota*cosiota)*e*(1-e*e)*j*(Jjp1 - Jjm1);
	result -= (cos(2*beta)*(1 + cosiota*cosiota)*(e*e-2)+e*e*sqrt(1-cosiota*cosiota))*Jj;

	*Cpj = result*2/e/e; // CPj

    result  = e*Jjm1;
	result -= (1+(1-e*e)*j)*Jj;

	*Spj = result*4/e/e*sin(2*beta)*(1+cosiota*cosiota)*sqrt(1-e*e); // Spj

    result  = 2*e*(1-e*e)*j*Jjm1;
    result -= 2*(1+j*(1-e*e)-e*e/2)*Jj;

    *Ccj = result*4/e/e*sin(2*beta)*cosiota;  // Ccj

    result  = e*Jjm1;
    result -= (1+(1-e*e)*j)*Jj;

    *Scj = result*8/e/e*sqrt(1-e*e)*cos(2*beta)*cosiota;  // Scj

}

// Get transfer information for eccentric or circular GBs
CUDA_CALLABLE_MEMBER
void set_const_trans_eccTrip(double* DPr, double* DPi, double* DCr, double* DCi,
                             double amp, double cosiota, double psi, double beta, double e, int mode_j, int bin_i, int num_bin)
{
	double sinps, cosps;

	double Cpj, Spj, Ccj, Scj;

    double DPr_temp, DPi_temp, DCr_temp, DCi_temp;

    cosps = cos(2.*psi);
    sinps = sin(2.*psi);

    // if eccentricity is very small, assume circular
    if (e < 0.000001)
    {
        //Calculate GW polarization amplitudes
    	double Aplus  = amp*(1. + cosiota*cosiota);
    	double Across = -2.0*amp*cosiota;

    	//Calculate constant pieces of transfer functions
    	DPr_temp   =  Aplus*cosps;
    	DPi_temp   = -Across*sinps;
    	DCr_temp   = -Aplus*sinps;
    	DCi_temp   = -Across*cosps;
    }

    else
    {
        // Get constants determining eccentric mode weighting
        get_j_consts(&Cpj, &Spj, &Ccj, &Scj, beta, e, cosiota, mode_j);

    	// Calculate constant pieces of transfer functions
        DPr_temp    =  amp*(Cpj*cosps - Scj*sinps);
    	DPi_temp    = -amp*(Scj*cosps + Ccj*sinps);
    	DCr_temp    = -amp*(Cpj*sinps + Scj*cosps);
    	DCi_temp    =  amp*(Spj*sinps - Ccj*cosps);
    }

    // read out
    DPr[bin_i] = DPr_temp;
    DPi[bin_i] = DPi_temp;
    DCr[bin_i] = DCr_temp;
    DCi[bin_i] = DCi_temp;
}

// get the u, v, k basis vectors based on sky location of source
CUDA_CALLABLE_MEMBER
void get_basis_vecs(double *k, double *u, double *v, double phi, double theta, int bin_i, int num_bin)
{
	double costh, sinth, cosph, sinph;

    costh = cos(theta);
	sinth = sqrt(1.0-costh*costh);

	cosph = cos(phi);
	sinph = sin(phi);

    // read out
	u[0] =  costh*cosph;  u[1] =  costh*sinph;  u[2] = -sinth;
	v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
	k[0 * num_bin + bin_i] = -sinth*cosph;  k[1 * num_bin + bin_i] = -sinth*sinph;  k[2 * num_bin + bin_i] = -costh;
}

// TODO: check if this can be upped by reusing shared memory

// declare constants for GPU shared memory declarations
#define  NUM_THREADS 256
#define  MAX_MODES 4

// get basis tensors (eplus and ecross) and k array describing sky location of source
CUDA_KERNEL
void get_basis_tensors(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                       double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin)
{
    // Prepare shared memory if compiling for GPU
    #ifdef __CUDACC__
    CUDA_SHARED double u_all[3 * NUM_THREADS];
	CUDA_SHARED double v_all[3 * NUM_THREADS];

    double* u = &u_all[3 * threadIdx.x];
    double* v = &v_all[3 * threadIdx.x];
    #endif

    // setup loops for GPU or CPU
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

        // CPU placeholders for GPU shared memory
        // must be declared inside openMP loop to not have memory access issues
        #ifdef __CUDACC__
        #else

        double u_all[3];
        double v_all[3];

        double* u = &u_all[0];
        double* v = &v_all[0];

        #endif

        // get k, u, v
        get_basis_vecs(k, u, v, lam[bin_i], beta[bin_i], bin_i, num_bin); //Gravitational Wave source basis vectors

        // get the transfer constants
        set_const_trans_eccTrip(DPr, DPi, DCr, DCi, amp[bin_i], cosiota[bin_i], psi[bin_i], beta1[bin_i], e1[bin_i], mode_j, bin_i, num_bin);  // set the constant pieces of transfer function

        //GW polarization basis tensors
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                eplus[(i*3 + j) * num_bin + bin_i]  = v[i]*v[j] - u[i]*u[j];
                ecross[(i*3 + j) * num_bin + bin_i] = u[i]*v[j] + v[i]*u[j];
            }
        }
    }
}

// Wrap for call in Cython
// get the basis tensors and transfer constants
void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin)
{

    // change based on GPU or CPU
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    get_basis_tensors<<<num_blocks, NUM_THREADS>>>(
        eplus, ecross, DPr, DPi, DCr, DCi, k,
        amp, cosiota, psi, lam, beta, e1, beta1, mode_j, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    get_basis_tensors(
        eplus, ecross, DPr, DPi, DCr, DCi, k,
        amp, cosiota, psi, lam, beta, e1, beta1, mode_j, num_bin
    );

    #endif
}


// get x, y, z of spacecraft at a given time t
CUDA_CALLABLE_MEMBER
void spacecraft(double t, double* x, double* y, double* z, int n, int N)
{
    // n and N are left debugging purposes

    // kappa and lambda are constants determined in the Constants.h file
	double alpha;
	double beta1, beta2, beta3;
	double sa, sb, ca, cb;

	alpha = 2.*M_PI*fm*t + kappa;

	beta1 = 0. + lambda0;
	beta2 = 2.*M_PI/3. + lambda0;
	beta3 = 4.*M_PI/3. + lambda0;

	sa = sin(alpha);
	ca = cos(alpha);

    // spacecraft 1
	sb = sin(beta1);
	cb = cos(beta1);
	x[0] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[0] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[0] = -SQ3*AU*ec*(ca*cb + sa*sb);

    // spacecraft 2
    sb = sin(beta2);
	cb = cos(beta2);
	x[1] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[1] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[1] = -SQ3*AU*ec*(ca*cb + sa*sb);

    // spacecraft 3
	sb = sin(beta3);
	cb = cos(beta3);
	x[2] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[2] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[2] = -SQ3*AU*ec*(ca*cb + sa*sb);
}

// get u for inversion of Kepler's equation in relation to third body orbit
// TODO: change integration to actually integrating orbit?
CUDA_CALLABLE_MEMBER
double get_u(double l, double e)
{
	///////////////////////
	//
	// Invert Kepler's equation l = u - e sin(u)
	// Using Mikkola's method (1987)
	// referenced Tessmer & Gopakumar 2007
	//
	///////////////////////

	double u0;							// initial guess at eccentric anomaly
	double z, alpha, beta, s, w;		// auxiliary variables
	double mult;						// multiple number of 2pi

	int neg		 = 0;					// check if l is negative
	int over2pi  = 0;					// check if over 2pi
	int overpi	 = 0;					// check if over pi but not 2pi

	double f, f1, f2, f3, f4;			// pieces of root finder
	double u, u1, u2, u3, u4;

	// enforce the mean anomaly to be in the domain -pi < l < pi
	if (l < 0)
	{
		neg = 1;
		l   = -l;
	}
	if (l > 2.*M_PI)
	{
		over2pi = 1;
		mult	= floor(l/(2.*M_PI));
		l	   -= mult*2.*M_PI;
	}
	if (l > M_PI)
	{
		overpi = 1;
		l	   = 2.*M_PI - l;
	}

	alpha = (1. - e)/(4.*e + 0.5);
	beta  = 0.5*l/(4.*e + 0.5);

	z = sqrt(beta*beta + alpha*alpha*alpha);
	if (neg == 1) z = beta - z;
	else	      z = beta + z;

	// to handle nan's from negative arguments
	if (z < 0.) z = -pow(-z, 0.3333333333333333);
	else 	    z =  pow( z, 0.3333333333333333);

	s  = z - alpha/z;
	w  = s - 0.078*s*s*s*s*s/(1. + e);

	u0 = l + e*(3.*w - 4.*w*w*w);

	// now this initial guess must be iterated once with a 4th order Newton root finder
	f  = u0 - e*sin(u0) - l;
	f1 = 1. - e*cos(u0);
	f2 = u0 - f - l;
	f3 = 1. - f1;
	f4 = -f2;

	f2 *= 0.5;
	f3 *= 0.166666666666667;
	f4 *= 0.0416666666666667;

	u1 = -f/f1;
	u2 = -f/(f1 + f2*u1);
	u3 = -f/(f1 + f2*u2 + f3*u2*u2);
	u4 = -f/(f1 + f2*u3 + f3*u3*u3 + f4*u3*u3*u3);

	u = u0 + u4;

	if (overpi  == 1) u = 2.*M_PI - u;
	if (over2pi == 1) u = 2.*M_PI*mult + u;
	if (neg		== 1) u = -u;

	return u;
}

// get phi value for Line-of-sight velocity. See arXiv:1806.00500
CUDA_CALLABLE_MEMBER
double get_phi(double t, double T, double e, double n)
{
	double u, beta;

	u = get_u(n*(t-T), e);

	if (e == 0.) return u;

    // adjust if not circular
	beta = (1. - sqrt(1. - e*e))/e;

	return u + 2.*atan2( beta*sin(u), 1. - beta*cos(u));
}

// calculate the line-of-site velocity
// see equation 13 in arXiv:1806.00500
CUDA_CALLABLE_MEMBER
double get_vLOS(double A2, double omegabar, double e2, double n2, double T2, double t)
{
 	double phi2;

	phi2 = get_phi(t, T2, e2, n2);

	return A2*(sin(phi2 + omegabar) + e2*sin(omegabar));
}

// Calculate xi (delay to spacecraft) and fonfs (f over the LISA transfer frequency)
// call changes based on whether there is a third body
CUDA_CALLABLE_MEMBER
#ifdef __THIRD__
void calc_xi_f_eccTrip(double* x, double* y, double* z, double* k, double* xi, double* fonfs,
               double f0, double dfdt, double d2fdt2, double T, double t, int n, int N, int j,
               double A2, double omegabar, double e2, double n2, double T2)
#else
void calc_xi_f(double* x, double* y, double* z, double* k, double* xi, double* fonfs,
               double f0, double dfdt, double d2fdt2, double T, double t, int n, int N, int j)
#endif
{
	double f0_0, dfdt_0, d2fdt2_0;

	double kdotx_temp, f_temp, xi_temp;

    // rescale frequency information
	f0_0       = f0/T;
	dfdt_0   = dfdt/T/T;
	d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0_0;

    // get spacecraft positions
	spacecraft(t, x, y, z, n, N); // Calculate position of each spacecraft at time t
	for(int i = 0; i < 3; i++)
	{
        // sky position dotted with spacecraft location
		kdotx_temp = (x[i] * k[0] + y[i] * k[1] + z[i] * k[2])/C;

		//Wave arrival time at spacecraft i
		xi_temp    = t - kdotx_temp;
        xi[i] = xi_temp;

		//Second order approximation to frequency at spacecraft i
		f_temp     = f0_0 + dfdt_0 * xi_temp + 0.5 * d2fdt2_0 * xi_temp * xi_temp;

        // Add LOS velocity contribution to the frequency
        // Also adjust for j mode in eccentricity expansion
        #ifdef __THIRD__
        f_temp     *= (1. + get_vLOS(A2, omegabar, e2, n2, T2, xi_temp)/C)*(double)j/2.;
        #else
        f_temp     *= (double)j/2.;
        #endif

		//Ratio of true frequency to transfer frequency
		fonfs[i] = f_temp/fstar;
	}
}

// Get the LISA spacecraft separation vectors
CUDA_CALLABLE_MEMBER
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

// get sky position dotted with the spacecraft separation vectors
CUDA_CALLABLE_MEMBER
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
}

// get matrices describing transform through projections along arms
CUDA_CALLABLE_MEMBER
void calc_d_matrices(double *dplus, double *dcross, double* eplus, double* ecross,
                     double *r12, double *r21, double *r13, double *r31, double *r23, double *r32, int n)
{
    //Zero arrays to be summed
    dplus [(0*3 + 0)] = 0.0;
	dplus [(0*3 + 1)] = 0.0;
	dplus [(0*3 + 2)] = 0.0;
	dplus [(1*3 + 0)] = 0.;
	dplus [(1*3 + 1)] = 0.0;
    dplus [(1*3 + 2)] = 0.0;
	dplus [(2*3 + 0)] = 0.0;
	dplus [(2*3 + 1)] = 0.0;
	dplus [(2*3 + 2)] = 0.;
    dcross [(0*3 + 0)] = 0.0;
	dcross [(0*3 + 1)] = 0.0;
	dcross [(0*3 + 2)] = 0.0;
	dcross [(1*3 + 0)] = 0.;
	dcross [(1*3 + 1)] = 0.0;
    dcross [(1*3 + 2)] = 0.0;
	dcross [(2*3 + 0)] = 0.0;
	dcross [(2*3 + 1)] = 0.0;
	dcross [(2*3 + 2)] = 0.;

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
}

// get frqeuency of GW at time t to second order
CUDA_CALLABLE_MEMBER
double get_fGW(double f0, double dfdt, double d2fdt2, double T, double t)
{
    double dfdt_0, d2fdt2_0;
	f0        = f0/T;
	dfdt_0    = dfdt/T/T;
	d2fdt2_0  = 11./3.*dfdt_0*dfdt_0/f0;

	// assuming t0 = 0.
	return f0 + dfdt_0*t + 0.5*d2fdt2_0*t*t;
}

// Get step in integration functin of third body orbit
// Was a parabolic integration
// now uses trapezoidal integration
CUDA_CALLABLE_MEMBER
double parab_step_ET(double f0, double dfdt, double d2fdt2, double A2, double omegabar, double e2, double n2, double T2, double t0, double t0_old, int j, double T)
{
	// step in an integral using trapezoidal approximation to integrand
	// g1 starting point
    // g2 is midpoint if wanted to switch back to parabolic
	// g3 end-point

	double g1, g2, g3;

    double dtt = t0 - t0_old;

	g1 = get_vLOS(A2, omegabar, e2, n2, T2, t0_old)*get_fGW( f0,  dfdt,  d2fdt2, T, t0_old);
	//g2 = get_vLOS(A2, omegabar, e2, n2, T2, (t0 + t0_old)/2.)*get_fGW(f0,  dfdt,  d2fdt2, T, (t0 + t0_old)/2.);
	g3 = get_vLOS(A2, omegabar, e2, n2, T2, t0)*get_fGW(f0,  dfdt,  d2fdt2, T, t0);

    // return area from trapezoidal rule
    return (dtt * (g1 + g3)/2.*PI2/C)*(double)j/2.;
}

// get the transfer function
// call changes based on if there is a third body or not
CUDA_CALLABLE_MEMBER
#ifdef __THIRD__
void get_transfer_ET(int q, double f0, double dfdt, double d2fdt2, double phi0,
                 double T, double t, int n, int N,
                 double *kdotr, double *TR, double *TI,
                 double *dplus, double *dcross,
				 double *xi, double *fonfs,
                 double A2, double omegabar, double e2, double n2, double T2,
                 double DPr, double DPi, double DCr, double DCi, int mode_j, double* sum, double* prev_xi, int bin_i)
#else
void get_transfer_ET(int q, double f0, double dfdt, double d2fdt2, double phi0,
                 double T, double t, int n, int N,
                 double *kdotr, double *TR, double *TI,
                 double *dplus, double *dcross,
				 double *xi, double *fonfs,
                 double DPr, double DPi, double DCr, double DCi, int mode_j, int bin_i)
#endif
{
	double tran1r, tran1i;
	double tran2r, tran2i;
	double aevol;			// amplitude evolution factor
	//double arg1, arg2, sinc;
	double f0_0, dfdt_0, d2fdt2_0;
	double df;

    // rescale parameters
	f0       = f0/T;
	dfdt_0   = dfdt/T/T;

    // TODO: make adjustable again
    d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0;

    // adjusted df based on the frequency this waveform is occuring at
	df = PI2*(((double)q)/T);

	for(int i = 0; i < 3; i++)
	{
        // if there is a third body, need to take an integration step
        #ifdef __THIRD__
        sum[i] += parab_step_ET(f0 * T, dfdt,  d2fdt2,  A2,  omegabar,  e2,  n2,  T2,  xi[i], prev_xi[i], mode_j, T);
        #endif

		for(int j = 0; j < 3; j++)
		{
			if(i!=j)
			{
				//Argument of transfer function
				double arg1 = 0.5*fonfs[i]*(1. - kdotr[(i*3 + j)]);

				//Argument of complex exponentials
				double arg2 = (PI2*f0*xi[i] + phi0 + M_PI*dfdt_0*xi[i]*xi[i] + M_PI*d2fdt2_0*xi[i]*xi[i]*xi[i]/3.0) * (double)mode_j/2. - df*t ;

                // Add contribution from third body if needed
                #ifdef __THIRD__
                if (xi[i] > 0.0) arg2 += sum[i];
                #endif

                //Transfer function
				double sinc = 0.25*sin(arg1)/arg1;

				//Evolution of amplitude
				aevol = 1.0 + 0.66666666666666666666*dfdt_0/f0*xi[i];

				///Real and imaginary pieces of time series (no complex exponential)
                // -dplus due to difference with original fastGB

				tran1r = aevol*(-dplus[(i*3 + j)]*DPr + dcross[(i*3 + j)]*DCr);
				tran1i = aevol*(-dplus[(i*3 + j)]*DPi + dcross[(i*3 + j)]*DCi);

				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);

				//Real & Imaginary part of the slowly evolving signal
				TR[(i*3 + j)] = sinc*(tran1r*tran2r - tran1i*tran2i);
				TI[(i*3 + j)] = sinc*(tran1r*tran2i + tran1i*tran2r);

            }
            // fill with zeros in diagonal terms
            else
            {
                TR[(i*3 + j)] = 0.0;
				TI[(i*3 + j)] = 0.0;
            }
		}
	}
}

// fill the complex time series with terms from the transfer functions
CUDA_CALLABLE_MEMBER
void fill_time_series(int bin_i, int num_bin, int n, int N, double *TR, double *TI,
					  cmplx *data12, cmplx *data21, cmplx *data13,
					  cmplx *data31, cmplx *data23, cmplx *data32)
{
	data12[n * num_bin + bin_i]   = cmplx(TR[(0*3 + 1)], TI[(0*3 + 1)]);
	data21[n * num_bin + bin_i]   = cmplx(TR[(1*3 + 0)], TI[(1*3 + 0)]);
	data31[n * num_bin + bin_i]   = cmplx(TR[(2*3 + 0)], TI[(2*3 + 0)]);
	data13[n * num_bin + bin_i]   = cmplx(TR[(0*3 + 2)], TI[(0*3 + 2)]);
	data23[n * num_bin + bin_i]   = cmplx(TR[(1*3 + 2)], TI[(1*3 + 2)]);
	data32[n * num_bin + bin_i]   = cmplx(TR[(2*3 + 1)], TI[(2*3 + 1)]);
}

// define number of threads for the GenWave kernel
// Needs to be small to accomodate all the shared memory
#define NUM_THREADS_2 32

// Generate the time domain waveform information
// Call changes if there is a third body present
CUDA_KERNEL
#ifdef __THIRD__
void GenWave(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_in, double T, int N, int mode_j, int num_bin)

#else
void GenWave(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_in, double T, int N, int mode_j, int num_bin)

#endif
{

    // declare all of the shared memory to be used
    #ifdef __CUDACC__
    CUDA_SHARED double x_all[3 * NUM_THREADS_2];
    CUDA_SHARED double y_all[3 * NUM_THREADS_2];
    CUDA_SHARED double z_all[3 * NUM_THREADS_2];
    CUDA_SHARED double k_all[3 * NUM_THREADS_2];
    CUDA_SHARED double xi_all[3 * NUM_THREADS_2];
    CUDA_SHARED double fonfs_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r12_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r21_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r13_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r31_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r23_all[3 * NUM_THREADS_2];
    CUDA_SHARED double r32_all[3 * NUM_THREADS_2];
    CUDA_SHARED double sum_all[3 * NUM_THREADS_2];
    CUDA_SHARED double prev_xi_all[3 * NUM_THREADS_2];
    CUDA_SHARED double dplus_all[9 * NUM_THREADS_2];
    CUDA_SHARED double dcross_all[9 * NUM_THREADS_2];
    CUDA_SHARED double eplus_all[9 * NUM_THREADS_2];
    CUDA_SHARED double ecross_all[9 * NUM_THREADS_2];
    CUDA_SHARED double kdotr_all[9 * NUM_THREADS_2];
    CUDA_SHARED double TR_all[9 * NUM_THREADS_2];
    CUDA_SHARED double TI_all[9 * NUM_THREADS_2];

    // get shared arrays for the specific thread on the GPU
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
    double* sum = &sum_all[3 * threadIdx.x];
    double* prev_xi = &prev_xi_all[3 * threadIdx.x];
    double* dplus = &dplus_all[9 * threadIdx.x];
    double* dcross = &dcross_all[9 * threadIdx.x];
    double* eplus = &eplus_all[9 * threadIdx.x];
    double* ecross = &ecross_all[9 * threadIdx.x];
    double* kdotr = &kdotr_all[9 * threadIdx.x];
    double* TR = &TR_all[9 * threadIdx.x];
    double* TI = &TI_all[9 * threadIdx.x];

    #endif

    // prepare loops for CPU/GPU
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
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {

        #ifdef __CUDACC__
        #else
        // CPU placeholders for GPU shared memory
        // must be declared in the loop to avoid openMP issues
        double x_all[3];
        double y_all[3];
        double z_all[3];
        double k_all[3];
        double xi_all[3];
        double fonfs_all[3];
        double r12_all[3];
        double r21_all[3];
        double r13_all[3];
        double r31_all[3];
        double r23_all[3];
        double r32_all[3];
        double sum_all[3];
        double prev_xi_all[3];
        double dplus_all[9];
        double dcross_all[9];
        double eplus_all[9];
        double ecross_all[9];
        double kdotr_all[9];
        double TR_all[9];
        double TI_all[9];


        double* x = &x_all[0];
        double* y = &y_all[0];
        double* z = &z_all[0];
        double* k = &k_all[0];
        double* xi= &xi_all[0];
        double* fonfs = &fonfs_all[0];
        double* r12 = &r12_all[0];
        double* r21 = &r21_all[0];
        double* r13 = &r13_all[0];
        double* r31 = &r31_all[0];
        double* r23 = &r23_all[0];
        double* r32 = &r32_all[0];
        double* sum = &sum_all[0];
        double* prev_xi = &prev_xi_all[0];
        double* dplus = &dplus_all[0];
        double* dcross = &dcross_all[0];
        double* eplus = &eplus_all[0];
        double* ecross = &ecross_all[0];
        double* kdotr = &kdotr_all[0];
        double* TR = &TR_all[0];
        double* TI = &TI_all[0];

        #endif

        // prepare sum for thir body
        #ifdef __THIRD__
        for (int i = 0; i < 3; i += 1)
        {
            sum[i] = 0.0;
        }
        #endif

        // sky location vector
        k[0] = k_in[0 * num_bin + bin_i];
        k[1] = k_in[1 * num_bin + bin_i];
        k[2] = k_in[2 * num_bin + bin_i];

        // get all the parameters
        double f0 = f0_all[bin_i];
        double dfdt = dfdt_all[bin_i];
        double d2fdt2 = d2fdt2_all[bin_i];
        double phi0 = phi0_all[bin_i];

        // get all third body parameters
        #ifdef __THIRD__
        double A2 = A2_all[bin_i];
        double omegabar = omegabar_all[bin_i];
        double e2 = e2_all[bin_i];
        double n2 = n2_all[bin_i];
        double T2 = T2_all[bin_i];

        for (int i = 0; i < 3; i += 1)
        {
            prev_xi[i] = 0.0;
        }
        #endif

        // get transfer information
        double DPr = DPr_all[bin_i];
        double DPi = DPi_all[bin_i];
        double DCr = DCr_all[bin_i];
        double DCi = DCi_all[bin_i];

        int q = (int) f0*(double)mode_j/2.;

        // loop over points in the TD waveform information
        for (int n = 0;
    			 n < N;
    			 n += 1)
        {
            // get the polarization tensors
            for (int i = 0; i < 3; i ++)
            {
                for (int j = 0; j < 3; j++)
                {
                    eplus[(i * 3 + j)] = eplus_in[(i * 3 + j) * num_bin + bin_i];
                    ecross[(i * 3 + j)] = ecross_in[(i * 3 + j) * num_bin + bin_i];
                }
            }

            // get the time
            double t = T*(double)(n)/(double)N;

            // get xi and fonfs
            #ifdef __THIRD__
            calc_xi_f_eccTrip(x, y, z, k, xi, fonfs, f0, dfdt, d2fdt2, T, t, n, N, mode_j, A2, omegabar, e2, n2, T2); // calc frequency and time variables
            #else
            calc_xi_f(x, y, z, k, xi, fonfs, f0, dfdt, d2fdt2, T, t, n, N, mode_j); // calc frequency and time variables
            #endif

            // separation vectors of spacecraft
            calc_sep_vecs(r12, r21, r13, r31, r23, r32, x, y, z, n, N);       // calculate the S/C separation vectors
            // get projection matrices
            calc_d_matrices(dplus, dcross, eplus, ecross, r12, r21, r13, r31, r23, r32, n);    // calculate pieces of waveform
            // sky location dotted with separation vectors
            calc_kdotr(k, kdotr, r12, r21, r13, r31, r23, r32);    // calculate dot product

            // build the transfer functions
            // changes based on if there is a third body or not
            #ifdef __THIRD__
            get_transfer_ET(q, f0, dfdt, d2fdt2, phi0,
                          T, t, n, N,
                          kdotr, TR, TI,
                          dplus, dcross,
            				  xi, fonfs,
                          A2, omegabar, e2, n2, T2,
                          DPr, DPi, DCr, DCi, mode_j, sum, prev_xi, bin_i);     // Calculating Transfer function
            #else
            get_transfer_ET(q, f0, dfdt, d2fdt2, phi0,
                          T, t, n, N,
                          kdotr, TR, TI,
                          dplus, dcross,
                              xi, fonfs,
                          DPr, DPi, DCr, DCi, mode_j, bin_i);     // Calculating Transfer function
            #endif

            // Fill the time series with transfer information
            fill_time_series(bin_i, num_bin, n, N, TR, TI, data12, data21, data13, data31, data23, data32); // Fill  time series data arrays with slowly evolving signal.

            // if integrating over third body orbit, store previous xi values
            #ifdef __THIRD__
            for (int i = 0; i < 3; i += 1)
            {
                prev_xi[i] = xi[i];
            }
            #endif
        }
    }
}

// Wrapping function
#ifdef __THIRD__
void GenWaveThird_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int mode_j, int num_bin)
#else
void GenWave_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int mode_j, int num_bin)
#endif

{
    // adjust call based on third body and/or GPU
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS_2 -1)/NUM_THREADS_2);

    #ifdef __THIRD__
    GenWave<<<num_blocks, NUM_THREADS_2>>>(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         A2_all, omegabar_all, e2_all, n2_all, T2_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );
    #else
    GenWave<<<num_blocks, NUM_THREADS_2>>>(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );
    #endif
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    #ifdef __THIRD__
    GenWave(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         A2_all, omegabar_all, e2_all, n2_all, T2_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );
    #else
    GenWave(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );
    #endif
    #endif
}

// prepare the data for TDI calculations
CUDA_KERNEL
void unpack_data_1(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                   int N, int num_bin)
{

    // prepare the loop for CPU / GPU
    int start, end, increment;

    int ind_st = N/2;

    double N_double = (double) N;

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
        for (int i = 0;
    			 i < ind_st;
    			 i += 1)
        {
    		// populate from most negative (Nyquist) to most positive (Nyquist-1)
            int ind1 = i * num_bin + bin_i;
            int ind2 = (ind_st + i) * num_bin + bin_i;

            cmplx temp1, temp2;

            temp1 = data12[ind1];
            temp2 = data12[ind2];
            data12[ind2] = 0.5 * temp1 / N_double;
            data12[ind1] = 0.5 * temp2 / N_double;

            temp1 = data21[ind1];
            temp2 = data21[ind2];
            data21[ind2] = 0.5 * temp1 / N_double;
            data21[ind1] = 0.5 * temp2 / N_double;

            temp1 = data13[ind1];
            temp2 = data13[ind2];
            data13[ind2] = 0.5 * temp1 / N_double;
            data13[ind1] = 0.5 * temp2 / N_double;

            temp1 = data31[ind1];
            temp2 = data31[ind2];
            data31[ind2] = 0.5 * temp1 / N_double;
            data31[ind1] = 0.5 * temp2 / N_double;

            temp1 = data23[ind1];
            temp2 = data23[ind2];
            data23[ind2] = 0.5 * temp1 / N_double;
            data23[ind1] = 0.5 * temp2 / N_double;

            temp1 = data32[ind1];
            temp2 = data32[ind2];
            data32[ind2] = 0.5 * temp1 / N_double;
            data32[ind1] = 0.5 * temp2 / N_double;
    	}
    }
}

// wrap function for unpacking
void unpack_data_1_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                   int N, int num_bin)
{
    // call changes for GPU/CPU
    #ifdef __CUDACC__
    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    unpack_data_1<<<num_blocks, NUM_THREADS>>>(
        data12, data21, data13, data31, data23, data32,
        N, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else

    unpack_data_1(
        data12, data21, data13, data31, data23, data32,
        N, num_bin
    );

    #endif
}

// Get XYZ
CUDA_CALLABLE_MEMBER
void XYZ_sub(int i, int bin_i, int num_bin, cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32, double f0, int q, int M, double dt, double Tobs, double *XLS_r, double *YLS_r, double *ZLS_r,
					double* XSL_r, double* YSL_r, double* ZSL_r, double *XLS_i, double *YLS_i, double *ZLS_i, double *XSL_i, double *YSL_i, double *ZSL_i)
{
	double fonfs;
	double c3, s3, c2, s2, c1, s1;
	double f;
	double phiLS, cLS, sLS, phiSL, cSL, sSL;

	double X_re, X_im, Y_re, Y_im, Z_re, Z_im;

    // TDI phasing
	phiLS = PI2*f0*(dt/2.0-Larm/C);

	cLS = cos(phiLS);
	sLS = sin(phiLS);

	phiSL = M_PI/2.0-2.0*M_PI*f0*(Larm/C);
	cSL = cos(phiSL);
	sSL = sin(phiSL);

	f = ((double)(q + i - M/2))/Tobs;

	fonfs = f/fstar;

    c3 = cos(3.*fonfs);  c2 = cos(2.*fonfs);  c1 = cos(1.*fonfs);
	s3 = sin(3.*fonfs);  s2 = sin(2.*fonfs);  s1 = sin(1.*fonfs);

    cmplx temp;

    temp = a12[i * num_bin + bin_i];
    double a12_r = temp.real();
    double a12_i = temp.imag();

    temp = a21[i * num_bin + bin_i];
    double a21_r = temp.real();
    double a21_i = temp.imag();

    temp = a13[i * num_bin + bin_i];
    double a13_r = temp.real();
    double a13_i = temp.imag();

    temp = a31[i * num_bin + bin_i];
    double a31_r = temp.real();
    double a31_i = temp.imag();

    temp = a23[i * num_bin + bin_i];
    double a23_r = temp.real();
    double a23_i = temp.imag();

    temp = a32[i * num_bin + bin_i];
    double a32_r = temp.real();
    double a32_i = temp.imag();

	X_re   = (a12_r-a13_r)*c3 + (a12_i-a13_i)*s3 +
	           (a21_r-a31_r)*c2 + (a21_i-a31_i)*s2 +
	           (a13_r-a12_r)*c1 + (a13_i-a12_i)*s1 +
	           (a31_r-a21_r);

	X_im = (a12_i-a13_i)*c3 - (a12_r-a13_r)*s3 +
	           (a21_i-a31_i)*c2 - (a21_r-a31_r)*s2 +
	           (a13_i-a12_i)*c1 - (a13_r-a12_r)*s1 +
	           (a31_i-a21_i);

	Y_re   = (a23_r-a21_r)*c3 + (a23_i-a21_i)*s3 +
	           (a32_r-a12_r)*c2 + (a32_i-a12_i)*s2+
	           (a21_r-a23_r)*c1 + (a21_i-a23_i)*s1+
	           (a12_r-a32_r);

	Y_im = (a23_i-a21_i)*c3 - (a23_r-a21_r)*s3+
	           (a32_i-a12_i)*c2 - (a32_r-a12_r)*s2+
	           (a21_i-a23_i)*c1 - (a21_r-a23_r)*s1+
	           (a12_i-a32_i);

	Z_re   = (a31_r-a32_r)*c3 + (a31_i-a32_i)*s3+
	           (a13_r-a23_r)*c2 + (a13_i-a23_i)*s2+
	           (a32_r-a31_r)*c1 + (a32_i-a31_i)*s1+
	           (a23_r-a13_r);

	Z_im = (a31_i-a32_i)*c3 - (a31_r-a32_r)*s3+
	           (a13_i-a23_i)*c2 - (a13_r-a23_r)*s2+
	           (a32_i-a31_i)*c1 - (a32_r-a31_r)*s1+
	           (a23_i-a13_i);

	// Alternative polarization definition
	*XLS_r   =  (X_re*cLS - X_im*sLS);
	*XLS_i =  -(X_re*sLS + X_im*cLS);
	*YLS_r   =  (Y_re*cLS - Y_im*sLS);
	*YLS_i =  -(Y_re*sLS + Y_im*cLS);
	*ZLS_r   =  (Z_re*cLS - Z_im*sLS);
	*ZLS_i =  -(Z_re*sLS + Z_im*cLS);

    // original fastGB polarization (?)
	*XSL_r   =  2.0*fonfs*(X_re*cSL - X_im*sSL);
	*XSL_i =  2.0*fonfs*(X_re*sSL + X_im*cSL);
	*YSL_r   =  2.0*fonfs*(Y_re*cSL - Y_im*sSL);
	*YSL_i =  2.0*fonfs*(Y_re*sSL + Y_im*cSL);
	*ZSL_r   =  2.0*fonfs*(Z_re*cSL - Z_im*sSL);
	*ZSL_i =  2.0*fonfs*(Z_re*sSL + Z_im*cSL);

	return;
}

// main function for computing TDIs
CUDA_KERNEL
void XYZ(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
              double *f0_all,
              int num_bin, int N, double dt, double T, double df, int mode_j){

    int M = (int) N;

    // prepare loop for GPU/CPU
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
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {

        // get initial frequency information
        double f0 = f0_all[bin_i] * (double)mode_j/2;
        int q = (int) f0;

		for (int i = 0;
				 i < M;
				 i += 1)
		{

	        double XLS_r, YLS_r, ZLS_r, XSL_r, YSL_r, ZSL_r, XLS_i, YLS_i, ZLS_i, XSL_i, YSL_i, ZSL_i;

            // get sub
    		XYZ_sub(i, bin_i, num_bin, a12, a21, a13, a31, a23, a32, f0/T, q, N, dt, T,
    				&XLS_r, &YLS_r, &ZLS_r, &XSL_r, &YSL_r, &ZSL_r, &XLS_i, &YLS_i, &ZLS_i, &XSL_i, &YSL_i, &ZSL_i);

            // For reading out X, Y , Z
            //XLS[bin_i*(2*M) + 2*i] = XLS_r;
            //XLS[bin_i*(2*M) + 2*i+1] = XLS_i;
            //YLS[bin_i*(2*M) + 2*i] = YLS_r;
            //YLS[bin_i*(2*M) + 2*i+1] = YLS_i;
            //ZLS[bin_i*(2*M) + 2*i] = ZLS_r;
            //ZLS[bin_i*(2*M) + 2*i+1] = ZLS_i;

            double A_r, E_r, T_r, A_i, E_i, T_i;

            A_r = (2.0*XLS_r - YLS_r - ZLS_r)/3.0;
            A_i = (2.0*XLS_i - YLS_i - ZLS_i)/3.0;

            E_r = (ZLS_r-YLS_r) * invsqrt3;
            E_i = (ZLS_i-YLS_i) * invsqrt3;

            // read out
            // reuses memory to save memory
            // multiply by sqrt(T)
            a12[i * num_bin + bin_i] = sqrt(T) * cmplx(XLS_r, XLS_i);
            a21[i * num_bin + bin_i] = sqrt(T) * cmplx(A_r, A_i);
            a13[i * num_bin + bin_i] = sqrt(T) * cmplx(E_r, E_i);
        }
    }
}

// wrapper for TDI computation
void XYZ_wrap(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
              double *f0_all,
              int num_bin, int N, double dt, double T, double df, int mode_j)
{
    // Analyze based on GPU/CPU
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    XYZ<<<num_blocks, NUM_THREADS>>>(
        a12, a21, a13, a31, a23, a32,
        f0_all,
        num_bin, N, dt, T, df, mode_j
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    XYZ(
        a12, a21, a13, a31, a23, a32,
        f0_all,
        num_bin, N, dt, T, df, mode_j
    );

    #endif
}



// Add functionality for proper summation in the kernel
#ifdef __CUDACC__
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
#endif

// Add functionality for proper summation in the kernel
CUDA_CALLABLE_MEMBER
void atomicAddComplex(cmplx* a, cmplx b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables

  #ifdef __CUDACC__
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
  #else
  #pragma omp atomic
  *x += b.real();
  #pragma omp atomic
  *y += b.imag();
  #endif
}



// calculate batched log likelihood
CUDA_KERNEL
void fill_global(cmplx* A_glob, cmplx* E_glob, cmplx* A_template, cmplx* E_template, double* A_noise_factor, double* E_noise_factor, int* start_ind_all, int M, int num_bin, int per_group, int data_length)
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

    //#pragma omp parallel for
    #endif
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];
        int group_i = bin_i / per_group;
        int num_groups = num_bin / per_group;

		for (int i = 0;
				 i < M;
				 i += 1)
		{
            int j = start_ind + i;

            cmplx temp_A = A_template[i * num_bin + bin_i] * A_noise_factor[j];
            cmplx temp_E = E_template[i * num_bin + bin_i] * E_noise_factor[j];


            atomicAddComplex(&A_glob[group_i * data_length + j], temp_A);
            atomicAddComplex(&E_glob[group_i * data_length + j], temp_E);
            //printf("CHECK: %d %e %e %d %d %d %d %d %d\n", bin_i, A_template[i * num_bin + bin_i], temp_A, group_i, data_length, j, num_groups, per_group, i);
        }
    }
}


// wrapper for log likelihood
void fill_global_wrap(cmplx* A_glob, cmplx* E_glob, cmplx* A_template, cmplx* E_template, double* A_noise_factor, double* E_noise_factor, int* start_ind_all, int M, int num_bin, int per_group, int data_length)
{
    // GPU / CPU difference
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    fill_global<<<num_blocks, NUM_THREADS>>>(
        A_glob, E_glob, A_template, E_template, A_noise_factor, E_noise_factor, start_ind_all, M, num_bin, per_group, data_length
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    fill_global(
        A_glob, E_glob, A_template, E_template, A_noise_factor, E_noise_factor, start_ind_all, M, num_bin, per_group, data_length
    );

    #endif
}

// calculate batched log likelihood
CUDA_KERNEL
void get_ll(double* d_h, double* h_h, cmplx* A_template, cmplx* E_template, cmplx* A_data, cmplx* E_data, double* A_noise_factor, double* E_noise_factor, int* start_ind_all, int M, int num_bin)
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

    #pragma omp parallel for
    #endif
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];

        // initialize likelihood
        double h_h_temp = 0.0;
        double d_h_temp = 0.0;
		for (int i = 0;
				 i < M;
				 i += 1)
		{
            int j = start_ind + i;

            // calculate h term
            cmplx h_A = A_template[i * num_bin + bin_i] * A_noise_factor[j];
            cmplx h_E = E_template[i * num_bin + bin_i] * E_noise_factor[j];

            // get <d|h> term
            d_h_temp += gcmplx::real(gcmplx::conj(A_data[j]) * h_A);
            d_h_temp += gcmplx::real(gcmplx::conj(E_data[j]) * h_E);

            // <h|h>
            h_h_temp += gcmplx::real(gcmplx::conj(h_A) * h_A);
            h_h_temp += gcmplx::real(gcmplx::conj(h_E) * h_E);
        }

        // read out
        d_h[bin_i] =  4. * d_h_temp;
        h_h[bin_i] =  4. * h_h_temp;
    }
}


// wrapper for log likelihood
void get_ll_wrap(double* d_h, double* h_h,
                 cmplx* A_template, cmplx* E_template,
                 cmplx* A_data, cmplx* E_data,
                 double* A_noise_factor, double* E_noise_factor,
                 int* start_ind, int M, int num_bin)
{
    // GPU / CPU difference
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    get_ll<<<num_blocks, NUM_THREADS>>>(
        d_h, h_h, A_template, E_template, A_data, E_data, A_noise_factor, E_noise_factor, start_ind, M, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    get_ll(
        d_h, h_h, A_template, E_template, A_data, E_data, A_noise_factor, E_noise_factor, start_ind, M, num_bin
    );

    #endif
}
