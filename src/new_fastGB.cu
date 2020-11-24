#include "stdio.h"
#include "new_fastGB.hh"
#include "global.h"
#include "LISA.h"
#include "cuda_complex.hpp"
#include "omp.h"

#ifdef __CUDACC__
#include <cufft.h>
#else
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>
#endif


CUDA_CALLABLE_MEMBER
void get_j_consts(double* Cpj, double* Spj, double* Ccj, double* Scj, double beta, double e, double cosiota, int j)
{

	double result;

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

CUDA_CALLABLE_MEMBER
void set_const_trans_eccTrip(double* DPr, double* DPi, double* DCr, double* DCi,
                             double amp, double cosiota, double psi, double beta, double e, int mode_j, int bin_i, int num_bin)
{
	double sinps, cosps;

	double Cpj, Spj, Ccj, Scj;

    get_j_consts(&Cpj, &Spj, &Ccj, &Scj, beta, e, cosiota, mode_j);

	//Calculate cos and sin of polarization
	cosps = cos(2.*psi);
	sinps = sin(2.*psi);

	//Calculate constant pieces of transfer functions

    //printf("%e %e %e %e %e %e %e %e %e %d \n", Cpj, Spj, Ccj, Scj, psi, amp, beta, e, cosiota, mode_j);
	DPr[bin_i]    =  amp*(Cpj*cosps - Scj*sinps);
	DPi[bin_i]    = -amp*(Scj*cosps + Ccj*sinps);
	DCr[bin_i]    = -amp*(Cpj*sinps + Scj*cosps);
	DCi[bin_i]    =  amp*(Spj*sinps - Ccj*cosps);

    //if (bin_i == 0) printf("%e %e %e %e %e\n", DPr[bin_i], amp, beta, e, cosps);
}


CUDA_CALLABLE_MEMBER
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


CUDA_CALLABLE_MEMBER
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
#define  NUM_THREADS 256
#define  MAX_MODES 4
CUDA_KERNEL
void get_basis_tensors(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                       double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin)
{
	 // GW basis vectors

    #ifdef __CUDACC__
    CUDA_SHARED double u_all[3 * NUM_THREADS];
	CUDA_SHARED double v_all[3 * NUM_THREADS];

    double* u = &u_all[3 * threadIdx.x];
    double* v = &v_all[3 * threadIdx.x];
    #endif

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

        #ifdef __CUDACC__
        #else

        double u_all[3];
        double v_all[3];

        double* u = &u_all[0];
        double* v = &v_all[0];

        #endif

        get_basis_vecs(k, u, v, lam[bin_i], beta[bin_i], bin_i, num_bin); //Gravitational Wave source basis vectors

        //printf("%d %d %d\n", jj, j, num_modes);
        set_const_trans_eccTrip(DPr, DPi, DCr, DCi, amp[bin_i], cosiota[bin_i], psi[bin_i], beta1[bin_i], e1[bin_i], mode_j, bin_i, num_bin);  // set the constant pieces of transfer function
        //set_const_trans(DPr, DPi, DCr, DCi, amp[bin_i], cosiota[bin_i], psi[bin_i], bin_i);

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
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin)
{

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



CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
double get_phi(double t, double T, double e, double n)
{
	double u, beta;

	u = get_u(n*(t-T), e);

	if (e == 0.) return u;

	beta = (1. - sqrt(1. - e*e))/e;

	return u + 2.*atan2( beta*sin(u), 1. - beta*cos(u));
}

CUDA_CALLABLE_MEMBER
double get_vLOS(double A2, double omegabar, double e2, double n2, double T2, double t)
{
 	double phi2;

    T2 *= YEAR;
	phi2 = get_phi(t, T2, e2, n2); //if (t == 0.) fprintf(stdout, "phi2_{0}: %f\n", phi2);

	return A2*(sin(phi2 + omegabar) + e2*sin(omegabar));
}


CUDA_CALLABLE_MEMBER
void calc_xi_f_eccTrip(double* x, double* y, double* z, double* k, double* xi, double* fonfs,
               double f0, double dfdt, double d2fdt2, double T, double t, int n, int N, int j,
               double A2, double omegabar, double e2, double n2, double T2)
{
	double f0_0, dfdt_0, d2fdt2_0;

	double kdotx_temp, f_temp, xi_temp;

	f0_0       = f0/T;
	dfdt_0   = dfdt/T/T;
	d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0_0;

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
        f_temp     *= (1. + get_vLOS(A2, omegabar, e2, n2, T2, xi_temp)/C)*(double)j/2.;

		//Ratio of true frequency to transfer frequency
		fonfs[i] = f_temp/fstar;

	}
}


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


CUDA_CALLABLE_MEMBER
void ansfer(int q, double f0, double dfdt, double d2fdt2, double phi0,
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

                //if ((i == 0) && (j == 1) && (t == 1.536000000000000000e+05))
                //    printf("%.18e %.18e %.18e %.18e\n", kdotr[(i*3 + j)], fonfs[i], xi[i], phi0);
			}
            else
            {
                TR[(i*3 + j)] = 0.0;
				TI[(i*3 + j)] = 0.0;
            }
		}
	}
}

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

CUDA_CALLABLE_MEMBER
double parab_step_ET(double f0, double dfdt, double d2fdt2, double A2, double omegabar, double e2, double n2, double T2, double t0, double t0_old, int j, double T)
{
	// step in an integral using parabolic approximation to integrand
	// g1 starting point
	// g2 mid-point
	// g3 end-point

	double g1, g2, g3;

    double dtt = t0 - t0_old;

	g1 = get_vLOS(A2, omegabar, e2, n2, T2, t0_old)*get_fGW( f0,  dfdt,  d2fdt2, T, t0_old);
	//g2 = get_vLOS(A2, omegabar, e2, n2, T2, (t0 + t0_old)/2.)*get_fGW(f0,  dfdt,  d2fdt2, T, (t0 + t0_old)/2.);
	g3 = get_vLOS(A2, omegabar, e2, n2, T2, t0)*get_fGW(f0,  dfdt,  d2fdt2, T, t0);

    return (dtt * (g1 + g3)/2.*PI2/C)*(double)j/2.;
	//return (dtt * (g1 + g2)/2.*PI2/C)*(double)j/2.;
}


CUDA_CALLABLE_MEMBER
void get_transfer_ET(int q, double f0, double dfdt, double d2fdt2, double phi0,
                 double T, double t, int n, int N,
                 double *kdotr, double *TR, double *TI,
                 double *dplus, double *dcross,
				 double *xi, double *fonfs,
                 double A2, double omegabar, double e2, double n2, double T2,
                 double DPr, double DPi, double DCr, double DCi, int mode_j, double* sum, double* prev_xi, int bin_i)
{
	double tran1r, tran1i;
	double tran2r, tran2i;
	double aevol;			// amplitude evolution factor
	//double arg1, arg2, sinc;
	double f0_0, dfdt_0, d2fdt2_0;
	double df;

	f0       = f0/T;
	dfdt_0   = dfdt/T/T;

    // TODO: make adjustable again
    d2fdt2_0 = 11./3.*dfdt_0*dfdt_0/f0;

	df = PI2*(((double)q)/T);

	for(int i = 0; i < 3; i++)
	{
        sum[i] += parab_step_ET(f0 * T, dfdt,  d2fdt2,  A2,  omegabar,  e2,  n2,  T2,  xi[i], prev_xi[i], mode_j, T);

		for(int j = 0; j < 3; j++)
		{
			if(i!=j)
			{
				//Argument of transfer function
				// FIXME
				//arg1 = 0.5*fonfs[i]*(1. - kdotr[i][j]);
				double arg1 = 0.5*fonfs[i]*(1. - kdotr[(i*3 + j)]);

				//Argument of complex exponentials
				double arg2 = (PI2*f0*xi[i] + phi0 + M_PI*dfdt_0*xi[i]*xi[i] + M_PI*d2fdt2_0*xi[i]*xi[i]*xi[i]/3.0) * mode_j/2. - df*t ;

                if (xi[i] > 0.0) arg2 += sum[i];

                //if ((i == 2) && (bin_i == 0) && (j == 1)) printf(" %d %d %e %.18e %e\n", i, j, xi[i], arg2, sum[i]);


                //if ((i == 0) && (j == 1))printf("%d %d %e\n", n, i, xi[i]);
				//Transfer function
				double sinc = 0.25*sin(arg1)/arg1;

				//Evolution of amplitude
				aevol = 1.0 + 0.66666666666666666666*dfdt_0/f0*xi[i];

				///Real and imaginary pieces of time series (no complex exponential)
                // -plus due to difference with original fastGB

				tran1r = aevol*(-dplus[(i*3 + j)]*DPr + dcross[(i*3 + j)]*DCr);
				tran1i = aevol*(-dplus[(i*3 + j)]*DPi + dcross[(i*3 + j)]*DCi);

				//Real and imaginry components of complex exponential
				tran2r = cos(arg1 + arg2);
				tran2i = sin(arg1 + arg2);

				//Real & Imaginary part of the slowly evolving signal
				TR[(i*3 + j)] = sinc*(tran1r*tran2r - tran1i*tran2i);
				TI[(i*3 + j)] = sinc*(tran1r*tran2i + tran1i*tran2r);

                //if ((i == 0) && (j == 1) && (t == 1.536000000000000000e+05))
                //    printf("%.18e %.18e %.18e %.18e\n", kdotr[(i*3 + j)], fonfs[i], xi[i], phi0);
			}
            else
            {
                TR[(i*3 + j)] = 0.0;
				TI[(i*3 + j)] = 0.0;
            }
		}
	}
}



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


#define NUM_THREADS_2 32


CUDA_KERNEL
void GenWave(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_in, double T, int N, int mode_j, int num_bin)
{


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

    double dtt = T/(double)(N-1);

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

        for (int i = 0; i < 3; i += 1)
        {
            sum[i] = 0.0;
        }

        k[0] = k_in[0 * num_bin + bin_i];
        k[1] = k_in[1 * num_bin + bin_i];
        k[2] = k_in[2 * num_bin + bin_i];

        double f0 = f0_all[bin_i];
        double dfdt = dfdt_all[bin_i];
        double d2fdt2 = d2fdt2_all[bin_i];
        double phi0 = phi0_all[bin_i];

        double A2 = A2_all[bin_i];
        double omegabar = omegabar_all[bin_i];
        double e2 = e2_all[bin_i];
        double n2 = n2_all[bin_i];
        double T2 = T2_all[bin_i];

        for (int i = 0; i < 3; i += 1)
        {
            prev_xi[i] = 0.0;
        }

        for (int n = 0;
    			 n < N;
    			 n += 1)
        {

            double DPr = DPr_all[bin_i];
            double DPi = DPi_all[bin_i];
            double DCr = DCr_all[bin_i];
            double DCi = DCi_all[bin_i];

            int q = (int) f0;

            for (int i = 0; i < 3; i ++)
            {
                for (int j = 0; j < 3; j++)
                {
                    eplus[(i * 3 + j)] = eplus_in[(i * 3 + j) * num_bin + bin_i];
                    ecross[(i * 3 + j)] = ecross_in[(i * 3 + j) * num_bin + bin_i];
                }
            }

            double t = T*(double)(n)/(double)N;

            calc_xi_f_eccTrip(x, y, z, k, xi, fonfs, f0, dfdt, d2fdt2, T, t, n, N, mode_j, A2, omegabar, e2, n2, T2);		  // calc frequency and time variables
            calc_sep_vecs(r12, r21, r13, r31, r23, r32, x, y, z, n, N);       // calculate the S/C separation vectors
            calc_d_matrices(dplus, dcross, eplus, ecross, r12, r21, r13, r31, r23, r32, n);    // calculate pieces of waveform
            calc_kdotr(k, kdotr, r12, r21, r13, r31, r23, r32);    // calculate dot product
            get_transfer_ET(q, f0, dfdt, d2fdt2, phi0,
                          T, t, n, N,
                          kdotr, TR, TI,
                          dplus, dcross,
            				  xi, fonfs,
                          A2, omegabar, e2, n2, T2,
                          DPr, DPi, DCr, DCi, mode_j, sum, prev_xi, bin_i);     // Calculating Transfer function
            fill_time_series(bin_i, num_bin, n, N, TR, TI, data12, data21, data13, data31, data23, data32); // Fill  time series data arrays with slowly evolving signal.

            for (int i = 0; i < 3; i += 1)
            {
                prev_xi[i] = xi[i];
            }
        }
    }
}


void GenWave_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int mode_j, int num_bin)
{

    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS_2 -1)/NUM_THREADS_2);

    GenWave<<<num_blocks, NUM_THREADS_2>>>(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         A2_all, omegabar_all, e2_all, n2_all, T2_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    GenWave(
        data12, data21, data13, data31, data23, data32,
        eplus_in, ecross_in,
         f0_all, dfdt_all, d2fdt2_all, phi0_all,
         A2_all, omegabar_all, e2_all, n2_all, T2_all,
         DPr_all, DPi_all, DCr_all, DCi_all,
         k_all, T, N, mode_j, num_bin
    );

    #endif
}


/*
void fft_data_wrap(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32, int num_bin, int N)
{

    cufftHandle plan;

    if (cufftPlan1d(&plan, N, CUFFT_Z2Z, num_bin) != CUFFT_SUCCESS){
          	fprintf(stderr, "CUFFT error: Plan creation failed");
          	return;	}

    if (cufftExecZ2Z(plan, (cufftDoubleComplex*)data12, (cufftDoubleComplex*)data12, -1) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
	return;}
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

    cufftDestroy(plan);
}
*/


CUDA_KERNEL
void unpack_data_1(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                   int N, int num_bin)
{


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

void unpack_data_1_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                   int N, int num_bin)
{
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



CUDA_CALLABLE_MEMBER
void XYZ_sub(int i, int bin_i, int num_bin, cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32, double f0, int q, int M, double dt, double Tobs, double *XLS_r, double *YLS_r, double *ZLS_r,
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

        //if (i == 0) printf("%.18e %.18e %.18e %.18e %d %d \n", fonfs, f, fstar, Tobs, q, M/2);
		//printf("Stas fonfs = %f, %f, %f, %f \n", fonfs, f, fstar, Tobs);
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

		X_1   = (a12_r-a13_r)*c3 + (a12_i-a13_i)*s3 +
		           (a21_r-a31_r)*c2 + (a21_i-a31_i)*s2 +
		           (a13_r-a12_r)*c1 + (a13_i-a12_i)*s1 +
		           (a31_r-a21_r);

		X_2 = (a12_i-a13_i)*c3 - (a12_r-a13_r)*s3 +
		           (a21_i-a31_i)*c2 - (a21_r-a31_r)*s2 +
		           (a13_i-a12_i)*c1 - (a13_r-a12_r)*s1 +
		           (a31_i-a21_i);

		Y_1   = (a23_r-a21_r)*c3 + (a23_i-a21_i)*s3 +
		           (a32_r-a12_r)*c2 + (a32_i-a12_i)*s2+
		           (a21_r-a23_r)*c1 + (a21_i-a23_i)*s1+
		           (a12_r-a32_r);

		Y_2 = (a23_i-a21_i)*c3 - (a23_r-a21_r)*s3+
		           (a32_i-a12_i)*c2 - (a32_r-a12_r)*s2+
		           (a21_i-a23_i)*c1 - (a21_r-a23_r)*s1+
		           (a12_i-a32_i);

		Z_1   = (a31_r-a32_r)*c3 + (a31_i-a32_i)*s3+
		           (a13_r-a23_r)*c2 + (a13_i-a23_i)*s2+
		           (a32_r-a31_r)*c1 + (a32_i-a31_i)*s1+
		           (a23_r-a13_r);

		Z_2 = (a31_i-a32_i)*c3 - (a31_r-a32_r)*s3+
		           (a13_i-a23_i)*c2 - (a13_r-a23_r)*s2+
		           (a32_i-a31_i)*c1 - (a32_r-a31_r)*s1+
		           (a23_i-a13_i);

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
		*XLS_i =  -(X_1*sLS + X_2*cLS);
		*YLS_r   =  (Y_1*cLS - Y_2*sLS);
		*YLS_i =  -(Y_1*sLS + Y_2*cLS);
		*ZLS_r   =  (Z_1*cLS - Z_2*sLS);
		*ZLS_i =  -(Z_1*sLS + Z_2*cLS);

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

CUDA_KERNEL
void XYZ(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
              double *f0_all,
              int num_bin, int N, double dt, double T, double df){

	int add_ind;
	double asd1, asd2, asd3;

    double *temp_XLS, *temp_YLS, *temp_ZLS;

    int M = (int) N;

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

        double f0 = f0_all[bin_i];
        int q = (int) f0;

		for (int i = 0;
				 i < M;
				 i += 1)
		{


	        double XLS_r, YLS_r, ZLS_r, XSL_r, YSL_r, ZSL_r, XLS_i, YLS_i, ZLS_i, XSL_i, YSL_i, ZSL_i;

    		XYZ_sub(i, bin_i, num_bin, a12, a21, a13, a31, a23, a32, f0/T, q, N, dt, T,
    				&XLS_r, &YLS_r, &ZLS_r, &XSL_r, &YSL_r, &ZSL_r, &XLS_i, &YLS_i, &ZLS_i, &XSL_i, &YSL_i, &ZSL_i);

    		//add_ind = (wfm->q + i - M/2);

            /*XLS[bin_i*(2*M) + 2*i] = XLS_r;
            XLS[bin_i*(2*M) + 2*i+1] = XLS_i;
            YLS[bin_i*(2*M) + 2*i] = YLS_r;
            YLS[bin_i*(2*M) + 2*i+1] = YLS_i;
            ZLS[bin_i*(2*M) + 2*i] = ZLS_r;
            ZLS[bin_i*(2*M) + 2*i+1] = ZLS_i;*/

            double A_r, E_r, T_r, A_i, E_i, T_i;

            A_r = (2.0*XLS_r - YLS_r - ZLS_r)/3.0;
            A_i = (2.0*XLS_i - YLS_i - ZLS_i)/3.0;

            E_r = (ZLS_r-YLS_r) * invsqrt3;
            E_i = (ZLS_i-YLS_i) * invsqrt3;

            a12[i * num_bin + bin_i] = sqrt(T) * cmplx(XLS_r, XLS_i);
            a21[i * num_bin + bin_i] = sqrt(T) * cmplx(A_r, A_i);
            a13[i * num_bin + bin_i] = sqrt(T) * cmplx(E_r, E_i);

    		//atomicAddDouble(&XLS[2*add_ind], XLS_r/asd1);
    		//atomicAddDouble(&XLS[2*add_ind+1], XLS_i/asd1);

    		//atomicAddDouble(&YLS[2*add_ind], YLS_r/asd2);
    		//atomicAddDouble(&YLS[2*add_ind+1], YLS_i/asd2);

    		//atomicAddDouble(&ZLS[2*add_ind], ZLS_r/asd3);
    		//atomicAddDouble(&ZLS[2*add_ind+1], ZLS_i/asd3);

    		/*atomicAddDouble(&XSL[2*add_ind], XSL_r/asd1);
    		atomicAddDouble(&XSL[2*add_ind+1], XSL_i)/asd1;

    		atomicAddDouble(&YSL[2*add_ind], YSL_r/asd2);
    		atomicAddDouble(&YSL[2*add_ind+1], YSL_i/asd2);

    		atomicAddDouble(&YSL[2*add_ind], ZSL_r/asd3);
    		atomicAddDouble(&ZSL[2*add_ind+1], ZSL_i/asd3);*/


        }
    }
}


void XYZ_wrap(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
              double *f0_all,
              int num_bin, int N, double dt, double T, double df)
{

    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    XYZ<<<num_blocks, NUM_THREADS>>>(
        a12, a21, a13, a31, a23, a32,
        f0_all,
        num_bin, N, dt, T, df
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    XYZ(
        a12, a21, a13, a31, a23, a32,
        f0_all,
        num_bin, N, dt, T, df
    );

    #endif
}
