#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <curand_kernel.h>

#include "block_io.hpp"
#include "common.hpp"

#include "SharedMemoryGBGPU.hpp"
#include "LISA.h"
#include "global.h"
#include "math.h"

__device__ void spacecraft(double P1[3], double P2[3], double P3[3], double t)
{
    // """Compute space craft positions as a function of time"""
    // # kappa and lambda are constants determined in the Constants.h file

    // # angular quantities defining orbit
    double alpha = 2.0 * M_PI * fm * t + kappa;

    double beta1 = 0.0 + lambda0;
    double beta2 = 2.0 * M_PI / 3.0 + lambda0;
    double beta3 = 4.0 * M_PI / 3.0 + lambda0;

    double sa = sin(alpha);
    double ca = cos(alpha);

    // spacecraft 1
    double sb = sin(beta1);
    double cb = cos(beta1);

    P1[0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb);
    P1[1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb);
    P1[2] = -SQ3 * AU * ec * (ca * cb + sa * sb);

    // spacecraft 2
    sb = sin(beta2);
    cb = cos(beta2);
    P2[0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb);
    P2[1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb);
    P2[2] = -SQ3 * AU * ec * (ca * cb + sa * sb);

    // spacecraft 3
    sb = sin(beta3);
    cb = cos(beta3);
    P3[0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb);
    P3[1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb);
    P3[2] = -SQ3 * AU * ec * (ca * cb + sa * sb);
}

__inline__ __device__ void get_eplus(double eplus[], double u[], double v[])
{
    double outer_val_v, outer_val_u;
    for (int i = 0; i < 3; i += 1)
    {
        for (int j = 0; j < 3; j += 1)
        {
            outer_val_v = v[i] * v[j];
            outer_val_u = u[i] * u[j];
            eplus[i * 3 + j] = outer_val_v - outer_val_u;
        }
    }
}

__inline__ __device__ void get_ecross(double ecross[], double u[], double v[])
{
    double outer_val_1, outer_val_2;
#pragma unroll
    for (int i = 0; i < 3; i += 1)
    {
#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            outer_val_1 = u[i] * v[j];
            outer_val_2 = v[i] * u[j];
            ecross[i * 3 + j] = outer_val_1 + outer_val_2;
        }
    }
}

__inline__ __device__ void AET_from_XYZ_swap(cmplx *X_in, cmplx *Y_in, cmplx *Z_in)
{
    cmplx X = *X_in;
    cmplx Y = *Y_in;
    cmplx Z = *Z_in;

    // A
    *X_in = (Z - X) / sqrt(2.0);

    // E
    *Y_in = (X - 2.0 * Y + Z) / sqrt(6.0);

    // T
    *Z_in = (X + Y + Z) / sqrt(3.0);
}

template <class FFT>
__device__ void build_single_waveform(
    cmplx *wave,
    unsigned int *start_ind,
    double amp,
    double f0,
    double fdot0,
    double fddot0,
    double phi0,
    double iota,
    double psi,
    double lam,
    double theta,
    double T,
    double dt,
    int N,
    int bin_i)
{

    using complex_type = cmplx;

    double eplus[9] = {0.};
    double ecross[9] = {0.};
    double u[3] = {0.};
    double v[3] = {0.};
    double k[3] = {0.};
    double P1[3] = {0.};
    double P2[3] = {0.};
    double P3[3] = {0.};
    double kdotr[6] = {0.};
    double kdotP[3] = {0.};
    double xi[3] = {0.};
    double fonfs[3] = {0.};
    cmplx Aij[3] = {0.};
    double r12[3] = {0.};
    double r13[3] = {0.};
    double r23[3] = {0.};
    double r31[3] = {0.};
    double argS;
    cmplx tmp_r12p;
    cmplx tmp_r31p;
    cmplx tmp_r23p;

    cmplx tmp_r12c;
    cmplx tmp_r31c;
    cmplx tmp_r23c;
    double fi;
    double arg_ij;
    unsigned int s;
    unsigned int A_ind;
    cmplx tmp_X, tmp_Y, tmp_Z;
    cmplx tmp2_X, tmp2_Y, tmp2_Z;

    cmplx I(0.0, 1.0);

    cmplx Gs[6] = {0.};
    // order is (12, 23, 31, 21, 32, 13)
    // 21 is good
    // 31 in right spot but -1
    //
    unsigned int s_all[6] = {0, 1, 2, 1, 2, 0};

    cmplx *X = &wave[0];
    cmplx *Y = &wave[N];
    cmplx *Z = &wave[2 * N];

    // index of nearest Fourier bin
    double q = rint(f0 * T);
    double df = 2.0 * M_PI * (q / T);
    double om = 2.0 * M_PI * f0;
    double f;
    double omL, SomL;
    cmplx fctr, fctr2, fctr3, tdi2_factor;
    double omegaL;

    // get initial setup
    double cosiota = cos(iota);
    double cosps = cos(2.0 * psi);
    double sinps = sin(2.0 * psi);
    double Aplus = amp * (1.0 + cosiota * cosiota);
    double Across = -2.0 * amp * cosiota;

    double sinth = sin(theta);
    double costh = cos(theta);
    double sinph = sin(lam);
    double cosph = cos(lam);

    u[0] = costh * cosph;
    u[1] = costh * sinph;
    u[2] = -sinth;

    v[0] = sinph;
    v[1] = -cosph;
    v[2] = 0.0;

    k[0] = -sinth * cosph;
    k[1] = -sinth * sinph;
    k[2] = -costh;

    get_eplus(eplus, u, v);
    get_ecross(ecross, u, v);

    cmplx DP = {Aplus * cosps, -Across * sinps};
    cmplx DC = {-Aplus * sinps, -Across * cosps};

    double delta_t_slow = T / (double)N;
    double t, xi_tmp;

    // construct slow part
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        t = i * delta_t_slow;
        spacecraft(P1, P2, P3, t);

// reset sum terms
#pragma unroll
        for (int j = 0; j < 6; j += 1)
        {
            kdotr[j] = 0.0;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            kdotP[j] = 0.0;
            Aij[j] = 0.0;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            r12[j] = (P2[j] - P1[j]) / Larm;
            r13[j] = (P3[j] - P1[j]) / Larm;
            r31[j] = -r13[j];
            r23[j] = (P3[j] - P2[j]) / Larm;

            kdotr[0] += k[j] * r12[j];
            kdotr[1] += k[j] * r23[j];
            kdotr[2] += k[j] * r31[j];

            kdotP[0] += (k[j] * P1[j]) / Clight;
            kdotP[1] += (k[j] * P2[j]) / Clight;
            kdotP[2] += (k[j] * P3[j]) / Clight;

            // 12
        }

        kdotr[3] = -kdotr[0];
        kdotr[4] = -kdotr[1];
        kdotr[5] = -kdotr[2];

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            xi_tmp = t - kdotP[j];
            xi[j] = xi_tmp;
            fi = (f0 + fdot0 * xi_tmp + 1 / 2.0 * fddot0 * (xi_tmp * xi_tmp));
            fonfs[j] = fi / fstar;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            tmp_r12p = 0.0;
            tmp_r31p = 0.0;
            tmp_r23p = 0.0;

            tmp_r12c = 0.0;
            tmp_r31c = 0.0;
            tmp_r23c = 0.0;

#pragma unroll
            for (int k = 0; k < 3; k += 1)
            {
                tmp_r12p += eplus[j * 3 + k] * r12[k];
                tmp_r31p += eplus[j * 3 + k] * r31[k];
                tmp_r23p += eplus[j * 3 + k] * r23[k];

                tmp_r12c += ecross[j * 3 + k] * r12[k];
                tmp_r31c += ecross[j * 3 + k] * r31[k];
                tmp_r23c += ecross[j * 3 + k] * r23[k];
            }

            Aij[0] += (tmp_r12p * r12[j] * DP + tmp_r12c * r12[j] * DC);
            Aij[1] += (tmp_r23p * r23[j] * DP + tmp_r23c * r23[j] * DC);
            Aij[2] += (tmp_r31p * r31[j] * DP + tmp_r31c * r31[j] * DC);
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            xi_tmp = xi[j];
            argS = (phi0 + (om - df) * t + M_PI * fdot0 * (xi_tmp * xi_tmp) + 1. / 3. * M_PI * fddot0 * (xi_tmp * xi_tmp * xi_tmp));

            kdotP[j] = om * kdotP[j] - argS;
        }

#pragma unroll
        for (int j = 0; j < 6; j += 1)
        {

            s = s_all[j];
            A_ind = j % 3;
            arg_ij = 0.5 * fonfs[s] * (1 + kdotr[j]);

            Gs[j] = (0.25 * sin(arg_ij) / arg_ij * exp(-I * (arg_ij + kdotP[s])) * Aij[A_ind]);
        }

        f = (f0 + fdot0 * t + 1 / 2.0 * fddot0 * (t * t));

        omL = f / fstar;
        SomL = sin(omL);
        fctr = gcmplx::exp(-I * omL);
        fctr2 = 4.0 * omL * SomL * fctr / amp;

        // order is (12, 23, 31, 21, 32, 13)

        // Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        tmp_X = fctr2 * ((Gs[3] - Gs[2] + (Gs[0] - Gs[5]) * fctr));

        // tmp_X = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t);
        // tmp2_X = complex_type {tmp_X.real(), tmp_X.imag()};
        X[i] = tmp_X;

        tmp_Y = fctr2 * ((Gs[4] - Gs[0] + (Gs[1] - Gs[3]) * fctr));
        // tmp2_Y = complex_type {tmp_Y.real(), tmp_Y.imag()};
        Y[i] = tmp_Y;

        tmp_Z = fctr2 * ((Gs[5] - Gs[1] + (Gs[2] - Gs[4]) * fctr));
        // tmp_Z = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t);
        // tmp2_Z = complex_type {tmp_Z.real(), tmp_Z.imag()};
        Z[i] = tmp_Z;
    }

    __syncthreads();

    FFT().execute(reinterpret_cast<void *>(X));
    FFT().execute(reinterpret_cast<void *>(Y));
    FFT().execute(reinterpret_cast<void *>(Z));

    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        X[i] *= amp;
        Y[i] *= amp;
        Z[i] *= amp;
    }

    __syncthreads();

    cmplx tmp_switch;
    cmplx tmp1_switch;
    cmplx tmp2_switch;

    int N_over_2 = N / 2;
    fctr3 = 0.5 * T / (double)N;

    // if (tdi2)
    //{
    //     omegaL = 2. * M_PI * f0_out * (Larm / Clight);
    //     tdi2_factor = 2.0 * I * sin(2. * omegaL) * exp(-2. * I * omegaL);
    //     fctr3 *= tdi2_factor;
    // }

    // fftshift in numpy and multiply by fctr3
    for (int i = threadIdx.x; i < N_over_2; i += blockDim.x)
    {
        // X
        tmp1_switch = X[i];
        tmp2_switch = X[N_over_2 + i];

        X[N_over_2 + i] = tmp1_switch * fctr3;
        X[i] = tmp2_switch * fctr3;

        // Y
        tmp1_switch = Y[i];
        tmp2_switch = Y[N_over_2 + i];

        Y[N_over_2 + i] = tmp1_switch * fctr3;
        Y[i] = tmp2_switch * fctr3;

        // Z
        tmp1_switch = Z[i];
        tmp2_switch = Z[N_over_2 + i];

        Z[N_over_2 + i] = tmp1_switch * fctr3;
        Z[i] = tmp2_switch * fctr3;
    }

    // convert to A, E, T (they sit in X,Y,Z)
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        AET_from_XYZ_swap(&X[i], &Y[i], &Z[i]);
    }

    __syncthreads();

    double fmin = (q - ((double)N) / 2.) / T;
    *start_ind = (int)rint(fmin * T);

    return;
}



template <class FFT>
__device__ void build_new_single_waveform(
    cmplx *wave,
    unsigned int *start_ind,
    SingleGalacticBinary gb_in,
    int bin_i)
{

    using complex_type = cmplx;

    double eplus[9] = {0.};
    double ecross[9] = {0.};
    double u[3] = {0.};
    double v[3] = {0.};
    double k[3] = {0.};
    double P1[3] = {0.};
    double P2[3] = {0.};
    double P3[3] = {0.};
    double kdotr[6] = {0.};
    double kdotP[3] = {0.};
    double xi[3] = {0.};
    double fonfs[3] = {0.};
    cmplx Aij[3] = {0.};
    double r12[3] = {0.};
    double r13[3] = {0.};
    double r23[3] = {0.};
    double r31[3] = {0.};
    double argS;
    cmplx tmp_r12p;
    cmplx tmp_r31p;
    cmplx tmp_r23p;

    cmplx tmp_r12c;
    cmplx tmp_r31c;
    cmplx tmp_r23c;
    double fi;
    double arg_ij;
    unsigned int s;
    unsigned int A_ind;
    cmplx tmp_X, tmp_Y, tmp_Z;
    cmplx tmp2_X, tmp2_Y, tmp2_Z;

    cmplx I(0.0, 1.0);

    cmplx Gs[6] = {0.};
    // order is (12, 23, 31, 21, 32, 13)
    // 21 is good
    // 31 in right spot but -1
    //
    unsigned int s_all[6] = {0, 1, 2, 1, 2, 0};

    cmplx *X = &wave[0];
    cmplx *Y = &wave[gb_in.N];
    cmplx *Z = &wave[2 * gb_in.N];

    // index of nearest Fourier bin
    double q = rint(gb_in.f0 * gb_in.T);
    double df = 2.0 * M_PI * (q / gb_in.T);
    double om = 2.0 * M_PI * gb_in.f0;
    double f;
    double omL, SomL;
    cmplx fctr, fctr2, fctr3, tdi2_factor;
    double omegaL;

    // get initial setup
    double cosiota = cos(gb_in.inc);
    double cosps = cos(2.0 * gb_in.psi);
    double sinps = sin(2.0 * gb_in.psi);
    double Aplus = gb_in.amp * (1.0 + cosiota * cosiota);
    double Across = -2.0 * gb_in.amp * cosiota;

    double sinth = sin(gb_in.theta);
    double costh = cos(gb_in.theta);
    double sinph = sin(gb_in.lam);
    double cosph = cos(gb_in.lam);

    u[0] = costh * cosph;
    u[1] = costh * sinph;
    u[2] = -sinth;

    v[0] = sinph;
    v[1] = -cosph;
    v[2] = 0.0;

    k[0] = -sinth * cosph;
    k[1] = -sinth * sinph;
    k[2] = -costh;

    get_eplus(eplus, u, v);
    get_ecross(ecross, u, v);

    cmplx DP = {Aplus * cosps, -Across * sinps};
    cmplx DC = {-Aplus * sinps, -Across * cosps};

    double delta_t_slow = gb_in.T / (double)gb_in.N;
    double t, xi_tmp;

    // construct slow part
    for (int i = threadIdx.x; i < gb_in.N; i += blockDim.x)
    {
        t = i * delta_t_slow;
        spacecraft(P1, P2, P3, t);

// reset sum terms
#pragma unroll
        for (int j = 0; j < 6; j += 1)
        {
            kdotr[j] = 0.0;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            kdotP[j] = 0.0;
            Aij[j] = 0.0;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            r12[j] = (P2[j] - P1[j]) / Larm;
            r13[j] = (P3[j] - P1[j]) / Larm;
            r31[j] = -r13[j];
            r23[j] = (P3[j] - P2[j]) / Larm;

            kdotr[0] += k[j] * r12[j];
            kdotr[1] += k[j] * r23[j];
            kdotr[2] += k[j] * r31[j];

            kdotP[0] += (k[j] * P1[j]) / Clight;
            kdotP[1] += (k[j] * P2[j]) / Clight;
            kdotP[2] += (k[j] * P3[j]) / Clight;

            // 12
        }

        kdotr[3] = -kdotr[0];
        kdotr[4] = -kdotr[1];
        kdotr[5] = -kdotr[2];

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            xi_tmp = t - kdotP[j];
            xi[j] = xi_tmp;
            fi = (gb_in.f0 + gb_in.fdot * xi_tmp + 1 / 2.0 * gb_in.fddot * (xi_tmp * xi_tmp));
            fonfs[j] = fi / fstar;
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            tmp_r12p = 0.0;
            tmp_r31p = 0.0;
            tmp_r23p = 0.0;

            tmp_r12c = 0.0;
            tmp_r31c = 0.0;
            tmp_r23c = 0.0;

#pragma unroll
            for (int k = 0; k < 3; k += 1)
            {
                tmp_r12p += eplus[j * 3 + k] * r12[k];
                tmp_r31p += eplus[j * 3 + k] * r31[k];
                tmp_r23p += eplus[j * 3 + k] * r23[k];

                tmp_r12c += ecross[j * 3 + k] * r12[k];
                tmp_r31c += ecross[j * 3 + k] * r31[k];
                tmp_r23c += ecross[j * 3 + k] * r23[k];
            }

            Aij[0] += (tmp_r12p * r12[j] * DP + tmp_r12c * r12[j] * DC);
            Aij[1] += (tmp_r23p * r23[j] * DP + tmp_r23c * r23[j] * DC);
            Aij[2] += (tmp_r31p * r31[j] * DP + tmp_r31c * r31[j] * DC);
        }

#pragma unroll
        for (int j = 0; j < 3; j += 1)
        {
            xi_tmp = xi[j];
            argS = (gb_in.phi0 + (om - df) * t + M_PI * gb_in.fdot * (xi_tmp * xi_tmp) + 1. / 3. * M_PI * gb_in.fddot * (xi_tmp * xi_tmp * xi_tmp));

            kdotP[j] = om * kdotP[j] - argS;
        }

#pragma unroll
        for (int j = 0; j < 6; j += 1)
        {

            s = s_all[j];
            A_ind = j % 3;
            arg_ij = 0.5 * fonfs[s] * (1 + kdotr[j]);

            Gs[j] = (0.25 * sin(arg_ij) / arg_ij * exp(-I * (arg_ij + kdotP[s])) * Aij[A_ind]);
        }

        f = (gb_in.f0 + gb_in.fdot * t + 1 / 2.0 * gb_in.fddot * (t * t));

        omL = f / fstar;
        SomL = sin(omL);
        fctr = gcmplx::exp(-I * omL);
        fctr2 = 4.0 * omL * SomL * fctr / gb_in.amp;

        // order is (12, 23, 31, 21, 32, 13)

        // Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        tmp_X = fctr2 * ((Gs[3] - Gs[2] + (Gs[0] - Gs[5]) * fctr));

        // tmp_X = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t);
        // tmp2_X = complex_type {tmp_X.real(), tmp_X.imag()};
        X[i] = tmp_X;

        tmp_Y = fctr2 * ((Gs[4] - Gs[0] + (Gs[1] - Gs[3]) * fctr));
        // tmp2_Y = complex_type {tmp_Y.real(), tmp_Y.imag()};
        Y[i] = tmp_Y;

        tmp_Z = fctr2 * ((Gs[5] - Gs[1] + (Gs[2] - Gs[4]) * fctr));
        // tmp_Z = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t);
        // tmp2_Z = complex_type {tmp_Z.real(), tmp_Z.imag()};
        Z[i] = tmp_Z;
    }

    __syncthreads();

    FFT().execute(reinterpret_cast<void *>(X));
    FFT().execute(reinterpret_cast<void *>(Y));
    FFT().execute(reinterpret_cast<void *>(Z));

    __syncthreads();

    for (int i = threadIdx.x; i < gb_in.N; i += blockDim.x)
    {
        X[i] *= gb_in.amp;
        Y[i] *= gb_in.amp;
        Z[i] *= gb_in.amp;
    }

    __syncthreads();

    cmplx tmp_switch;
    cmplx tmp1_switch;
    cmplx tmp2_switch;

    int N_over_2 = gb_in.N / 2;
    fctr3 = 0.5 * gb_in.T / (double)gb_in.N;

    // if (tdi2)
    //{
    //     omegaL = 2. * M_PI * f0_out * (Larm / Clight);
    //     tdi2_factor = 2.0 * I * sin(2. * omegaL) * exp(-2. * I * omegaL);
    //     fctr3 *= tdi2_factor;
    // }

    // fftshift in numpy and multiply by fctr3
    for (int i = threadIdx.x; i < N_over_2; i += blockDim.x)
    {
        // X
        tmp1_switch = X[i];
        tmp2_switch = X[N_over_2 + i];

        X[N_over_2 + i] = tmp1_switch * fctr3;
        X[i] = tmp2_switch * fctr3;

        // Y
        tmp1_switch = Y[i];
        tmp2_switch = Y[N_over_2 + i];

        Y[N_over_2 + i] = tmp1_switch * fctr3;
        Y[i] = tmp2_switch * fctr3;

        // Z
        tmp1_switch = Z[i];
        tmp2_switch = Z[N_over_2 + i];

        Z[N_over_2 + i] = tmp1_switch * fctr3;
        Z[i] = tmp2_switch * fctr3;
    }

    // convert to A, E, T (they sit in X,Y,Z)
    for (int i = threadIdx.x; i < gb_in.N; i += blockDim.x)
    {
        AET_from_XYZ_swap(&X[i], &Y[i], &Z[i]);
    }

    __syncthreads();

    double fmin = (q - ((double)gb_in.N) / 2.) / gb_in.T;
    *start_ind = (int)rint(fmin * gb_in.T);

    return;
}


template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_waveform(
    cmplx *tdi_out,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);
    cmplx *wave = (cmplx *)shared_mem;

    for (int bin_i = blockIdx.x; bin_i < num_bin_all; bin_i += gridDim.x)
    {

        build_single_waveform<FFT>(
            wave,
            &start_ind,
            amp[bin_i],
            f0[bin_i],
            fdot0[bin_i],
            fddot0[bin_i],
            phi0[bin_i],
            iota[bin_i],
            psi[bin_i],
            lam[bin_i],
            theta[bin_i],
            T,
            dt,
            N,
            bin_i);

        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            for (int j = 0; j < 3; j += 1)
            {
                tdi_out[(bin_i * 3 + j) * N + i] = wave[j * N + i];
            }
        }

        __syncthreads();

        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void simple_block_fft(InputInfo inputs)
{
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    auto shared_memory_size_mine = 3 * N * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << shared_memory_size_mine << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_waveform<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    get_waveform<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.tdi_out,
        inputs.amp,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.phi0,
        inputs.iota,
        inputs.psi,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
}

template <unsigned int Arch, unsigned int N>
struct simple_block_fft_functor
{
    void operator()(InputInfo inputs) { return simple_block_fft<Arch, N>(inputs); }
};
void SharedMemoryWaveComp(
    cmplx *tdi_out,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all)
{

    InputInfo inputs;
    inputs.tdi_out = tdi_out;
    inputs.amp = amp;
    inputs.f0 = f0;
    inputs.fdot0 = fdot0;
    inputs.fddot0 = fddot0;
    inputs.phi0 = phi0;
    inputs.iota = iota;
    inputs.psi = psi;
    inputs.lam = lam;
    inputs.theta = theta;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<simple_block_fft_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<simple_block_fft_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<simple_block_fft_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<simple_block_fft_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<simple_block_fft_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<simple_block_fft_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<simple_block_fft_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_ll(
    cmplx *d_h,
    cmplx *h_h,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    cmplx *wave = (cmplx *)shared_mem;
    cmplx *A = &wave[0];
    cmplx *E = &wave[N];

    cmplx *d_h_temp = &wave[3 * N];
    cmplx *h_h_temp = &d_h_temp[FFT::block_dim.x];
    cmplx tmp1, tmp2;
    int data_ind, noise_ind;

    int jj = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d_A, d_E, h_A, h_E;
    double n_A, n_E;
    for (int bin_i = blockIdx.x; bin_i < num_bin_all; bin_i += gridDim.x)
    {

        data_ind = data_index[bin_i];
        noise_ind = noise_index[bin_i];

        build_single_waveform<FFT>(
            wave,
            &start_ind,
            amp[bin_i],
            f0[bin_i],
            fdot0[bin_i],
            fddot0[bin_i],
            phi0[bin_i],
            iota[bin_i],
            psi[bin_i],
            lam[bin_i],
            theta[bin_i],
            T,
            dt,
            N,
            bin_i);

        __syncthreads();
        tmp1 = 0.0;
        tmp2 = 0.0;
        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            jj = i + start_ind - start_freq_ind;
            d_A = data_A[data_ind * data_length + jj];
            d_E = data_E[data_ind * data_length + jj];
            n_A = noise_A[noise_ind * data_length + jj];
            n_E = noise_E[noise_ind * data_length + jj];

            h_A = A[i];
            h_E = E[i];

            tmp1 += (gcmplx::conj(d_A) * h_A / n_A + gcmplx::conj(d_E) * h_E / n_E);

            tmp2 += (gcmplx::conj(h_A) * h_A / n_A + gcmplx::conj(h_E) * h_E / n_E);
            __syncthreads();
        }

        d_h_temp[tid] = tmp1;
        h_h_temp[tid] = tmp2;
        // if (((bin_i == 10) || (bin_i == 400))) printf("%d %d  %e %e %e %e \n", bin_i, tid, d_h_temp[tid].real(), d_h_temp[tid].imag(), h_h_temp[tid].real(), h_h_temp[tid].imag());
        __syncthreads();
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                d_h_temp[tid] += d_h_temp[tid + s];
                h_h_temp[tid] += h_h_temp[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            d_h[bin_i] = 4.0 * df * d_h_temp[0];
            h_h[bin_i] = 4.0 * df * h_h_temp[0];
        }
        __syncthreads();

        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void get_ll_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    // first is waveforms, second is d_h_temp and h_h_temp
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 2 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_ll<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    get_ll<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.d_h,
        inputs.h_h,
        inputs.data_A,
        inputs.data_E,
        inputs.noise_A,
        inputs.noise_E,
        inputs.data_index,
        inputs.noise_index,
        inputs.amp,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.phi0,
        inputs.iota,
        inputs.psi,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    if (inputs.do_synchronize)
    {
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
}

template <unsigned int Arch, unsigned int N>
struct get_ll_wrap_functor
{
    void operator()(InputInfo inputs) { return get_ll_wrap<Arch, N>(inputs); }
};

void SharedMemoryLikeComp(
    cmplx *d_h,
    cmplx *h_h,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize)
{

    InputInfo inputs;
    inputs.d_h = d_h;
    inputs.h_h = h_h;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
    ;
    inputs.data_index = data_index;
    inputs.noise_index = noise_index;
    inputs.amp = amp;
    inputs.f0 = f0;
    inputs.fdot0 = fdot0;
    inputs.fddot0 = fddot0;
    inputs.phi0 = phi0;
    inputs.iota = iota;
    inputs.psi = psi;
    inputs.lam = lam;
    inputs.theta = theta;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<get_ll_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<get_ll_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<get_ll_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<get_ll_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<get_ll_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<get_ll_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<get_ll_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_swap_ll_diff(
    cmplx *d_h_remove,
    cmplx *d_h_add,
    cmplx *remove_remove,
    cmplx *add_add,
    cmplx *add_remove,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *amp_add,
    double *f0_add,
    double *fdot0_add,
    double *fddot0_add,
    double *phi0_add,
    double *iota_add,
    double *psi_add,
    double *lam_add,
    double *theta_add,
    double *amp_remove,
    double *f0_remove,
    double *fdot0_remove,
    double *fddot0_remove,
    double *phi0_remove,
    double *iota_remove,
    double *psi_remove,
    double *lam_remove,
    double *theta_remove,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length)
{
    using complex_type = cmplx;

    unsigned int start_ind_add = 0;
    unsigned int start_ind_remove = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    cmplx *wave_add = (cmplx *)shared_mem;
    cmplx *A_add = &wave_add[0];
    cmplx *E_add = &wave_add[N];

    cmplx *wave_remove = &wave_add[3 * N];
    cmplx *A_remove = &wave_remove[0];
    cmplx *E_remove = &wave_remove[N];

    cmplx *d_h_remove_arr = &wave_remove[3 * N];
    cmplx *d_h_add_arr = &d_h_remove_arr[FFT::block_dim.x];
    ;
    cmplx *remove_remove_arr = &d_h_add_arr[FFT::block_dim.x];
    ;
    cmplx *add_add_arr = &remove_remove_arr[FFT::block_dim.x];
    ;
    cmplx *add_remove_arr = &add_add_arr[FFT::block_dim.x];

    cmplx d_h_remove_temp = 0.0;
    cmplx d_h_add_temp = 0.0;
    cmplx remove_remove_temp = 0.0;
    cmplx add_add_temp = 0.0;
    cmplx add_remove_temp = 0.0;

    int data_ind, noise_ind;

    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_add_lower;
    int total_i_vals;

    cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
    int real_ind, real_ind_add, real_ind_remove;

    int jj = 0;
    int j = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d_A, d_E;
    double n_A, n_E;
    for (int bin_i = blockIdx.x; bin_i < num_bin_all; bin_i += gridDim.x)
    {

        data_ind = data_index[bin_i];
        noise_ind = noise_index[bin_i];

        d_h_remove_temp = 0.0;
        d_h_add_temp = 0.0;
        remove_remove_temp = 0.0;
        add_add_temp = 0.0;
        add_remove_temp = 0.0;

        build_single_waveform<FFT>(
            wave_add,
            &start_ind_add,
            amp_add[bin_i],
            f0_add[bin_i],
            fdot0_add[bin_i],
            fddot0_add[bin_i],
            phi0_add[bin_i],
            iota_add[bin_i],
            psi_add[bin_i],
            lam_add[bin_i],
            theta_add[bin_i],
            T,
            dt,
            N,
            bin_i);
        __syncthreads();
        build_single_waveform<FFT>(
            wave_remove,
            &start_ind_remove,
            amp_remove[bin_i],
            f0_remove[bin_i],
            fdot0_remove[bin_i],
            fddot0_remove[bin_i],
            phi0_remove[bin_i],
            iota_remove[bin_i],
            psi_remove[bin_i],
            lam_remove[bin_i],
            theta_remove[bin_i],
            T,
            dt,
            N,
            bin_i);
        __syncthreads();

        // subtract start_freq_ind to find index into subarray
        if (start_ind_remove <= start_ind_add)
        {
            lower_start_ind = start_ind_remove - start_freq_ind;
            upper_end_ind = start_ind_add - start_freq_ind + N;

            upper_start_ind = start_ind_add - start_freq_ind;
            lower_end_ind = start_ind_remove - start_freq_ind + N;

            is_add_lower = false;
        }
        else
        {
            lower_start_ind = start_ind_add - start_freq_ind;
            upper_end_ind = start_ind_remove - start_freq_ind + N;

            upper_start_ind = start_ind_remove - start_freq_ind;
            lower_end_ind = start_ind_add - start_freq_ind + N;

            is_add_lower = true;
        }
        total_i_vals = upper_end_ind - lower_start_ind;
        // ECK %d \n", total_i_vals);
        __syncthreads();
        if (total_i_vals < 2 * N)
        {
            for (int i = threadIdx.x;
                 i < total_i_vals;
                 i += blockDim.x)
            {

                j = lower_start_ind + i;

                n_A = noise_A[noise_ind * data_length + j];
                n_E = noise_E[noise_ind * data_length + j];

                d_A = data_A[data_ind * data_length + j];
                d_E = data_E[data_ind * data_length + j];

                // if ((bin_i == 0)){
                // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, noise_ind * data_length + j, data_ind * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
                // }

                // if ((bin_i == 0)) printf("%d %e %e %e %e %e %e %e %e %e %e %e %e \n", tid, d_h_remove_temp.real(), d_h_remove_temp.imag(), d_h_add_temp.real(), d_h_add_temp.imag(), add_add_temp.real(), add_add_temp.imag(), remove_remove_temp.real(), remove_remove_temp.imag(), add_remove_temp.real(), add_remove_temp.imag());

                if (j < upper_start_ind)
                {
                    real_ind = i;
                    if (is_add_lower)
                    {

                        h_A = A_add[real_ind];
                        h_E = E_add[real_ind];

                        // get <d|h> term
                        d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                        d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

                        // <h|h>
                        add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                        add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
                    }
                    else
                    {
                        h_A = A_remove[real_ind];
                        h_E = E_remove[real_ind];

                        // get <d|h> term
                        d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                        d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

                        // <h|h>
                        remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                        remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

                        // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                    }
                }
                else if (j >= lower_end_ind)
                {
                    real_ind = j - upper_start_ind;
                    if (!is_add_lower)
                    {

                        h_A_add = A_add[real_ind];
                        h_E_add = E_add[real_ind];

                        // get <d|h> term
                        d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                        d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                        // <h|h>
                        add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                        add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;
                    }
                    else
                    {
                        h_A_remove = A_remove[real_ind];
                        h_E_remove = E_remove[real_ind];

                        // get <d|h> term
                        d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                        d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                        // <h|h>
                        remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                        remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;
                    }
                }
                else // this is where the signals overlap
                {
                    if (is_add_lower)
                    {
                        real_ind_add = i;
                    }
                    else
                    {
                        real_ind_add = j - upper_start_ind;
                    }

                    h_A_add = A_add[real_ind_add];
                    h_E_add = E_add[real_ind_add];

                    // get <d|h> term
                    d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                    d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                    // <h|h>
                    add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                    add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;

                    if (!is_add_lower)
                    {
                        real_ind_remove = i;
                    }
                    else
                    {
                        real_ind_remove = j - upper_start_ind;
                    }

                    h_A_remove = A_remove[real_ind_remove];
                    h_E_remove = E_remove[real_ind_remove];

                    // get <d|h> term
                    d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                    d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                    // <h|h>
                    remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                    remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;

                    add_remove_temp += gcmplx::conj(h_A_remove) * h_A_add / n_A;
                    add_remove_temp += gcmplx::conj(h_E_remove) * h_E_add / n_E;
                }
            }
        }
        else
        {
            for (int i = threadIdx.x;
                 i < N;
                 i += blockDim.x)
            {

                j = start_ind_remove + i - start_freq_ind;

                n_A = noise_A[noise_ind * data_length + j];
                n_E = noise_E[noise_ind * data_length + j];

                d_A = data_A[data_ind * data_length + j];
                d_E = data_E[data_ind * data_length + j];

                // if ((bin_i == num_bin - 1))printf("CHECK remove: %d %e %e  \n", i, n_A, d_A.real());
                //  calculate h term
                h_A = A_remove[i];
                h_E = E_remove[i];

                // get <d|h> term
                d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

                // <h|h>
                remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;
            }

            for (int i = threadIdx.x;
                 i < N;
                 i += blockDim.x)
            {

                j = start_ind_add + i - start_freq_ind;

                n_A = noise_A[noise_ind * data_length + j];
                n_E = noise_E[noise_ind * data_length + j];

                d_A = data_A[data_ind * data_length + j];
                d_E = data_E[data_ind * data_length + j];

                // if ((bin_i == 0))printf("CHECK add: %d %d %e %e %e %e  \n", i, j, n_A, d_A.real(), n_E, d_E.real());
                //  calculate h term
                h_A = A_add[i];
                h_E = E_add[i];

                // get <d|h> term
                d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

                // <h|h>
                add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
            }
        }

        __syncthreads();
        d_h_remove_arr[tid] = d_h_remove_temp;

        d_h_add_arr[tid] = d_h_add_temp;
        add_add_arr[tid] = add_add_temp;
        remove_remove_arr[tid] = remove_remove_temp;
        add_remove_arr[tid] = add_remove_temp;
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                d_h_remove_arr[tid] += d_h_remove_arr[tid + s];
                d_h_add_arr[tid] += d_h_add_arr[tid + s];
                add_add_arr[tid] += add_add_arr[tid + s];
                remove_remove_arr[tid] += remove_remove_arr[tid + s];
                add_remove_arr[tid] += add_remove_arr[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            d_h_remove[bin_i] = 4.0 * df * d_h_remove_arr[0];
            d_h_add[bin_i] = 4.0 * df * d_h_add_arr[0];
            add_add[bin_i] = 4.0 * df * add_add_arr[0];
            remove_remove[bin_i] = 4.0 * df * remove_remove_arr[0];
            add_remove[bin_i] = 4.0 * df * add_remove_arr[0];
        }
        __syncthreads();

        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void get_swap_ll_diff_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    // first is waveforms, second is d_h_temp and h_h_temp
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 3 * N * sizeof(cmplx) + 5 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_swap_ll_diff<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    get_swap_ll_diff<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.d_h_remove,
        inputs.d_h_add,
        inputs.remove_remove,
        inputs.add_add,
        inputs.add_remove,
        inputs.data_A,
        inputs.data_E,
        inputs.noise_A,
        inputs.noise_E,
        inputs.data_index,
        inputs.noise_index,
        inputs.amp_add,
        inputs.f0_add,
        inputs.fdot0_add,
        inputs.fddot0_add,
        inputs.phi0_add,
        inputs.iota_add,
        inputs.psi_add,
        inputs.lam_add,
        inputs.theta_add,
        inputs.amp_remove,
        inputs.f0_remove,
        inputs.fdot0_remove,
        inputs.fddot0_remove,
        inputs.phi0_remove,
        inputs.iota_remove,
        inputs.psi_remove,
        inputs.lam_remove,
        inputs.theta_remove,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    if (inputs.do_synchronize)
    {
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
}

template <unsigned int Arch, unsigned int N>
struct get_swap_ll_diff_wrap_functor
{
    void operator()(InputInfo inputs) { return get_swap_ll_diff_wrap<Arch, N>(inputs); }
};

void SharedMemorySwapLikeComp(
    cmplx *d_h_remove,
    cmplx *d_h_add,
    cmplx *remove_remove,
    cmplx *add_add,
    cmplx *add_remove,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *amp_add,
    double *f0_add,
    double *fdot0_add,
    double *fddot0_add,
    double *phi0_add,
    double *iota_add,
    double *psi_add,
    double *lam_add,
    double *theta_add,
    double *amp_remove,
    double *f0_remove,
    double *fdot0_remove,
    double *fddot0_remove,
    double *phi0_remove,
    double *iota_remove,
    double *psi_remove,
    double *lam_remove,
    double *theta_remove,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize)
{

    InputInfo inputs;
    inputs.d_h_remove = d_h_remove;
    inputs.d_h_add = d_h_add;
    inputs.remove_remove = remove_remove;
    inputs.add_add = add_add;
    inputs.add_remove = add_remove;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
    ;
    inputs.data_index = data_index;
    inputs.noise_index = noise_index;
    inputs.amp_add = amp_add;
    inputs.f0_add = f0_add;
    inputs.fdot0_add = fdot0_add;
    inputs.fddot0_add = fddot0_add;
    inputs.phi0_add = phi0_add;
    inputs.iota_add = iota_add;
    inputs.psi_add = psi_add;
    inputs.lam_add = lam_add;
    inputs.theta_add = theta_add;
    inputs.amp_remove = amp_remove;
    inputs.f0_remove = f0_remove;
    inputs.fdot0_remove = fdot0_remove;
    inputs.fddot0_remove = fddot0_remove;
    inputs.phi0_remove = phi0_remove;
    inputs.iota_remove = iota_remove;
    inputs.psi_remove = psi_remove;
    inputs.lam_remove = lam_remove;
    inputs.theta_remove = theta_remove;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<get_swap_ll_diff_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_chi_squared(
    cmplx *h1_h1,
    cmplx *h2_h2,
    cmplx *h1_h2,
    double *noise_A,
    double *noise_E,
    int *noise_index,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length)
{
    using complex_type = cmplx;

    unsigned int start_ind_1 = 0;
    unsigned int start_ind_2 = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    cmplx *wave_1 = (cmplx *)shared_mem;
    cmplx *A_1 = &wave_1[0];
    cmplx *E_1 = &wave_1[N];

    cmplx *wave_2 = &wave_1[3 * N];
    cmplx *A_2 = &wave_2[0];
    cmplx *E_2 = &wave_2[N];

    cmplx *h1_h1_arr = &wave_2[3 * N];
    ;
    cmplx *h2_h2_arr = &h1_h1_arr[FFT::block_dim.x];
    ;
    cmplx *h1_h2_arr = &h2_h2_arr[FFT::block_dim.x];

    cmplx h1_h1_temp = 0.0;
    cmplx h2_h2_temp = 0.0;
    cmplx h1_h2_temp = 0.0;

    int noise_ind;

    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_1_lower;
    int total_i_vals;

    cmplx h_A, h_E, h_A_1, h_E_1, h_A_2, h_E_2;
    int real_ind, real_ind_1, real_ind_2;

    int jj = 0;
    int j = 0;
    int last_row_end;
    int output_ind, output_ind2;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);
    double n_A, n_E;
    for (int bin_i = blockIdx.x; bin_i < num_bin_all; bin_i += gridDim.x)
    {
        for (int bin_j = blockIdx.y + bin_i + 1; bin_j < num_bin_all; bin_j += gridDim.y)
        {
            h1_h1_temp = 0.0;
            h1_h2_temp = 0.0;
            h2_h2_temp = 0.0;
            noise_ind = noise_index[bin_i]; // must be the same

            // get index into upper triangular array (without diagonal)
            last_row_end = bin_i * (bin_i + 1) / 2;
            output_ind = num_bin_all * bin_i + bin_j - int(((bin_i + 2) * (bin_i + 1)) / 2);
            // output_ind2 = last_row_end + bin_j; // INDEX SO DOES NOT HAVE +1 

            build_single_waveform<FFT>(
                wave_1,
                &start_ind_1,
                amp[bin_i],
                f0[bin_i],
                fdot0[bin_i],
                fddot0[bin_i],
                phi0[bin_i],
                iota[bin_i],
                psi[bin_i],
                lam[bin_i],
                theta[bin_i],
                T,
                dt,
                N,
                bin_i);
            __syncthreads();
            build_single_waveform<FFT>(
                wave_2,
                &start_ind_2,
                amp[bin_j],
                f0[bin_j],
                fdot0[bin_j],
                fddot0[bin_j],
                phi0[bin_j],
                iota[bin_j],
                psi[bin_j],
                lam[bin_j],
                theta[bin_j],
                T,
                dt,
                N,
                bin_i);
            __syncthreads();

            // subtract start_freq_ind to find index into subarray
            if (start_ind_2 <= start_ind_1)
            {
                lower_start_ind = start_ind_2 - start_freq_ind;
                upper_end_ind = start_ind_1 - start_freq_ind + N;

                upper_start_ind = start_ind_1 - start_freq_ind;
                lower_end_ind = start_ind_2 - start_freq_ind + N;

                is_1_lower = false;
            }
            else
            {
                lower_start_ind = start_ind_1 - start_freq_ind;
                upper_end_ind = start_ind_2 - start_freq_ind + N;

                upper_start_ind = start_ind_2 - start_freq_ind;
                lower_end_ind = start_ind_1 - start_freq_ind + N;

                is_1_lower = true;
            }
            total_i_vals = upper_end_ind - lower_start_ind;

            __syncthreads();
            if (total_i_vals < 2 * N)
            {
                for (int i = threadIdx.x;
                    i < total_i_vals;
                    i += blockDim.x)
                {

                    j = lower_start_ind + i;

                    n_A = noise_A[noise_ind * data_length + j];
                    n_E = noise_E[noise_ind * data_length + j];

                    // if ((bin_i == 0)){
                    // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, noise_ind * data_length + j, data_ind * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
                    // }

                    // if ((bin_i == 0)) printf("%d %e %e %e %e %e %e %e %e %e %e %e %e \n", tid, d_h_2_temp.real(), d_h_2_temp.imag(), d_h_1_temp.real(), d_h_1_temp.imag(), h2_h2_temp.real(), h2_h2_temp.imag(), h1_h1_temp.real(), h1_h1_temp.imag(), h1_h2_temp.real(), h1_h2_temp.imag());

                    if (j < upper_start_ind)
                    {
                        real_ind = i;
                        if (is_1_lower)
                        {

                            h_A = A_1[real_ind];
                            h_E = E_1[real_ind];

                            // <h|h>
                            h2_h2_temp += gcmplx::conj(h_A) * h_A / n_A;
                            h2_h2_temp += gcmplx::conj(h_E) * h_E / n_E;
                        }
                        else
                        {
                            h_A = A_2[real_ind];
                            h_E = E_2[real_ind];

                            // <h|h>
                            h1_h1_temp += gcmplx::conj(h_A) * h_A / n_A;
                            h1_h1_temp += gcmplx::conj(h_E) * h_E / n_E;

                            // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                        }
                    }
                    else if (j >= lower_end_ind)
                    {
                        real_ind = j - upper_start_ind;
                        if (!is_1_lower)
                        {

                            h_A_1 = A_1[real_ind];
                            h_E_1 = E_1[real_ind];

                            // <h|h>
                            h2_h2_temp += gcmplx::conj(h_A_1) * h_A_1 / n_A;
                            h2_h2_temp += gcmplx::conj(h_E_1) * h_E_1 / n_E;
                        }
                        else
                        {
                            h_A_2 = A_2[real_ind];
                            h_E_2 = E_2[real_ind];

                            // <h|h>
                            h1_h1_temp += gcmplx::conj(h_A_2) * h_A_2 / n_A;
                            h1_h1_temp += gcmplx::conj(h_E_2) * h_E_2 / n_E;
                        }
                    }
                    else // this is where the signals overlap
                    {
                        if (is_1_lower)
                        {
                            real_ind_1 = i;
                        }
                        else
                        {
                            real_ind_1 = j - upper_start_ind;
                        }

                        h_A_1 = A_1[real_ind_1];
                        h_E_1 = E_1[real_ind_1];

                        // <h|h>
                        h2_h2_temp += gcmplx::conj(h_A_1) * h_A_1 / n_A;
                        h2_h2_temp += gcmplx::conj(h_E_1) * h_E_1 / n_E;

                        if (!is_1_lower)
                        {
                            real_ind_2 = i;
                        }
                        else
                        {
                            real_ind_2 = j - upper_start_ind;
                        }

                        h_A_2 = A_2[real_ind_2];
                        h_E_2 = E_2[real_ind_2];

                        // <h|h>
                        h1_h1_temp += gcmplx::conj(h_A_2) * h_A_2 / n_A;
                        h1_h1_temp += gcmplx::conj(h_E_2) * h_E_2 / n_E;

                        h1_h2_temp += gcmplx::conj(h_A_2) * h_A_1 / n_A;
                        h1_h2_temp += gcmplx::conj(h_E_2) * h_E_1 / n_E;
                    }
                }
            }
            else
            {
                if (tid == 0)
                {
                    h2_h2[output_ind] = 1e9;
                    h1_h1[output_ind] = 1e9;
                    h1_h2[output_ind] = 0.0;
                }
                __syncthreads();
                continue;
            }

            __syncthreads();
            h2_h2_arr[tid] = h2_h2_temp;
            h1_h1_arr[tid] = h1_h1_temp;
            h1_h2_arr[tid] = h1_h2_temp;
            __syncthreads();

            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    h2_h2_arr[tid] += h2_h2_arr[tid + s];
                    h1_h1_arr[tid] += h1_h1_arr[tid + s];
                    h1_h2_arr[tid] += h1_h2_arr[tid + s];
                    // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                    // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
                }
                __syncthreads();
            }
            __syncthreads();

            if (tid == 0)
            {
                h2_h2[output_ind] = 4.0 * df * h2_h2_arr[0];
                h1_h1[output_ind] = 4.0 * df * h1_h1_arr[0];
                h1_h2[output_ind] = 4.0 * df * h1_h2_arr[0];
            }
            __syncthreads();

        }
        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void get_chi_squared_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    // first is waveforms, second is d_h_temp and h_h_temp
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 3 * N * sizeof(cmplx) + 3 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_chi_squared<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    int grid_len_y;

    if (inputs.num_bin_all > 65000)
    {
        grid_len_y = 65000;
    }
    else
    {
        grid_len_y = inputs.num_bin_all;
    }
    int grid_len_x = inputs.num_bin_all;

    dim3 grid(1, 1); // grid(grid_len_x, grid_len_y);
    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    get_chi_squared<FFT><<<grid, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.h1_h1,
        inputs.h2_h2,
        inputs.h1_h2,
        inputs.noise_A,
        inputs.noise_E,
        inputs.noise_index,
        inputs.amp,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.phi0,
        inputs.iota,
        inputs.psi,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    if (inputs.do_synchronize)
    {
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
}

template <unsigned int Arch, unsigned int N>
struct get_chi_squared_wrap_functor
{
    void operator()(InputInfo inputs) { return get_chi_squared_wrap<Arch, N>(inputs); }
};

void SharedMemoryChiSquaredComp(
    cmplx *h1_h1,
    cmplx *h2_h2,
    cmplx *h1_h2,
    double *noise_A,
    double *noise_E,
    int *noise_index,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize)
{

    InputInfo inputs;
    inputs.h1_h1 = h1_h1;
    inputs.h2_h2 = h2_h2;
    inputs.h1_h2 = h1_h2;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
    ;

    inputs.noise_index = noise_index;
    inputs.amp = amp;
    inputs.f0 = f0;
    inputs.fdot0 = fdot0;
    inputs.fddot0 = fddot0;
    inputs.phi0 = phi0;
    inputs.iota = iota;
    inputs.psi = psi;
    inputs.lam = lam;
    inputs.theta = theta;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<get_chi_squared_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<get_chi_squared_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<get_chi_squared_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<get_chi_squared_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<get_chi_squared_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<get_chi_squared_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<get_chi_squared_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

// Add functionality for proper summation in the kernel
#ifdef __CUDACC__
CUDA_DEVICE
double atomicAddDouble(double *address, double val)
{
    unsigned long long *address_as_ull =
        (unsigned long long *)address;
    unsigned long long old = *address_as_ull, assumed;

    do
    {
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
CUDA_DEVICE
void atomicAddComplex(cmplx *a, cmplx b)
{
    // transform the addresses of real and imag. parts to double pointers
    double *x = (double *)a;
    double *y = x + 1;
    // use atomicAdd for double variables

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

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void generate_global_template(
    cmplx *template_A,
    cmplx *template_E,
    int *template_index,
    double *factors,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    int tid = threadIdx.x;
    cmplx *wave = (cmplx *)shared_mem;
    cmplx *A = &wave[0];
    cmplx *E = &wave[N];

    int template_ind;
    double factor;

    int jj = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d_A, d_E, h_A, h_E, n_A, n_E;
    for (int bin_i = blockIdx.x; bin_i < num_bin_all; bin_i += gridDim.x)
    {

        template_ind = template_index[bin_i];
        factor = factors[bin_i];

        build_single_waveform<FFT>(
            wave,
            &start_ind,
            amp[bin_i],
            f0[bin_i],
            fdot0[bin_i],
            fddot0[bin_i],
            phi0[bin_i],
            iota[bin_i],
            psi[bin_i],
            lam[bin_i],
            theta[bin_i],
            T,
            dt,
            N,
            bin_i);

        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            jj = i + start_ind - start_freq_ind;
            atomicAddComplex(&template_A[template_ind * data_length + jj], factor * A[i]);
            atomicAddComplex(&template_E[template_ind * data_length + jj], factor * E[i]);
        }
        __syncthreads();
    }
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void generate_global_template_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    // first is waveforms, second is d_h_temp and h_h_temp
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        generate_global_template<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    generate_global_template<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.data_A,
        inputs.data_E,
        inputs.data_index,
        inputs.factors,
        inputs.amp,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.phi0,
        inputs.iota,
        inputs.psi,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    if (inputs.do_synchronize)
    {
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
}

template <unsigned int Arch, unsigned int N>
struct generate_global_template_wrap_functor
{
    void operator()(InputInfo inputs) { return generate_global_template_wrap<Arch, N>(inputs); }
};

void SharedMemoryGenerateGlobal(
    cmplx *data_A,
    cmplx *data_E,
    int *data_index,
    double *factors,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize)
{

    InputInfo inputs;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.data_index = data_index;
    inputs.factors = factors;
    inputs.amp = amp;
    inputs.f0 = f0;
    inputs.fdot0 = fdot0;
    inputs.fddot0 = fddot0;
    inputs.phi0 = phi0;
    inputs.iota = iota;
    inputs.psi = psi;
    inputs.lam = lam;
    inputs.theta = theta;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<generate_global_template_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<generate_global_template_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<generate_global_template_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<generate_global_template_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<generate_global_template_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<generate_global_template_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<generate_global_template_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

#define NUM_THREADS_LIKE 64

__global__ void specialty_piece_wise_likelihoods(
    double *lnL,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    int *start_inds,
    int *lengths,
    double df,
    int num_parts,
    int start_freq_ind,
    int data_length)
{
    using complex_type = cmplx;

    int tid = threadIdx.x;
    __shared__ double lnL_tmp_for_sum[NUM_THREADS_LIKE];

    for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
    {
        lnL_tmp_for_sum[i] = 0.0;
    }
    __syncthreads();

    double tmp1;
    int data_ind, noise_ind, start_ind, length;

    int jj = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d_A, d_E, h_A, h_E;

    double n_A, n_E;
    for (int part_i = blockIdx.x; part_i < num_parts; part_i += gridDim.x)
    {

        data_ind = data_index[part_i];
        noise_ind = noise_index[part_i];
        start_ind = start_inds[part_i];
        length = lengths[part_i];

        tmp1 = 0.0;
        for (int i = threadIdx.x; i < length; i += blockDim.x)
        {
            jj = i + start_ind - start_freq_ind;
            d_A = data_A[data_ind * data_length + jj];
            d_E = data_E[data_ind * data_length + jj];
            n_A = noise_A[noise_ind * data_length + jj];
            n_E = noise_E[noise_ind * data_length + jj];

            // if (part_i == 0)
            //{
            //     printf("check vals %d %d %d %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", i, jj, start_ind, part_i, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E, df);
            // }
            tmp1 += (gcmplx::conj(d_A) * d_A / n_A + gcmplx::conj(d_E) * d_E / n_E).real();
        }
        __syncthreads();
        lnL_tmp_for_sum[tid] = tmp1;

        __syncthreads();
        if (tid == 0)
        {
            lnL[part_i] = -1. / 2. * (4.0 * df * lnL_tmp_for_sum[0]);
        }

        __syncthreads();
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                lnL_tmp_for_sum[tid] += lnL_tmp_for_sum[tid + s];
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            lnL[part_i] = -1. / 2. * (4.0 * df * lnL_tmp_for_sum[0]);
        }
        __syncthreads();

        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

void specialty_piece_wise_likelihoods_wrap(
    double *lnL,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    int *start_inds,
    int *lengths,
    double df,
    int num_parts,
    int start_freq_ind,
    int data_length,
    bool do_synchronize)
{
    if (num_parts == 0)
    {
        printf("num_parts is 0\n");
        return;
    }
    specialty_piece_wise_likelihoods<<<num_parts, NUM_THREADS_LIKE>>>(
        lnL,
        data_A,
        data_E,
        noise_A,
        noise_E,
        data_index,
        noise_index,
        start_inds,
        lengths,
        df,
        num_parts,
        start_freq_ind,
        data_length);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    if (do_synchronize)
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

// template <class FFT>
// __launch_bounds__(FFT::max_threads_per_block) __global__ void make_move(
//     DataPackage *data,
//     BandPackage *band_info,
//     GalacticBinaryParams *params_curr,
//     GalacticBinaryParams *params_prop,
//     MCMCInfo *mcmc_info
// )
// {
//     using complex_type = cmplx;

//     unsigned int start_ind = 0;

//     extern __shared__ unsigned char shared_mem[];

//     // auto this_block_data = tdi_out
//     //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

//     double df = data->df;
//     int tid = threadIdx.x;
//     unsigned int start_ind_add = 0;
//     unsigned int start_ind_remove = 0;

//     extern __shared__ unsigned char shared_mem[];

//     // auto this_block_data = tdi_out
//     //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

//     cmplx *wave_add = (cmplx *)shared_mem;
//     cmplx *A_add = &wave_add[0];
//     cmplx *E_add = &wave_add[params_curr->N];

//     cmplx *wave_remove = &wave_add[2 * params_curr->N];
//     cmplx *A_remove = &wave_remove[0];
//     cmplx *E_remove = &wave_remove[params_curr->N];

//     cmplx *d_h_remove_arr = &wave_remove[2 * params_curr->N];
//     cmplx *d_h_add_arr = &d_h_remove_arr[FFT::block_dim.x];
//     ;
//     cmplx *remove_remove_arr = &d_h_add_arr[FFT::block_dim.x];
//     ;
//     cmplx *add_add_arr = &remove_remove_arr[FFT::block_dim.x];
//     ;
//     cmplx *add_remove_arr = &add_add_arr[FFT::block_dim.x];

//     int this_band_start_index, this_band_length, j, k;
//     int this_band_start_bin_ind, this_band_num_bin;
//     int this_band_data_index, this_band_noise_index;
//     double this_binary_inv_temp;

//     double amp_prop, f0_prop, fdot0_prop, fddot0_prop, phi0_prop, iota_prop, psi_prop, lam_prop, theta_prop;
//     double amp_curr, f0_curr, fdot0_curr, fddot0_curr, phi0_curr, iota_curr, psi_curr, lam_curr, theta_curr;

//     int current_binary_start_index, base_index;
//     double ll_diff, lp_diff;
//     double prior_curr, prior_prop, factors, lnpdiff, random_val;
//     bool accept;

//     cmplx d_h_remove_temp = 0.0;
//     cmplx d_h_add_temp = 0.0;
//     cmplx remove_remove_temp = 0.0;
//     cmplx add_add_temp = 0.0;
//     cmplx add_remove_temp = 0.0;

//     int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
//     bool is_add_lower;
//     int total_i_vals;

//     cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
//     int real_ind, real_ind_add, real_ind_remove;
//     cmplx d_A, d_E;
//     double n_A, n_E;
//     double opt_snr, det_snr;

//     for (int band_i = blockIdx.x; band_i < band_info->num_bands; band_i += gridDim.x)
//     {
//         this_band_start_index = band_info->band_start_data_ind[band_i]; // overall index to which binary
//         this_band_length = band_info->band_data_lengths[band_i];
//         this_band_start_bin_ind = band_info->band_start_bin_ind[band_i];
//         this_band_num_bin = band_info->band_num_bins[band_i];
//         this_band_data_index = band_info->data_index[band_i];
//         this_band_noise_index = band_info->noise_index[band_i];

//         for (int bin_i = 0; bin_i < this_band_num_bin; bin_i += 1)
//         {
//             current_binary_start_index = this_band_start_bin_ind + bin_i;

//             this_binary_inv_temp = mcmc_info->band_inv_temperatures_all[current_binary_start_index];

//             prior_curr = mcmc_info->prior_all_curr[current_binary_start_index];
//             prior_prop = mcmc_info->prior_all_prop[current_binary_start_index];

//             lp_diff = prior_prop - prior_curr;
//             factors = mcmc_info->factors_all[current_binary_start_index];
//             random_val = mcmc_info->random_val_all[current_binary_start_index];

//             // get the parameters to add and remove
//             amp_curr = params_curr->amp[current_binary_start_index];
//             f0_curr = params_curr->f0_ms[current_binary_start_index];
//             fdot0_curr = params_curr->fdot0[current_binary_start_index];
//             phi0_curr = params_curr->phi0[current_binary_start_index];
//             iota_curr = params_curr->iota[current_binary_start_index];
//             psi_curr = params_curr->psi[current_binary_start_index];
//             lam_curr = params_curr->lam[current_binary_start_index];
//             theta_curr = params_curr->theta[current_binary_start_index];

//             amp_prop = params_prop->amp[current_binary_start_index];
//             f0_prop = params_prop->f0[current_binary_start_index];
//             fdot0_prop = params_prop->fdot0[current_binary_start_index];
//             fddot0_prop = params_prop->fddot0[current_binary_start_index];
//             phi0_prop = params_prop->phi0[current_binary_start_index];
//             iota_prop = params_prop->iota[current_binary_start_index];
//             psi_prop = params_prop->psi[current_binary_start_index];
//             lam_prop = params_prop->lam[current_binary_start_index];
//             theta_prop = params_prop->theta[current_binary_start_index];

//             __syncthreads();
//             build_single_waveform<FFT>(
//                 wave_remove,
//                 &start_ind_remove,
//                 amp_curr,
//                 f0_curr,
//                 fdot0_curr,
//                 fddot0_curr,
//                 phi0_curr,
//                 iota_curr,
//                 psi_curr,
//                 lam_curr,
//                 theta_curr,
//                 params_curr->T,
//                 params_curr->dt,
//                 params_curr->N,
//                 bin_i);
//             __syncthreads();
//             build_single_waveform<FFT>(
//                 wave_add,
//                 &start_ind_add,
//                 amp_prop,
//                 f0_prop,
//                 fdot0_prop,
//                 fddot0_prop,
//                 phi0_prop,
//                 iota_prop,
//                 psi_prop,
//                 lam_prop,
//                 theta_prop,
//                 params_curr->T,
//                 params_curr->dt,
//                 params_curr->N,
//                 bin_i);
//             __syncthreads();

//             // subtract start_freq_ind to find index into subarray
//             if (start_ind_remove <= start_ind_add)
//             {
//                 lower_start_ind = start_ind_remove - params_curr->start_freq_ind;
//                 upper_end_ind = start_ind_add - params_curr->start_freq_ind + params_curr->N;

//                 upper_start_ind = start_ind_add - params_curr->start_freq_ind;
//                 lower_end_ind = start_ind_remove - params_curr->start_freq_ind + params_curr->N;

//                 is_add_lower = false;
//             }
//             else
//             {
//                 lower_start_ind = start_ind_add - params_curr->start_freq_ind;
//                 upper_end_ind = start_ind_remove - params_curr->start_freq_ind + params_curr->N;

//                 upper_start_ind = start_ind_remove - params_curr->start_freq_ind;
//                 lower_end_ind = start_ind_add - params_curr->start_freq_ind + params_curr->N;

//                 is_add_lower = true;
//             }
//             total_i_vals = upper_end_ind - lower_start_ind;
//             // ECK %d \n", total_i_vals);
//             __syncthreads();
//             d_h_remove_temp = 0.0;
//             d_h_add_temp = 0.0;
//             remove_remove_temp = 0.0;
//             add_add_temp = 0.0;
//             add_remove_temp = 0.0;

//             if (total_i_vals < 2 * params_curr->N)
//             {
//                 for (int i = threadIdx.x;
//                      i < total_i_vals;
//                      i += blockDim.x)
//                 {

//                     j = lower_start_ind + i;

//                     n_A = data->psd_A[this_band_noise_index * data->data_length + j];
//                     n_E = data->psd_E[this_band_noise_index * data->data_length + j];

//                     d_A = data->data_A[this_band_data_index * data->data_length + j];
//                     d_E = data->data_E[this_band_data_index * data->data_length + j];

//                     // if ((bin_i == 0)){
//                     // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, this_band_noise_index * data->data_length + j, this_band_data_index * data->data_length + j, data->data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
//                     // }

//                     // if ((bin_i == 0)) printf("%d %e %e %e %e %e %e %e %e %e %e %e %e \n", tid, d_h_remove_temp.real(), d_h_remove_temp.imag(), d_h_add_temp.real(), d_h_add_temp.imag(), add_add_temp.real(), add_add_temp.imag(), remove_remove_temp.real(), remove_remove_temp.imag(), add_remove_temp.real(), add_remove_temp.imag());

//                     if (j < upper_start_ind)
//                     {
//                         real_ind = i;
//                         if (is_add_lower)
//                         {

//                             h_A = A_add[real_ind];
//                             h_E = E_add[real_ind];

//                             // get <d|h> term
//                             d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
//                             d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

//                             // <h|h>
//                             add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
//                             add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
//                         }
//                         else
//                         {
//                             h_A = A_remove[real_ind];
//                             h_E = E_remove[real_ind];

//                             // get <d|h> term
//                             d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
//                             d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

//                             // <h|h>
//                             remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
//                             remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

//                             // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
//                         }
//                     }
//                     else if (j >= lower_end_ind)
//                     {
//                         real_ind = j - upper_start_ind;
//                         if (!is_add_lower)
//                         {

//                             h_A_add = A_add[real_ind];
//                             h_E_add = E_add[real_ind];

//                             // get <d|h> term
//                             d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
//                             d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

//                             // <h|h>
//                             add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
//                             add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;
//                         }
//                         else
//                         {
//                             h_A_remove = A_remove[real_ind];
//                             h_E_remove = E_remove[real_ind];

//                             // get <d|h> term
//                             d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
//                             d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

//                             // <h|h>
//                             remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
//                             remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;
//                         }
//                     }
//                     else // this is where the signals overlap
//                     {
//                         if (is_add_lower)
//                         {
//                             real_ind_add = i;
//                         }
//                         else
//                         {
//                             real_ind_add = j - upper_start_ind;
//                         }

//                         h_A_add = A_add[real_ind_add];
//                         h_E_add = E_add[real_ind_add];

//                         // get <d|h> term
//                         d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
//                         d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

//                         // <h|h>
//                         add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
//                         add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;

//                         if (!is_add_lower)
//                         {
//                             real_ind_remove = i;
//                         }
//                         else
//                         {
//                             real_ind_remove = j - upper_start_ind;
//                         }

//                         h_A_remove = A_remove[real_ind_remove];
//                         h_E_remove = E_remove[real_ind_remove];

//                         // get <d|h> term
//                         d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
//                         d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

//                         // <h|h>
//                         remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
//                         remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;

//                         add_remove_temp += gcmplx::conj(h_A_remove) * h_A_add / n_A;
//                         add_remove_temp += gcmplx::conj(h_E_remove) * h_E_add / n_E;
//                     }
//                 }
//             }
//             else
//             {
//                 for (int i = threadIdx.x;
//                      i < params_curr->N;
//                      i += blockDim.x)
//                 {

//                     j = start_ind_remove + i - params_curr->start_freq_ind;

//                     n_A = data->psd_A[this_band_noise_index * data->data_length + j];
//                     n_E = data->psd_E[this_band_noise_index * data->data_length + j];

//                     d_A = data->data_A[this_band_data_index * data->data_length + j];
//                     d_E = data->data_E[this_band_data_index * data->data_length + j];

//                     // if ((bin_i == num_bin - 1))printf("CHECK remove: %d %e %e  \n", i, n_A, d_A.real());
//                     //  calculate h term
//                     h_A = A_remove[i];
//                     h_E = E_remove[i];

//                     // get <d|h> term
//                     d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
//                     d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

//                     // <h|h>
//                     remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
//                     remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;
//                 }

//                 for (int i = threadIdx.x;
//                      i < params_curr->N;
//                      i += blockDim.x)
//                 {

//                     j = start_ind_add + i - params_curr->start_freq_ind;

//                     n_A = data->psd_A[this_band_noise_index * data->data_length + j];
//                     n_E = data->psd_E[this_band_noise_index * data->data_length + j];

//                     d_A = data->data_A[this_band_data_index * data->data_length + j];
//                     d_E = data->data_E[this_band_data_index * data->data_length + j];

//                     // if ((bin_i == 0))printf("CHECK add: %d %d %e %e %e %e  \n", i, j, n_A, d_A.real(), n_E, d_E.real());
//                     //  calculate h term
//                     h_A = A_add[i];
//                     h_E = E_add[i];

//                     // get <d|h> term
//                     d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
//                     d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

//                     // <h|h>
//                     add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
//                     add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
//                 }
//             }

//             __syncthreads();
//             d_h_remove_arr[tid] = 4 * df * d_h_remove_temp;

//             d_h_add_arr[tid] = 4 * df * d_h_add_temp;
//             add_add_arr[tid] = 4 * df * add_add_temp;
//             remove_remove_arr[tid] = 4 * df * remove_remove_temp;
//             add_remove_arr[tid] = 4 * df * add_remove_temp;
//             __syncthreads();

//             for (unsigned int s = 1; s < blockDim.x; s *= 2)
//             {
//                 if (tid % (2 * s) == 0)
//                 {
//                     d_h_remove_arr[tid] += d_h_remove_arr[tid + s];
//                     d_h_add_arr[tid] += d_h_add_arr[tid + s];
//                     add_add_arr[tid] += add_add_arr[tid + s];
//                     remove_remove_arr[tid] += remove_remove_arr[tid + s];
//                     add_remove_arr[tid] += add_remove_arr[tid + s];
//                     // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
//                     // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
//                 }
//                 __syncthreads();
//             }
//             __syncthreads();

//             ll_diff = -1. / 2. * (-2. * d_h_add_arr[0] + 2. * d_h_remove_arr[0] - 2. * add_remove_arr[0] + add_add_arr[0] + remove_remove_arr[0]).real();
//             __syncthreads();

//             // determine detailed balance with tempering on the Likelihood term
//             lnpdiff = factors + (this_binary_inv_temp * ll_diff) + lp_diff;

//             // accept or reject
//             accept = lnpdiff > random_val;
//             // accept = false;
//             if ((mcmc_info->is_rj) && (amp_prop / amp_curr > 1e10))
//             {
//                 det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
//                 opt_snr = sqrt(add_add_arr[0].real());
//                 // put an snr limit on rj
//                 if ((opt_snr < mcmc_info->snr_lim) || (det_snr < mcmc_info->snr_lim) || (abs(1.0 - det_snr / opt_snr) > 0.5))
//                 {
//                     // if ((this_band_data_index == 0) && (threadIdx.x == 0)) printf("NIXED %e %e %e %e\n", mcmc_info->snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
//                     accept = false;
//                 }
//                 else
//                 {
//                     //if ((this_binary_inv_temp < 0.005) & (threadIdx.x == 0)) printf("KEPT  %e %e %e %e %e\n", this_binary_inv_temp, mcmc_info->snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
//                 }
//             }

//             // if ((mcmc_info->is_rj) && (amp_curr / amp_prop > 1e10) && (!accept) && (N == 1024) && (this_band_start_index >= 740556))
//             // {
//             //     //  + add_remove_arr[0].real()
//             //     det_snr = ((-d_h_remove_arr[0].real()) / sqrt(remove_remove_arr[0].real()));
//             //     opt_snr = sqrt(remove_remove_arr[0].real());
//             //     if ((threadIdx.x == 0))
//             //         //printf("SNR info  %e %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr), this_binary_inv_temp);
//             //         printf("lnpdiff: %e, beta: %e, factors: %e, ll_diff: %e, lp_diff: %e\n", lnpdiff, this_binary_inv_temp, factors, ll_diff, lp_diff);
//             // }

//             // if ((blockIdx.x == 0) && (threadIdx.x == 0))printf("%d %d %.12e %.12e %.12e %.12e %e %e %e %e %e %e\n", bin_i, accept, f0_prop, f0_curr, wave_remove[0].real(), wave_add[0].real(), ll_diff, this_binary_inv_temp, lp_diff, factors, lnpdiff, random_val);

//             // readout if it was accepted
//             mcmc_info->accepted_out[current_binary_start_index] = accept;

//             if (accept)
//             {
//                 if (tid == 0)
//                 {
//                     mcmc_info->L_contribution[band_i] += ll_diff;
//                     mcmc_info->p_contribution[band_i] += lp_diff;
//                 }
//                 __syncthreads();
//                 // change current Likelihood

//                 for (int i = threadIdx.x;
//                      i < params_curr->N;
//                      i += blockDim.x)
//                 {

//                     j = start_ind_remove + i - params_curr->start_freq_ind;

//                     h_A = A_remove[i];
//                     h_E = E_remove[i];

//                     // if (i == 0) printf("start: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data->data_A[this_band_data_index * data->data_length + j].real(), data->data_A[this_band_data_index * data->data_length + j].imag(), data->data_E[this_band_data_index * data->data_length + j].real(), data->data_E[this_band_data_index * data->data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());

//                     // data->data_A[this_band_data_index * data->data_length + j] += h_A;
//                     // data->data_E[this_band_data_index * data->data_length + j] += h_E;

//                     atomicAddComplex(&data->data_A[this_band_data_index * data->data_length + j], h_A);
//                     atomicAddComplex(&data->data_E[this_band_data_index * data->data_length + j], h_E);
//                     // if (i == 0) printf("remove: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n", bin_i, data->data_A[this_band_data_index * data->data_length + j].real(), data->data_A[this_band_data_index * data->data_length + j].imag(), data->data_E[this_band_data_index * data->data_length + j].real(), data->data_E[this_band_data_index * data->data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
//                 }
//                 __syncthreads();

//                 for (int i = threadIdx.x;
//                      i < params_curr->N;
//                      i += blockDim.x)
//                 {

//                     j = start_ind_add + i - params_curr->start_freq_ind;

//                     h_A = A_add[i];
//                     h_E = E_add[i];

//                     // data->data_A[this_band_data_index * data->data_length + j] -= h_A;
//                     // data->data_E[this_band_data_index * data->data_length + j] -= h_E;

//                     atomicAddComplex(&data->data_A[this_band_data_index * data->data_length + j], -h_A);
//                     atomicAddComplex(&data->data_E[this_band_data_index * data->data_length + j], -h_E);

//                     // if (i == 0) printf("add: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data->data_A[this_band_data_index * data->data_length + j].real(), data->data_A[this_band_data_index * data->data_length + j].imag(), data->data_E[this_band_data_index * data->data_length + j].real(), data->data_E[this_band_data_index * data->data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
//                 }
//                 __syncthreads();
//                 // do not need to adjust data as this one is already in there
//             }
//             __syncthreads();
//         }
//         __syncthreads();
//     }
// }

// // In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
// //
// // One block is run, it calculates two 128-point C2C double precision FFTs.
// // Data is generated on host, copied to device buffer, and then results are copied back to host.
// template <unsigned int Arch, unsigned int N>
// void make_move_wrap(InputInfo inputs)
// {
//     using namespace cufftdx;

//     if (inputs.device >= 0)
//     {
//         // set the device
//         CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
//     }

//     // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
//     // will be executed on block level. Shared memory is required for co-operation between threads.
//     // Additionally,

//     using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
//                          Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
//     using complex_type = cmplx;

//     // Allocate managed memory for input/output
//     auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
//     auto size_bytes = size * sizeof(cmplx);

//     // Shared memory must fit input data and must be big enough to run FFT
//     auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

//     // first is waveforms, second is ll, third is A, E data, fourth is A psd and E psd
//     auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 3 * N * sizeof(cmplx) + 5 * FFT::block_dim.x * sizeof(cmplx);
//     // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
//     // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
//     //     std::cout << data[i].x << " " << data[i].y << std::endl;
//     // }

//     // Increase max shared memory if needed
//     CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
//         make_move<FFT>,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         shared_memory_size_mine));

//     GalacticBinaryParams *params_curr_d;
//     GalacticBinaryParams *params_prop_d;
//     CUDA_CHECK_AND_EXIT(cudaMalloc(&params_curr_d, sizeof(GalacticBinaryParams)));
//     CUDA_CHECK_AND_EXIT(cudaMalloc(&params_prop_d, sizeof(GalacticBinaryParams)));
    
//     CUDA_CHECK_AND_EXIT(cudaMemcpy(params_curr_d, inputs.params_curr, sizeof(GalacticBinaryParams), cudaMemcpyHostToDevice));
//     CUDA_CHECK_AND_EXIT(cudaMemcpy(params_prop_d, inputs.params_prop, sizeof(GalacticBinaryParams), cudaMemcpyHostToDevice));

//     DataPackage *data_d;
//     CUDA_CHECK_AND_EXIT(cudaMalloc(&data_d, sizeof(DataPackage)));
//     CUDA_CHECK_AND_EXIT(cudaMemcpy(data_d, inputs.data, sizeof(DataPackage), cudaMemcpyHostToDevice));

//     BandPackage *band_info_d;
//     CUDA_CHECK_AND_EXIT(cudaMalloc(&band_info_d, sizeof(BandPackage)));
//     CUDA_CHECK_AND_EXIT(cudaMemcpy(band_info_d, inputs.band_info, sizeof(BandPackage), cudaMemcpyHostToDevice));
    
//     MCMCInfo *mcmc_info_d;
//     CUDA_CHECK_AND_EXIT(cudaMalloc(&mcmc_info_d, sizeof(MCMCInfo)));
//     CUDA_CHECK_AND_EXIT(cudaMemcpy(mcmc_info_d, inputs.mcmc_info, sizeof(MCMCInfo), cudaMemcpyHostToDevice));
    
    
//     //  Invokes kernel with FFT::block_dim threads in CUDA block
//     make_move<FFT><<<(inputs.band_info)->num_bands, FFT::block_dim, shared_memory_size_mine>>>(
//         data_d,
//         band_info_d,
//         params_curr_d,
//         params_prop_d,
//         mcmc_info_d);

//     CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
//     if (inputs.do_synchronize)
//     {
//         CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
//     }

//     // std::cout << "output [1st FFT]:\n";
//     // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
//     //     std::cout << data[i].x << " " << data[i].y << std::endl;
//     // }

//     // std::cout << shared_memory_size << std::endl;
//     // std::cout << "Success" <<  std::endl;
//     CUDA_CHECK_AND_EXIT(cudaFree(params_curr_d));
//     CUDA_CHECK_AND_EXIT(cudaFree(params_prop_d));
//     CUDA_CHECK_AND_EXIT(cudaFree(data_d));
//     CUDA_CHECK_AND_EXIT(cudaFree(band_info_d));
//     CUDA_CHECK_AND_EXIT(cudaFree(mcmc_info_d));
// }

// template <unsigned int Arch, unsigned int N>
// struct make_move_wrap_functor
// {
//     void operator()(InputInfo inputs) { return make_move_wrap<Arch, N>(inputs); }
// };

// void SharedMemoryMakeMove(
//     DataPackage *data,
//     BandPackage *band_info,
//     GalacticBinaryParams *params_curr,
//     GalacticBinaryParams *params_prop,
//     MCMCInfo *mcmc_info,
//     int device,
//     bool do_synchronize
// )
// {

//     InputInfo inputs;

//     inputs.data = data;
//     inputs.band_info = band_info;
//     inputs.params_curr = params_curr;
//     inputs.params_prop = params_prop;
//     inputs.mcmc_info = mcmc_info;
//     inputs.device = device;
//     inputs.do_synchronize = do_synchronize;

//     switch (params_curr->N)
//     {
//     // All SM supported by cuFFTDx
//     case 32:
//         example::sm_runner<make_move_wrap_functor, 32>(inputs);
//         return;
//     case 64:
//         example::sm_runner<make_move_wrap_functor, 64>(inputs);
//         return;
//     case 128:
//         example::sm_runner<make_move_wrap_functor, 128>(inputs);
//         return;
//     case 256:
//         example::sm_runner<make_move_wrap_functor, 256>(inputs);
//         return;
//     case 512:
//         example::sm_runner<make_move_wrap_functor, 512>(inputs);
//         return;
//     case 1024:
//         example::sm_runner<make_move_wrap_functor, 1024>(inputs);
//         return;
//     case 2048:
//         example::sm_runner<make_move_wrap_functor, 2048>(inputs);
//         return;

//     default:
//     {
//         throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
//     }
//     }

//     // const unsigned int arch = example::get_cuda_device_arch();
//     // simple_block_fft<800>(x);
// }

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////


template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void make_new_move(
    DataPackage *data_in,
    BandPackage *band_info_in,
    GalacticBinaryParams *params_curr_in,
    MCMCInfo *mcmc_info_in,
    PriorPackage *prior_info_in,
    StretchProposalPackage *stretch_info_in,
    PeriodicPackage *periodic_info_in,
    SingleBand *bands,
    bool use_global_memory,
    cmplx *global_memory_buffer
)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    int tid = threadIdx.x;
    unsigned int start_ind_add = 0;
    unsigned int start_ind_remove = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    BandPackage band_info = *band_info_in;
    GalacticBinaryParams params_curr = *params_curr_in;
    MCMCInfo mcmc_info = *mcmc_info_in;
    PriorPackage prior_info = *prior_info_in;
    StretchProposalPackage stretch_info = *stretch_info_in;
    PeriodicPackage periodic_info = *periodic_info_in;

    cmplx *wave_add = (cmplx *)shared_mem;
    cmplx *A_add = &wave_add[0];
    cmplx *E_add = &wave_add[params_curr.N];
    cmplx phase_factor = 0.0;

    cmplx *wave_remove = &wave_add[3 * params_curr.N];
    cmplx *A_remove = &wave_remove[0];
    cmplx *E_remove = &wave_remove[params_curr.N];

    cmplx *d_h_remove_arr = &wave_remove[3 * params_curr.N];
    cmplx *d_h_add_arr = &d_h_remove_arr[FFT::block_dim.x];
    ;
    cmplx *remove_remove_arr = &d_h_add_arr[FFT::block_dim.x];
    ;
    cmplx *add_add_arr = &remove_remove_arr[FFT::block_dim.x];
    ;
    cmplx *add_remove_arr = &add_add_arr[FFT::block_dim.x];

    cmplx *A_data;
    cmplx *E_data;

    cmplx I(0.0, 1.0);

    if (use_global_memory)
    {
        A_data = &global_memory_buffer[(2 * blockIdx.x) * band_info.max_data_store_size];
        E_data = &global_memory_buffer[(2 * blockIdx.x + 1) * band_info.max_data_store_size];
    }
    else
    {
        A_data = &add_remove_arr[FFT::block_dim.x];
        E_data = &A_data[band_info.max_data_store_size];
    }
    __syncthreads();

    int j, k, j_rj_data;
    double this_band_inv_temp;
    
    int current_binary_start_index; // , base_index;
    double ll_diff, lp_diff;
    double prior_curr, prior_prop, factors, lnpdiff;
    bool accept;
    double sens_val_prior;
    double phase_change;
    __shared__ int bin_i_gen;
    __shared__ curandState localState;
    __shared__ double random_val;

    if (threadIdx.x == 0)
    {
        localState = stretch_info.curand_states[blockIdx.x];
        // bin_i_gen = 0;
    }
    __syncthreads();

    cmplx d_h_remove_temp = 0.0;
    cmplx d_h_add_temp = 0.0;
    cmplx remove_remove_temp = 0.0;
    cmplx add_add_temp = 0.0;
    cmplx add_remove_temp = 0.0;
    cmplx d_h_add_term = 0.0;

    bool rj_is_there_already;
    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_add_lower;
    int total_i_vals;
    int f0_ind = 0;

    cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
    int real_ind, real_ind_add, real_ind_remove;
    cmplx d_A, d_E;
    double n_A, n_E;
    double opt_snr, det_snr;
    SingleGalacticBinary curr_binary(
        params_curr.N,
        params_curr.T,
        params_curr.Soms_d,
        params_curr.Sa_a,
        params_curr.Amp,
        params_curr.alpha,
        params_curr.sl1,
        params_curr.kn,
        params_curr.sl2
    );
    SingleGalacticBinary prop_binary(
        params_curr.N,
        params_curr.T,
        params_curr.Soms_d,
        params_curr.Sa_a,
        params_curr.Amp,
        params_curr.alpha,
        params_curr.sl1,
        params_curr.kn,
        params_curr.sl2
    );

    int tmp_data_index;

    DataPackage data = *data_in;
    double df = data.df;

    for (int band_i = blockIdx.x; band_i < band_info.num_bands; band_i += gridDim.x)
    {

        SingleBand band_here = bands[band_i];

        this_band_inv_temp = mcmc_info.band_inv_temperatures_all[band_i];

        // TODO need to change this
        if (band_here.update_data_index < band_here.data_index) tmp_data_index = band_here.update_data_index;
        else tmp_data_index = band_here.data_index;
        
        for (int i = threadIdx.x; i < band_here.band_data_lengths; i += blockDim.x)
        {
            A_data[i] = data.data_A[band_here.data_index * data.data_length + band_here.band_start_data_ind + i];
            E_data[i] = data.data_E[band_here.data_index * data.data_length + band_here.band_start_data_ind + i];
            // if (blockIdx.x == gridDim.x - 1) printf("%d %e, %e\n", i, A_data[i].real(), E_data[i].imag());

            // if ((i + band_here.band_start_data_ind < band_here.band_interest_start_data_ind - int((float)band_here.gb_params.N / 2.0) - 1) || (i + band_here.band_start_data_ind >= band_here.band_interest_start_data_ind + band_here.band_interest_data_lengths + int((float)band_here.gb_params.N / 2.0) + 2))
            // {
            //     A_data[i] += (data.data_A[band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i] - data.base_data_A[tmp_data_index * data.data_length + band_here.band_start_data_ind + i]);
            //     E_data[i] += (data.data_E[band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i] - data.base_data_E[tmp_data_index * data.data_length + band_here.band_start_data_ind + i]);
            //     // if ((band_here.band_start_data_ind == 3591) && (i % 25 == 0)) printf("HMMM2: %d %d %.12e %d %.12e %d %.12e %.12e \n", i, band_here.data_index * data.data_length + band_here.band_start_data_ind + i, data.data_A[band_here.data_index * data.data_length + band_here.band_start_data_ind + i].real(), band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i, data.data_A[band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i].real(), tmp_data_index * data.data_length + band_here.band_start_data_ind + i, data.base_data_A[tmp_data_index * data.data_length + band_here.band_start_data_ind + i].real(), A_data[i].real());
                
            // }
            
        }
        __syncthreads();
        for (int bin_i = 0; bin_i < band_here.band_num_bins; bin_i += 1)
        {
            current_binary_start_index = band_here.band_start_bin_ind + bin_i;
            // get the parameters to add and remove
            
            curr_binary.amp = band_here.gb_params.amp[bin_i];
            curr_binary.f0_ms = band_here.gb_params.f0_ms[bin_i];
            curr_binary.fdot = band_here.gb_params.fdot0[bin_i];
            curr_binary.phi0 = band_here.gb_params.phi0[bin_i];
            curr_binary.cosinc = band_here.gb_params.cosinc[bin_i];
            curr_binary.psi = band_here.gb_params.psi[bin_i];
            curr_binary.lam = band_here.gb_params.lam[bin_i];
            curr_binary.sinbeta = band_here.gb_params.sinbeta[bin_i];

            // adjust for rj
            if (mcmc_info.is_rj)
            {   
                rj_is_there_already = stretch_info.inds[band_here.band_start_bin_ind + bin_i];
                if (!rj_is_there_already)
                {
                    curr_binary.amp = 1e-50;
                }
            }

            curr_binary.transform();

            // if (threadIdx.x == 0)
            //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

            __syncthreads();
            build_new_single_waveform<FFT>(
                wave_remove,
                &start_ind_remove,
                curr_binary,
                bin_i);
            __syncthreads();

            //if ((threadIdx.x == 0) && (band_here.band_start_data_ind == 3811)) printf("HMMM35: %d %d %.12e %.12e  %.12e %.12e \n", bin_i, start_ind_remove, curr_binary.f0_ms, A_remove[0].real(), A_remove[1].real(), A_remove[3].real());

            for (int i = threadIdx.x; i < band_here.gb_params.N; i += blockDim.x)
            {
                // add to residual with (-)
                j = (start_ind_remove - band_here.band_start_data_ind) + i;
                if ((j < band_here.band_data_lengths) && (j >= 0))
                {
                    A_data[j] -= A_remove[i];
                    E_data[j] -= E_remove[i];
                }
                else
                {
                    printf("BAD 1 ");
                }
            }
            __syncthreads();
            // if ((threadIdx.x == 0) && (band_here.band_start_data_ind == 3811)) printf("HMMM27: %d %.12e %.12e \n", 100, A_data[100].real(), A_data[100].imag());
                
        }
        __syncthreads();

        // if (!mcmc_info.is_rj) continue;
        // double like_share_before = 0.0;
        // if (threadIdx.x == 0)
        // {
        //     for (int i = 0; i < band_here.band_data_lengths; i += 1)
        //     {
        //         // if ((band_here.band_start_data_ind == 3811) && (i == 100)) printf("HMMM3: %d %.12e %.12e \n", i, A_data[i].real(), A_data[i].imag());
                
        //         like_share_before += (-1./2. * 4 * df * A_data[i] * gcmplx::conj(A_data[i]) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
        //         like_share_before += (-1./2. * 4 * df * E_data[i] * gcmplx::conj(E_data[i]) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
        //     }
        // }
        // __syncthreads();

        for (int prop_i = 0; prop_i < stretch_info.num_proposals * band_here.band_num_bins; prop_i += 1)
        {
            if (threadIdx.x == 0)
            {
                random_val = log(curand_uniform_double(&localState));
                if (mcmc_info.is_rj)
                {
                    bin_i_gen = prop_i;
                    // printf("%d %d\n", blockIdx.x, bin_i_gen);
                }
                else
                {
                    bin_i_gen = ((int)(ceil(curand_uniform_double(&localState) * band_here.band_num_bins))) - 1;
                }
            }
            __syncthreads();
            if ((bin_i_gen >= band_here.band_num_bins) && (mcmc_info.is_rj))
            {
                continue;
            }
            
            current_binary_start_index = band_here.band_start_bin_ind + bin_i_gen;

            curr_binary.amp = band_here.gb_params.amp[bin_i_gen];
            curr_binary.f0_ms = band_here.gb_params.f0_ms[bin_i_gen];
            curr_binary.fdot = band_here.gb_params.fdot0[bin_i_gen];
            curr_binary.phi0 = band_here.gb_params.phi0[bin_i_gen];
            curr_binary.cosinc = band_here.gb_params.cosinc[bin_i_gen];
            curr_binary.psi = band_here.gb_params.psi[bin_i_gen];
            curr_binary.lam = band_here.gb_params.lam[bin_i_gen];
            curr_binary.sinbeta = band_here.gb_params.sinbeta[bin_i_gen];

            // if (threadIdx.x == 0) printf("curr: %d %d %e %e %e %e %e %e %e %e %e\n", band_i, bin_i_gen, curr_binary.amp, curr_binary.f0_ms, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.cosinc, curr_binary.psi, curr_binary.lam, curr_binary.sinbeta);

            if (!mcmc_info.is_rj)
            {
                stretch_info.get_proposal(&prop_binary, &factors, localState, curr_binary, periodic_info);
                prior_curr = mcmc_info.prior_all_curr[current_binary_start_index];
                
                f0_ind = (int)floor((prop_binary.f0_ms / 1e3) * prop_binary.T);
                if ((f0_ind > 0) && (f0_ind < data.data_length - 1))
                {
                    sens_val_prior = ((data.lisasens_A[band_here.noise_index * data.data_length + f0_ind + 1] - data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]) / data.df * ((prop_binary.f0_ms / 1e3) - (data.df * f0_ind)) + data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]);
                    // if ((threadIdx.x == 0) && (blockIdx.x < 5)) printf("PSD: %.12e %.12e %.12e %.12e %.12e %.12e\n", (prop_binary.f0_ms / 1e3) - (data.df * f0_ind), (prop_binary.f0_ms / 1e3), (data.df * f0_ind), data.psd_A[band_here.noise_index * data.data_length + f0_ind], sens_val_prior, data.psd_A[band_here.noise_index * data.data_length + f0_ind + 1]);
                }
                else
                {
                    sens_val_prior = -100.0;
                }
                prior_prop = prior_info.get_prior_val(prop_binary, 100, sens_val_prior);
            }
            else
            { 
                factors = stretch_info.factors[band_here.band_start_bin_ind + bin_i_gen];
                rj_is_there_already = stretch_info.inds[band_here.band_start_bin_ind + bin_i_gen];
                prop_binary = curr_binary;

                if (rj_is_there_already)
                {
                    prop_binary.amp = 1e-50;
                    f0_ind = (int)floor((curr_binary.f0_ms / 1e3) * curr_binary.T);
                    if ((f0_ind > 0) && (f0_ind < data.data_length - 1))
                    {
                        sens_val_prior = ((data.lisasens_A[band_here.noise_index * data.data_length + f0_ind + 1] - data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]) / data.df * ((curr_binary.f0_ms / 1e3) - (data.df * f0_ind)) + data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]);
                    }
                    else
                    {
                        sens_val_prior = -100.0;
                    }
                    prior_curr = prior_info.get_prior_val(curr_binary, 100, sens_val_prior);
                    prior_prop = 0.0;
                }
                else
                {
                    curr_binary.amp = 1e-50;
                    prior_curr = 0.0;
                    f0_ind = (int)floor((prop_binary.f0_ms / 1e3) * prop_binary.T);
                    if ((f0_ind > 0) && (f0_ind < data.data_length - 1))
                    {
                        sens_val_prior = ((data.lisasens_A[band_here.noise_index * data.data_length + f0_ind + 1] - data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]) / data.df * ((prop_binary.f0_ms / 1e3) - (data.df * f0_ind)) + data.lisasens_A[band_here.noise_index * data.data_length + f0_ind]);
                    }
                    else
                    {
                        sens_val_prior = -100.0;
                    }
                    prior_prop = prior_info.get_prior_val(prop_binary, 100, sens_val_prior);
                }
            }

            lp_diff = prior_prop - prior_curr;
            
            // if ((threadIdx.x == 0) && (blockIdx.x < 10)) printf("%e %e %e\n", lp_diff, prior_prop, prior_curr);
            // if (threadIdx.x == 0) printf("prop: %d %e %e %e %e %e %e %e %e %e %e %e %e\n", band_i, prior_prop, band_here.fmin_allow, band_here.fmax_allow, prop_binary.amp, prop_binary.f0_ms, prop_binary.fdot, prop_binary.fddot, prop_binary.phi0, prop_binary.cosinc, prop_binary.psi, prop_binary.lam, prop_binary.sinbeta);

            if ((prior_prop > -1e100) && (prop_binary.f0_ms / 1e3 >= band_here.fmin_allow) && (prop_binary.f0_ms / 1e3 <= band_here.fmax_allow) && (abs(curr_binary.f0_ms - prop_binary.f0_ms) / 1e3 < (band_here.gb_params.N * df / 2.)))
            { 
                
                curr_binary.transform();
                prop_binary.transform();

                //if ((blockIdx.x == 100) && (threadIdx.x == 0)) printf("%d %e %e %e\n %e %e %e %e %e %e %e %e\n %e %e %e %e %e %e %e %e\n\n\n", bin_i_gen, prior_curr, prior_prop, factors, curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta, prop_binary.amp, prop_binary.f0, prop_binary.fdot, prop_binary.phi0, prop_binary.inc, prop_binary.psi, prop_binary.lam, prop_binary.theta);

                
                // if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);
                // if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("%e %e %e %e %e %e %e %e %e\n", prop_binary.amp, prop_binary.f0, prop_binary.fdot, prop_binary.fddot, prop_binary.phi0, prop_binary.inc, prop_binary.psi, prop_binary.lam, prop_binary.theta);

                __syncthreads();
                build_new_single_waveform<FFT>(
                    wave_remove,
                    &start_ind_remove,
                    curr_binary,
                    bin_i_gen);
                __syncthreads();
                build_new_single_waveform<FFT>(
                    wave_add,
                    &start_ind_add,
                    prop_binary,
                    bin_i_gen);
                __syncthreads();

                if (true)  // abs((double)start_ind_add - (double)start_ind_remove) < 20.0)
                {

                    // subtract start_freq_ind to find index into subarray
                    if (start_ind_remove <= start_ind_add)
                    {
                        lower_start_ind = start_ind_remove - band_here.band_start_data_ind;
                        upper_end_ind = start_ind_add - band_here.band_start_data_ind + band_here.gb_params.N;

                        upper_start_ind = start_ind_add - band_here.band_start_data_ind;
                        lower_end_ind = start_ind_remove - band_here.band_start_data_ind + band_here.gb_params.N;

                        is_add_lower = false;
                    }
                    else
                    {
                        lower_start_ind = start_ind_add - band_here.band_start_data_ind;
                        upper_end_ind = start_ind_remove - band_here.band_start_data_ind + band_here.gb_params.N;

                        upper_start_ind = start_ind_remove - band_here.band_start_data_ind;
                        lower_end_ind = start_ind_add - band_here.band_start_data_ind + band_here.gb_params.N;

                        is_add_lower = true;
                    }
                    total_i_vals = upper_end_ind - lower_start_ind;
                    __syncthreads();
                    d_h_remove_temp = 0.0;
                    d_h_add_temp = 0.0;
                    remove_remove_temp = 0.0;
                    add_add_temp = 0.0;
                    add_remove_temp = 0.0;

                    if (total_i_vals < 2 * band_here.gb_params.N)
                    {
                        for (int i = threadIdx.x;
                            i < total_i_vals;
                            i += blockDim.x)
                        {

                            j = lower_start_ind + i;

                            if ((j > band_here.band_data_lengths) || (j < 0))
                            {
                                d_h_add_temp += -1e300;
                            }

                            // noise is still in global memory
                            // TODO: Maybe put this in shared memory if possible
                            if ((j < band_here.band_data_lengths) && (j >= 0))
                            {
                                d_A = A_data[j];
                                d_E = E_data[j];

                                n_A = data.psd_A[band_here.noise_index * data.data_length + j + band_here.band_start_data_ind];
                                n_E = data.psd_E[band_here.noise_index * data.data_length + j + band_here.band_start_data_ind];
                            }
                            else
                            {
                                d_A = 0.0;
                                d_E = 0.0;
                                n_A = 1e100;
                                n_E = 1e100;
                                printf("BAD 3 ");
                            }

                            if (j < upper_start_ind)
                            {
                                real_ind = i;
                                if (is_add_lower)
                                {

                                    h_A = A_add[real_ind];
                                    h_E = E_add[real_ind];

                                    // get <d|h> term
                                    d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                                    d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

                                    // <h|h>
                                    add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                                    add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
                                }
                                else
                                {
                                    h_A = A_remove[real_ind];
                                    h_E = E_remove[real_ind];

                                    // get <d|h> term
                                    d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                                    d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

                                    // <h|h>
                                    remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                                    remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

                                    // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                                }
                            }
                            else if (j >= lower_end_ind)
                            {
                                real_ind = j - upper_start_ind;
                                if (!is_add_lower)
                                {

                                    h_A_add = A_add[real_ind];
                                    h_E_add = E_add[real_ind];

                                    // get <d|h> term
                                    d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                                    d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                                    // <h|h>
                                    add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                                    add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;
                                }
                                else
                                {
                                    h_A_remove = A_remove[real_ind];
                                    h_E_remove = E_remove[real_ind];

                                    // get <d|h> term
                                    d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                                    d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                                    // <h|h>
                                    remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                                    remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;

                                }
                            }
                            else // this is where the signals overlap
                            {
                                if (is_add_lower)
                                {
                                    real_ind_add = i;
                                }
                                else
                                {
                                    real_ind_add = j - upper_start_ind;
                                }

                                h_A_add = A_add[real_ind_add];
                                h_E_add = E_add[real_ind_add];

                                // get <d|h> term
                                d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                                d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                                // <h|h>
                                add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                                add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;

                                if (!is_add_lower)
                                {
                                    real_ind_remove = i;
                                }
                                else
                                {
                                    real_ind_remove = j - upper_start_ind;
                                }

                                h_A_remove = A_remove[real_ind_remove];
                                h_E_remove = E_remove[real_ind_remove];

                                // get <d|h> term
                                d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                                d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                                // <h|h>
                                remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                                remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;

                                add_remove_temp += gcmplx::conj(h_A_remove) * h_A_add / n_A;
                                add_remove_temp += gcmplx::conj(h_E_remove) * h_E_add / n_E;

                            }
                        }
                    }
                    else
                    {
                        // __syncthreads();
                        // if (threadIdx.x == 0)
                        // {
                        //     double d_h_d_h_check = 0.0;
                        //     for (int i = 0; i < band_here.band_data_lengths; i += 1)
                        //     {
                        //         d_h_d_h_check += (-1./2. * 4 * df * A_data[i] * gcmplx::conj(A_data[i]) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                        //         d_h_d_h_check += (-1./2. * 4 * df * E_data[i] * gcmplx::conj(E_data[i]) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                        //     }
                        //     printf("d_h_d_h : %d %.12e \n", band_i, d_h_d_h_check);

                        //     // printf("start ind: %d\n A: \n", start_ind_remove);
                        //     // for (int i = 0; i < band_here.gb_params.N; i += 1)
                        //     // {
                        //     //     printf("%.16e + 1j * %.16e, ", A_remove[i].real(), A_remove[i].imag());
                        //     // }
                        //     // printf("\n\nstart ind: %d\n E: \n", start_ind_remove);
                        //     // for (int i = 0; i < band_here.gb_params.N; i += 1)
                        //     // {
                        //     //     printf("%.16e + 1j * %.16e, ", E_remove[i].real(), E_remove[i].imag());
                        //     // }
                        //     // printf("\n\n\n");
                        // }
                        // __syncthreads();
                        // for (int i = threadIdx.x;
                        //      i < band_here.gb_params.N;
                        //      i += blockDim.x)
                        // {
                        //     j = start_ind_remove + i - band_here.band_start_data_ind;
                        //     //  calculate h term
                        //     h_A = A_remove[i];
                        //     h_E = E_remove[i];

                        //     A_data[j] += h_A;
                        //     E_data[j] += h_E;
                        // }
                        // __syncthreads();
                        // // for (int i = threadIdx.x;
                        // //      i < band_here.gb_params.N;
                        // //      i += blockDim.x)
                        // // {
                        // //     j = start_ind_add + i - band_here.band_start_data_ind;
                        // //     //  calculate h term
                        // //     h_A = A_add[i];
                        // //     h_E = E_add[i];

                        // //     A_data[j] -= h_A;
                        // //     E_data[j] -= h_E;
                        // // }
                        // __syncthreads();
                        // if (threadIdx.x == 0)
                        // {
                        //     double like_share_check1 = 0.0;
                        //     for (int i = 0; i < band_here.band_data_lengths; i += 1)
                        //     {
                        //         like_share_check1 += (-1./2. * 4 * df * A_data[i] * gcmplx::conj(A_data[i]) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                        //         like_share_check1 += (-1./2. * 4 * df * E_data[i] * gcmplx::conj(E_data[i]) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                        //     }
                        //     printf("d_h_r_d_h_r : %d %.12e \n", band_i, like_share_check1);
                        // }
                        // __syncthreads();
                        // for (int i = threadIdx.x;
                        //      i < band_here.gb_params.N;
                        //      i += blockDim.x)
                        // {
                        //     j = start_ind_remove + i - band_here.band_start_data_ind;
                        //     //  calculate h term
                        //     h_A = A_remove[i];
                        //     h_E = E_remove[i];

                        //     A_data[j] -= h_A;
                        //     E_data[j] -= h_E;
                        // }
                        // __syncthreads();
                        // for (int i = threadIdx.x;
                        //      i < band_here.gb_params.N;
                        //      i += blockDim.x)
                        // {
                        //     j = start_ind_add + i - band_here.band_start_data_ind;
                        //     //  calculate h term
                        //     h_A = A_add[i];
                        //     h_E = E_add[i];

                        //     A_data[j] += h_A;
                        //     E_data[j] += h_E;
                        // }
                        __syncthreads();





                        

                        for (int i = threadIdx.x;
                            i < band_here.gb_params.N;
                            i += blockDim.x)
                        {

                            j = start_ind_remove + i - band_here.band_start_data_ind;

                            //  calculate h term
                            h_A = A_remove[i];
                            h_E = E_remove[i];

                            n_A = data.psd_A[band_here.noise_index * data.data_length + start_ind_remove + i];
                            n_E = data.psd_E[band_here.noise_index * data.data_length + start_ind_remove + i];

                            // <h|h>
                            remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                            remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

                            if ((j < band_here.band_data_lengths) && (j >= 0))
                            {
                                d_A = A_data[j];
                                d_E = E_data[j];

                                // get <d|h> term
                                d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                                d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;
                            }
                            else
                            {
                                printf("BAD 4 ");
                                d_h_remove_temp += -1e300;
                            }
                            
                        }
                        __syncthreads();
                        for (int i = threadIdx.x;
                            i < band_here.gb_params.N;
                            i += blockDim.x)
                        {

                            j = start_ind_add + i - band_here.band_start_data_ind;

                            // if ((bin_i == 0))printf("CHECK add: %d %d %e %e %e %e  \n", i, j, n_A, d_A.real(), n_E, d_E.real());
                            //  calculate h term
                            h_A = A_add[i];
                            h_E = E_add[i];

                            n_A = data.psd_A[band_here.noise_index * data.data_length + start_ind_add + i];
                            n_E = data.psd_E[band_here.noise_index * data.data_length + start_ind_add + i];

                            // <h|h>
                            add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                            add_add_temp += gcmplx::conj(h_E) * h_E / n_E;

                            // get <d|h> term
                            if ((j < band_here.band_data_lengths) && (j >= 0))
                            {    
                                d_A = A_data[j];
                                d_E = E_data[j];
                                
                                d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                                d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;
                            }
                            else
                            {
                                printf("BAD 5 ");
                                d_h_add_temp += -1e300;
                            }
                        }
                    }

                    __syncthreads();
                    d_h_remove_arr[tid] = 4 * df * d_h_remove_temp;

                    d_h_add_arr[tid] = 4 * df * d_h_add_temp;
                    add_add_arr[tid] = 4 * df * add_add_temp;
                    remove_remove_arr[tid] = 4 * df * remove_remove_temp;
                    add_remove_arr[tid] = 4 * df * add_remove_temp;
                    __syncthreads();

                    for (unsigned int s = 1; s < blockDim.x; s *= 2)
                    {
                        if (tid % (2 * s) == 0)
                        {
                            d_h_remove_arr[tid] += d_h_remove_arr[tid + s];
                            d_h_add_arr[tid] += d_h_add_arr[tid + s];
                            add_add_arr[tid] += add_add_arr[tid + s];
                            remove_remove_arr[tid] += remove_remove_arr[tid + s];
                            add_remove_arr[tid] += add_remove_arr[tid + s];
                            // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                            // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
                        }
                        __syncthreads();
                    }
                    __syncthreads();

                    if (mcmc_info.phase_maximize)
                    {
                        d_h_add_term = gcmplx::abs(d_h_add_arr[0]);

                        phase_change = gcmplx::arg(d_h_add_arr[0]);
                        prop_binary.phi0 -= phase_change;
                        while(prop_binary.phi0 < 0.0)
                        {
                            prop_binary.phi0 += 2 * M_PI;
                        }
                        while(prop_binary.phi0 > 2 * M_PI)
                        {
                            prop_binary.phi0 -= 2 * M_PI;
                        }

                        phase_factor = gcmplx::exp(-I * phase_change);

                        __syncthreads();
                        for (int i = threadIdx.x; i < prop_binary.N; i +=  blockDim.x)
                        {

                            A_add[i] *= phase_factor;
                            E_add[i] *= phase_factor;
                        }
                        __syncthreads();
                    }
                    else
                    {
                        d_h_add_term = d_h_add_arr[0];
                    }
                    
                    ll_diff = -1. / 2. * (-2. * d_h_add_term + 2. * d_h_remove_arr[0] - 2. * add_remove_arr[0] + add_add_arr[0] + remove_remove_arr[0]).real();
                    __syncthreads();
                    // if ((blockIdx.x == 100) && (threadIdx.x == 0)) printf("%d %e %e %e %e %e %e\n\n\n", bin_i_gen, ll_diff, d_h_add_arr[0].real(), d_h_remove_arr[0].real(), add_remove_arr[0].real(), add_add_arr[0].real(), remove_remove_arr[0].real());


                    // determine detailed balance with tempering on the Likelihood term
                    lnpdiff = factors + (this_band_inv_temp * ll_diff) + lp_diff;
                    //if ((this_band_inv_temp != 0.0) && (threadIdx.x == 0)) printf("WHAT3: %d %e\n", band_i, this_band_inv_temp);

                    // accept or reject
                    accept = lnpdiff > random_val;
                    // if ((mcmc_info.is_rj) && (band_here.band_num_bins > 10) && (threadIdx.x == 0)) printf("%d %e %e %e %e %e %e %e\n", band_i, lnpdiff, factors, ll_diff, this_band_inv_temp, lp_diff, prior_curr, prior_prop);
                }
                else {accept = false;}
            }
            else
            {
                // if ((mcmc_info.is_rj) && (threadIdx.x == 0)) printf("%d %d %e %e %.14e %.14e %.14e\n", band_i, bin_i_gen, prior_curr, prior_prop, prop_binary.f0_ms / 1e3, band_here.fmin_allow, band_here.fmax_allow);
                accept = false;
            }
            // if (mcmc_info.is_rj) accept = false;
            if ((mcmc_info.is_rj) && (prop_binary.amp / curr_binary.amp > 1e10))
            {
                // accept = false;
                // det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
                // opt_snr = sqrt(add_add_arr[0].real());
                // // put an snr limit on rj
                // if ((opt_snr < mcmc_info.snr_lim) || (det_snr < mcmc_info.snr_lim) || (abs(1.0 - det_snr / opt_snr) > 0.5))
                // {
                //     // if ((band_here.data_index == 0) && (threadIdx.x == 0)) printf("NIXED %e %e %e %e\n", mcmc_info.snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                //     accept = false;
                // }
                // else
                // {
                    //if ((this_band_inv_temp < 0.005) & (threadIdx.x == 0)) printf("KEPT  %e %e %e %e %e\n", this_band_inv_temp, mcmc_info.snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                // }
            }

            // if ((mcmc_info.is_rj) && (amp_curr / amp_prop > 1e10) && (!accept) && (N == 1024) && (band_here.band_start_data_ind >= 740556))
            // {
            //     //  + add_remove_arr[0].real()
            //     det_snr = ((-d_h_remove_arr[0].real()) / sqrt(remove_remove_arr[0].real()));
            //     opt_snr = sqrt(remove_remove_arr[0].real());
            //     if ((threadIdx.x == 0))
            //         //printf("SNR info  %e %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr), this_band_inv_temp);
            //         printf("lnpdiff: %e, beta: %e, factors: %e, ll_diff: %e, lp_diff: %e\n", lnpdiff, this_band_inv_temp, factors, ll_diff, lp_diff);
            // }

            // if ((blockIdx.x == 0) && (threadIdx.x == 0))printf("%d %d %.12e %.12e %.12e %.12e %e %e %e %e %e %e\n", bin_i, accept, f0_prop, f0_curr, wave_remove[0].real(), wave_add[0].real(), ll_diff, this_band_inv_temp, lp_diff, factors, lnpdiff, random_val);

            // readout if it was accepted
            // if (!(band_i == 123)) accept = false;
            if (accept)
            {
                if (tid == 0)
                {
                    mcmc_info.accepted_out[band_i] += 1;
                    mcmc_info.L_contribution[band_i] += ll_diff;
                    mcmc_info.p_contribution[band_i] += lp_diff;

                    // do not need to adjust data as this one is already in there
                    mcmc_info.prior_all_curr[current_binary_start_index] = prior_prop;

                    band_here.gb_params.amp[bin_i_gen] = prop_binary.amp;
                    band_here.gb_params.f0_ms[bin_i_gen] = prop_binary.f0_ms;
                    band_here.gb_params.fdot0[bin_i_gen] = prop_binary.fdot;
                    band_here.gb_params.phi0[bin_i_gen] = prop_binary.phi0;
                    band_here.gb_params.cosinc[bin_i_gen] = prop_binary.cosinc;
                    band_here.gb_params.psi[bin_i_gen] = prop_binary.psi;
                    band_here.gb_params.lam[bin_i_gen] = prop_binary.lam;
                    band_here.gb_params.sinbeta[bin_i_gen] = prop_binary.sinbeta;

                    if (mcmc_info.is_rj)
                    {
                        stretch_info.inds[band_here.band_start_bin_ind + bin_i_gen] = (!stretch_info.inds[band_here.band_start_bin_ind + bin_i_gen]);
                    }

                    // if ((band_i == 123) && (threadIdx.x == 0)) printf("base: %d %d %d %d %e %e %e %e %e %e\n", band_i, band_here.data_index, band_here.update_data_index, mcmc_info.accepted_out[band_i], ll_diff, lp_diff, factors, random_val, mcmc_info.L_contribution[band_i], this_band_inv_temp);
                    // if ((band_i == 123) && (threadIdx.x == 0)) printf("curr acc: %d %e %e %e %e %e %e %e %e %e\n", band_i, curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);
                    // if ((band_i == 123) && (threadIdx.x == 0)) printf("prop acc: %d %e %e %e %e %e %e %e %e %e\n", band_i, prop_binary.amp, prop_binary.f0, prop_binary.fdot, prop_binary.fddot, prop_binary.phi0, prop_binary.inc, prop_binary.psi, prop_binary.lam, prop_binary.theta);

                }
                __syncthreads();
                // change current Likelihood

                // if ((threadIdx.x == 0)) printf("base acc: %d %d %d %d %d %d %e %e %d %d\n", band_i, band_here.data_index, band_here.update_data_index, mcmc_info.accepted_out[band_i], start_ind_add, start_ind_remove, ll_diff, mcmc_info.L_contribution[band_i], band_here.band_start_bin_ind, band_here.band_data_lengths);
                // if ((threadIdx.x == 0)) printf("curr acc: %d %d %e %e %e %e %e %e %e %e %e\n", band_i, bin_i_gen, curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);
                // if ((threadIdx.x == 0)) printf("prop acc: %d %d %e %e %e %e %e %e %e %e %e\n\n", band_i, bin_i_gen, prop_binary.amp, prop_binary.f0, prop_binary.fdot, prop_binary.fddot, prop_binary.phi0, prop_binary.inc, prop_binary.psi, prop_binary.lam, prop_binary.theta);

                // if ((threadIdx.x == 0) && (band_here.band_start_data_ind == 3811)) printf("HMMM35: %d %d %.12e %.12e  %.12e %.12e %d %.12e %.12e  %.12e %.12e \n", bin_i_gen, start_ind_add, prop_binary.f0_ms, A_add[0].real(), A_add[1].real(), A_add[3].real(), start_ind_remove, curr_binary.f0_ms, A_remove[0].real(), A_remove[1].real(), A_remove[3].real());

                // if ((start_ind_remove - band_here.band_start_data_ind < 0) || (start_ind_remove - band_here.band_start_data_ind + band_here.gb_params.N > band_here.band_data_lengths))
                // {
                //     printf("%d %d %d %d %d\n", band_i, start_ind_remove - band_here.band_start_data_ind, start_ind_remove, band_here.band_start_data_ind, band_here.band_data_lengths);
                // }

                for (int i = threadIdx.x;
                     i < band_here.gb_params.N;
                     i += blockDim.x)
                {

                    j = start_ind_remove + i - band_here.band_start_data_ind;

                    h_A = A_remove[i];
                    h_E = E_remove[i];

                    // if (i == 0) printf("start: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data.data_A[band_here.data_index * data.data_length + j].real(), data.data_A[band_here.data_index * data.data_length + j].imag(), data.data_E[band_here.data_index * data.data_length + j].real(), data.data_E[band_here.data_index * data.data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());

                    // data.data_A[band_here.data_index * data.data_length + j] += h_A;
                    // data.data_E[band_here.data_index * data.data_length + j] += h_E;

                    // remove from residual means (+)
                    if ((j < band_here.band_data_lengths) && (j >= 0))
                    {    
                        A_data[j] +=  h_A;
                        E_data[j] +=  h_E;
                    }
                    else
                    {
                        printf("BAD 10 ");
                    }

                    if ((mcmc_info.is_rj) && (this_band_inv_temp == 1.0))
                    {
                        j_rj_data = start_ind_remove + i;
                        if ((j_rj_data < data.data_length) && (j_rj_data >= 0))
                        {
                            data.data_A[band_here.update_data_index * data.data_length + j_rj_data] += h_A;
                            data.data_E[band_here.update_data_index * data.data_length + j_rj_data] += h_E;
                        }
                        else
                        {
                            printf("BAD 6 ");
                        }
                    }
                    // if (i == 0) printf("remove: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n", bin_i, data.data_A[band_here.data_index * data.data_length + j].real(), data.data_A[band_here.data_index * data.data_length + j].imag(), data.data_E[band_here.data_index * data.data_length + j].real(), data.data_E[band_here.data_index * data.data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                }
                __syncthreads();

                // if ((start_ind_add - band_here.band_start_data_ind < 0) || (start_ind_add - band_here.band_start_data_ind + band_here.gb_params.N > band_here.band_data_lengths))
                // {
                //     printf("%d %d %d %d %d\n", band_i, start_ind_remove - band_here.band_start_data_ind, start_ind_remove, band_here.band_start_data_ind, band_here.band_data_lengths);
                // }

                for (int i = threadIdx.x;
                     i < band_here.gb_params.N;
                     i += blockDim.x)
                {

                    j = start_ind_add + i - band_here.band_start_data_ind;

                    h_A = A_add[i];
                    h_E = E_add[i];

                    // data.data_A[band_here.data_index * data.data_length + j] -= h_A;
                    // data.data_E[band_here.data_index * data.data_length + j] -= h_E;

                    // add from residual means (-)
                    if ((j < band_here.band_data_lengths) && (j >= 0))
                    {
                        A_data[j] -=  h_A;
                        E_data[j] -=  h_E;
                    }
                    else
                    {
                        printf("BAD 11 ");
                    }
                    // if (i == 0) printf("add: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data.data_A[band_here.data_index * data.data_length + j].real(), data.data_A[band_here.data_index * data.data_length + j].imag(), data.data_E[band_here.data_index * data.data_length + j].real(), data.data_E[band_here.data_index * data.data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                    if ((mcmc_info.is_rj) && (this_band_inv_temp == 1.0))
                    {
                        j_rj_data = start_ind_add + i;
                        if ((j_rj_data < data.data_length) && (j_rj_data >= 0))
                        {
                            data.data_A[band_here.update_data_index * data.data_length + j_rj_data] -= h_A;
                            data.data_E[band_here.update_data_index * data.data_length + j_rj_data] -= h_E;
                        }
                        else
                        {
                            printf("BAD 7 ");
                        }
                    }
                }
                __syncthreads();
                // if (threadIdx.x == 0)
                // {
                //     double like_share_middle = 0.0;
                //     for (int i = 0; i < band_here.band_data_lengths; i += 1)
                //     {
                //         // if ((band_here.band_start_data_ind == 3811) && (i == 100)) printf("HMMM4: %d %.12e %.12e \n", i, A_data[i].real(), A_data[i].imag());
                //         like_share_middle += (-1./2. * 4 * df * A_data[i] * gcmplx::conj(A_data[i]) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                //         like_share_middle += (-1./2. * 4 * df * E_data[i] * gcmplx::conj(E_data[i]) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
                //     }

                //     // printf("start ind: %d\n A: \n", start_ind_remove);
                //     // for (int i = 0; i < band_here.gb_params.N; i += 1)
                //     // {
                //     //     printf("%.16e + 1j * %.16e, ", A_remove[i].real(), A_remove[i].imag());
                //     // }
                //     // printf("\n\nstart ind: %d\n E: \n", start_ind_remove);
                //     // for (int i = 0; i < band_here.gb_params.N; i += 1)
                //     // {
                //     //     printf("%.16e + 1j * %.16e, ", E_remove[i].real(), E_remove[i].imag());
                //     // }
                //     // printf("\n\n\n");
                    
                //     //printf("middle: %d %e %e %e %e %e %.16e %.16e %.16e %.16e %.16e \n", band_i, like_share_before, like_share_middle, like_share_middle - like_share_before, mcmc_info.L_contribution[band_i].real(), like_share_middle - like_share_before - mcmc_info.L_contribution[band_i].real(), d_h_remove_arr[0].real(), d_h_add_arr[0].real(), add_remove_arr[0].real(), add_add_arr[0].real(), remove_remove_arr[0].real());

                // }
            }
            __syncthreads();
        }
        __syncthreads();

        if ((this_band_inv_temp == 1.0) && (!mcmc_info.is_rj))
        {
            // update the cold chain information
            // need to be careful not to overlap with other bands running simultaneous (every 3 or 4 or something)
            for (int bin_i = 0; bin_i < band_here.band_num_bins; bin_i += 1)
            {
                current_binary_start_index = band_here.band_start_bin_ind + bin_i;
                // get the parameters to add and remove
                
                curr_binary.amp = band_here.gb_params.amp_orig[bin_i];
                curr_binary.f0_ms = band_here.gb_params.f0_ms_orig[bin_i];
                curr_binary.fdot = band_here.gb_params.fdot0_orig[bin_i];
                curr_binary.phi0 = band_here.gb_params.phi0_orig[bin_i];
                curr_binary.cosinc = band_here.gb_params.cosinc_orig[bin_i];
                curr_binary.psi = band_here.gb_params.psi_orig[bin_i];
                curr_binary.lam = band_here.gb_params.lam_orig[bin_i];
                curr_binary.sinbeta = band_here.gb_params.sinbeta_orig[bin_i];

                // adjust for rj
                // if (mcmc_info.is_rj)
                // {
                //     // got to be the opposite becuase it is changed
                //     rj_is_there_already = !stretch_info.inds[band_here.band_start_bin_ind + bin_i];
                //     prop_binary = curr_binary;

                //     if (!rj_is_there_already)
                //     {
                //         curr_binary.amp = 1e-20;
                //     }
                //     else
                //     {
                //         prop_binary.amp = 1e-20;
                //     }
                // }
                // else
                // {
                prop_binary.amp = band_here.gb_params.amp[bin_i];
                prop_binary.f0_ms = band_here.gb_params.f0_ms[bin_i];
                prop_binary.fdot = band_here.gb_params.fdot0[bin_i];
                prop_binary.phi0 = band_here.gb_params.phi0[bin_i];
                prop_binary.cosinc = band_here.gb_params.cosinc[bin_i];
                prop_binary.psi = band_here.gb_params.psi[bin_i];
                prop_binary.lam = band_here.gb_params.lam[bin_i];
                prop_binary.sinbeta = band_here.gb_params.sinbeta[bin_i];
                //}

                // if ((threadIdx.x == 0) && (band_here.band_start_data_ind == 3811)) printf("HMMM25: %d %d %.12e \n", bin_i, start_ind_add, prop_binary.f0_ms);


                curr_binary.transform();
                prop_binary.transform();

                // if (threadIdx.x == 0)
                //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

                __syncthreads();
                build_new_single_waveform<FFT>(
                    wave_remove,
                    &start_ind_remove,
                    curr_binary,
                    bin_i);
                __syncthreads();
                build_new_single_waveform<FFT>(
                    wave_add,
                    &start_ind_add,
                    prop_binary,
                    bin_i_gen);
                __syncthreads();

                // if ((threadIdx.x == 0) && (band_here.band_start_data_ind == 3811)) printf("HMMM35: %d %d %.12e %.12e  %.12e %.12e %d %.12e %.12e  %.12e %.12e \n", bin_i, start_ind_add, prop_binary.f0_ms, A_add[0].real(), A_add[1].real(), A_add[3].real(), start_ind_remove, curr_binary.f0_ms, A_remove[0].real(), A_remove[1].real(), A_remove[3].real());

                // if ((threadIdx.x == 0)) printf("base up: %d %d %d %d %d %d\n", band_i, band_here.data_index, band_here.update_data_index, mcmc_info.accepted_out[band_i], start_ind_add, start_ind_remove);
                // if ((threadIdx.x == 0)) printf("curr up: %d %d %e %e %e %e %e %e %e %e %e\n", band_i, bin_i, curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);
                // if ((threadIdx.x == 0)) printf("prop up: %d %d %e %e %e %e %e %e %e %e %e\n", band_i, bin_i, prop_binary.amp, prop_binary.f0, prop_binary.fdot, prop_binary.fddot, prop_binary.phi0, prop_binary.inc, prop_binary.psi, prop_binary.lam, prop_binary.theta);


                // remove from residual with (+)
                for (int i = threadIdx.x; i < band_here.gb_params.N; i += blockDim.x)
                {
                    
                    j = start_ind_remove + i;
                    if ((j < data.data_length) && (j >= 0))
                    {
                        data.data_A[band_here.update_data_index * data.data_length + j] += A_remove[i];
                        data.data_E[band_here.update_data_index * data.data_length + j] += E_remove[i];
                    }
                    else
                    {
                        printf("BAD 8 ");
                    }
                }
                __syncthreads();
                // remove from residual with (+)
                for (int i = threadIdx.x; i < band_here.gb_params.N; i += blockDim.x)
                {
                    
                    j = start_ind_add + i;
                    if ((j < data.data_length) && (j >= 0))
                    {
                        data.data_A[band_here.update_data_index * data.data_length + j] -= A_add[i];
                        data.data_E[band_here.update_data_index * data.data_length + j] -= E_add[i];
                    }
                    else
                    {
                        printf("BAD 9 ");
                    }
                }
                __syncthreads();
            }
            __syncthreads();
            // add new residual from shared (or global memory)
        }
        __syncthreads();
        // double like_share_after = 0.0;
        // double like_all_check = 0.0;
        // if ((threadIdx.x == 0) && (this_band_inv_temp == 1.0))
        // {
               
        //     for (int i = 0; i < band_here.band_data_lengths; i += 1)
        //     {
        //         if ((i < (band_here.band_interest_start_data_ind - band_here.band_start_data_ind)) || (i > (band_here.band_interest_start_data_ind - band_here.band_start_data_ind + band_here.band_interest_data_lengths))) continue;
        //         // printf("%.16e + 1j * %.16e,", A_data[i].real(), A_data[i].imag());
        //         like_share_after += (-1./2. * 4 * df * A_data[i] * gcmplx::conj(A_data[i]) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
        //         like_share_after += (-1./2. * 4 * df * E_data[i] * gcmplx::conj(E_data[i]) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();

        //         int base_ind; 

        //         if (band_here.data_index > band_here.update_data_index) base_ind = band_here.update_data_index;
        //         else base_ind = band_here.data_index;
        //         cmplx tmp_A = data.data_A[band_here.data_index * data.data_length + band_here.band_start_data_ind + i] + data.data_A[band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i] - data.base_data_A[base_ind * data.data_length + band_here.band_start_data_ind + i];
        //         cmplx tmp_E = data.data_E[band_here.data_index * data.data_length + band_here.band_start_data_ind + i] + data.data_E[band_here.update_data_index * data.data_length + band_here.band_start_data_ind + i] - data.base_data_E[base_ind * data.data_length + band_here.band_start_data_ind + i];
        //         like_all_check += (-1./2. * 4 * df * tmp_A * gcmplx::conj(tmp_A) / data.psd_A[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
        //         like_all_check += (-1./2. * 4 * df * tmp_E * gcmplx::conj(tmp_E) / data.psd_E[band_here.noise_index * data.data_length + band_here.band_start_data_ind + i]).real();
        //         // if ((band_i)) printf("HMMM5: %d %.12e %.12e %.12e %.12e %.12e  \n", i, A_data[i].real(), E_data[i].real(), tmp_A.real(), tmp_E.real(), like_all_check, like_share_after);
                
        //     }
        //     if (abs(like_all_check - like_share_after) > 1e-5) printf("\nYAYAYAYAY: %d %d %d %.12e %.12e %.12e \n", band_i, band_here.band_start_data_ind, band_here.band_data_lengths, like_share_after, like_all_check, like_share_after - like_all_check);
        //     // mcmc_info.L_contribution[band_i] = like_share_after - like_share_before;
        // }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        stretch_info.curand_states[blockIdx.x] = localState;
    }
    __syncthreads();
}

__global__ void setup_curand_states(curandState *curand_states, int n)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        curand_init(1234, i, 0, &curand_states[i]);
    }
}


__global__
void setup_single_bands(SingleBand *bands, BandPackage *band_info, GalacticBinaryParams *gb_params_all, MCMCInfo *mcmc_info)
{
    SingleBand *band_here;
    for (int band_i = threadIdx.x + blockIdx.x * blockDim.x; band_i < band_info->num_bands; band_i += blockDim.x * gridDim.x)
    {
        band_here = &bands[band_i];
        // printf("%d %d %d %d %d %d %d %d %e %e %d %e\n", band_i, band_info->data_index[band_i],
        //     band_info->noise_index[band_i],
        //     band_info->band_start_bin_ind[band_i],
        //     band_info->band_num_bins[band_i],
        //     band_info->band_start_data_ind[band_i],
        //     band_info->band_data_lengths[band_i],
        //     band_info->max_data_store_size,
        //     band_info->fmin_allow[band_i],
        //     band_info->fmax_allow[band_i],
        //     band_info->update_data_index[band_i],
        //     gb_params_all->snr[band_info->band_start_data_ind[band_i]]);

        band_here->setup(
            band_info->loc_index[band_i],
            band_info->data_index[band_i],
            band_info->noise_index[band_i],
            band_info->band_start_bin_ind[band_i],
            band_info->band_num_bins[band_i],
            band_info->band_start_data_ind[band_i],
            band_info->band_data_lengths[band_i],
            band_info->band_interest_start_data_ind[band_i],
            band_info->band_interest_data_lengths[band_i],
            band_info->max_data_store_size,
            band_info->fmin_allow[band_i],
            band_info->fmax_allow[band_i],
            band_info->update_data_index[band_i],
            mcmc_info->band_inv_temperatures_all[band_i],
            band_info->band_ind[band_i],
            band_info->walker_ind[band_i],
            band_info->temp_ind[band_i],
            gb_params_all
        );
    }
}

__global__
void extract_single_bands(SingleBand *bands, BandPackage *band_info, GalacticBinaryParams *gb_params_all, MCMCInfo *mcmc_info, bool is_tempering_swap)
{
    for (int band_i = threadIdx.x + blockIdx.x * blockDim.x; band_i < band_info->num_bands; band_i += blockDim.x * gridDim.x)
    {
        SingleBand band_here = bands[band_i];
        // printf("%d %d %d %d %d %d %d %d %e %e %d %e\n", band_i, band_info->data_index[band_i],
        //     band_info->noise_index[band_i],
        //     band_info->band_start_bin_ind[band_i],
        //     band_info->band_num_bins[band_i],
        //     band_info->band_start_data_ind[band_i],
        //     band_info->band_data_lengths[band_i],
        //     band_info->max_data_store_size,
        //     band_info->fmin_allow[band_i],
        //     band_info->fmax_allow[band_i],
        //     band_info->update_data_index[band_i],
        //     gb_params_all->amp[band_info->band_start_data_ind[band_i]]);

        band_info->loc_index[band_i] = band_here.loc_index;
        band_info->band_start_bin_ind[band_i] = band_here.band_start_bin_ind;
        band_info->band_num_bins[band_i] = band_here.band_num_bins;

        if (is_tempering_swap)
        {
            mcmc_info->L_contribution[band_i] = band_here.current_like - band_here.start_like;
        }
    }
}


// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void make_new_move_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    auto memory_size_waveforms = 3 * N * sizeof(cmplx) + 3 * N * sizeof(cmplx);
    auto memory_size_likelihoods = 5 * FFT::block_dim.x * sizeof(cmplx);
    auto memory_size_data_streams = 2 * inputs.band_info->max_data_store_size * sizeof(cmplx);
    // first is waveforms, second is ll, third is A, E data, fourth is A psd and E psd
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    auto total_memory_size = memory_size_waveforms + memory_size_likelihoods + memory_size_data_streams;

    // Increase max shared memory if needed

    cudaFuncAttributes attr;
    CUDA_CHECK_AND_EXIT(cudaFuncGetAttributes(&attr, make_new_move<FFT>));

    // std::cout << "limit " << attr.maxDynamicSharedSizeBytes << std::endl;

    size_t global_memory_size_per_block, total_global_memory_size, shared_memory_size_mine;
    int num_blocks_per_sm, num_sm; 
    int num_blocks_run;
    cmplx *global_memory_buffer;
    bool use_global_memory;
    if (total_memory_size > 0)
    {
        use_global_memory = true;
        shared_memory_size_mine = memory_size_waveforms + memory_size_likelihoods;

        CUDA_CHECK_AND_EXIT(
            cudaDeviceGetAttribute(&num_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, inputs.device)
        );
        CUDA_CHECK_AND_EXIT(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, inputs.device)
        );
        // std::cout << "check sm " << num_blocks_per_sm << " " << num_sm << std::endl;
        
        global_memory_size_per_block = 2 * inputs.band_info->max_data_store_size * sizeof(cmplx);
        total_global_memory_size = num_blocks_per_sm * num_sm * global_memory_size_per_block;
        num_blocks_run = num_blocks_per_sm * num_sm;
        
        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&global_memory_buffer, total_global_memory_size)
        );
    }
    else
    {
        shared_memory_size_mine = total_memory_size;
        use_global_memory = false;

        num_blocks_run = (inputs.band_info)->num_bands;
    }

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        make_new_move<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    GalacticBinaryParams *params_curr_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&params_curr_d, sizeof(GalacticBinaryParams)));
    
    CUDA_CHECK_AND_EXIT(cudaMemcpy(params_curr_d, inputs.params_curr, sizeof(GalacticBinaryParams), cudaMemcpyHostToDevice));

    DataPackage *data_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&data_d, sizeof(DataPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(data_d, inputs.data, sizeof(DataPackage), cudaMemcpyHostToDevice));

    BandPackage *band_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&band_info_d, sizeof(BandPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(band_info_d, inputs.band_info, sizeof(BandPackage), cudaMemcpyHostToDevice));
    
    MCMCInfo *mcmc_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&mcmc_info_d, sizeof(MCMCInfo)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(mcmc_info_d, inputs.mcmc_info, sizeof(MCMCInfo), cudaMemcpyHostToDevice));
    
    PriorPackage *prior_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&prior_info_d, sizeof(PriorPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(prior_info_d, inputs.prior_info, sizeof(PriorPackage), cudaMemcpyHostToDevice));
    
    int nblocks_curand_setup = std::ceil((num_blocks_run + 32 - 1) / 32);
    // std::cout << "check " << 32 << std::endl; 
    CUDA_CHECK_AND_EXIT(cudaMalloc(&(inputs.stretch_info->curand_states), num_blocks_run * sizeof(curandState)));
    
    // setup the random in-kernel generator
    setup_curand_states<<<nblocks_curand_setup, 32>>>(inputs.stretch_info->curand_states, num_blocks_run);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());


    StretchProposalPackage *stretch_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&stretch_info_d, sizeof(StretchProposalPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(stretch_info_d, inputs.stretch_info, sizeof(StretchProposalPackage), cudaMemcpyHostToDevice));
    
    PeriodicPackage *periodic_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&periodic_info_d, sizeof(PeriodicPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(periodic_info_d, inputs.periodic_info, sizeof(PeriodicPackage), cudaMemcpyHostToDevice));
    
    SingleBand *bands;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&bands, (inputs.band_info)->num_bands * sizeof(SingleBand)));

    int num_blocks_band_setup = std::ceil(((inputs.band_info)->num_bands + 32 - 1) / 32);
    setup_single_bands<<<num_blocks_band_setup, 32>>>(
        bands, band_info_d, params_curr_d, mcmc_info_d
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "before real kernel " << num_blocks_run << std::endl; 
    
    // if (inputs.mcmc_info->is_rj) num_blocks_run = 1;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    make_new_move<FFT><<<num_blocks_run, FFT::block_dim, shared_memory_size_mine>>>(
        data_d,
        band_info_d,
        params_curr_d,
        mcmc_info_d,
        prior_info_d,
        stretch_info_d,
        periodic_info_d,
        bands,
        use_global_memory,
        global_memory_buffer
    );

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "after real kernel " << 32 << std::endl; 
    
    // if (inputs.do_synchronize)
    // {
    //     CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;

    CUDA_CHECK_AND_EXIT(cudaFree(params_curr_d));
    CUDA_CHECK_AND_EXIT(cudaFree(data_d));
    CUDA_CHECK_AND_EXIT(cudaFree(band_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(mcmc_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(prior_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(stretch_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(periodic_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree((inputs.stretch_info->curand_states)));
    CUDA_CHECK_AND_EXIT(cudaFree(bands));

    if (use_global_memory)
    {
        CUDA_CHECK_AND_EXIT(cudaFree(global_memory_buffer));
    }
}

template <unsigned int Arch, unsigned int N>
struct make_new_move_wrap_functor
{
    void operator()(InputInfo inputs) { return make_new_move_wrap<Arch, N>(inputs); }
};

void SharedMemoryMakeNewMove(
    DataPackage *data,
    BandPackage *band_info,
    GalacticBinaryParams *params_curr,
    MCMCInfo *mcmc_info,
    PriorPackage *prior_info,
    StretchProposalPackage *stretch_info,
    PeriodicPackage *periodic_info,
    int device,
    bool do_synchronize
)
{

    InputInfo inputs;

    inputs.data = data;
    inputs.band_info = band_info;
    inputs.params_curr = params_curr;
    inputs.mcmc_info = mcmc_info;
    inputs.prior_info = prior_info;
    inputs.stretch_info = stretch_info;
    inputs.periodic_info = periodic_info;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;

    switch (params_curr->N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<make_new_move_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<make_new_move_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<make_new_move_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<make_new_move_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<make_new_move_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<make_new_move_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<make_new_move_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}


//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////


template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void make_tempering_swap(
    DataPackage *data_in,
    BandPackage *band_info_in,
    GalacticBinaryParams *params_curr_in,
    MCMCInfo *mcmc_info_in,
    PriorPackage *prior_info_in,
    StretchProposalPackage *stretch_info_in,
    PeriodicPackage *periodic_info_in,
    SingleBand *bands,
    int num_swap_setups,
    bool use_global_memory,
    cmplx *global_memory_buffer,
    int min_val,
    int max_val
)
{
    using complex_type = cmplx;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

   
    int tid = threadIdx.x;
    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    DataPackage data = *data_in;
    BandPackage band_info = *band_info_in;
    GalacticBinaryParams params_curr = *params_curr_in;
    // MCMCInfo mcmc_info = *mcmc_info_in;
    // PriorPackage prior_info = *prior_info_in;
    StretchProposalPackage stretch_info = *stretch_info_in;
    // PeriodicPackage periodic_info = *periodic_info;

    double df = data.df;

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    cmplx *wave = (cmplx *)shared_mem;
    cmplx *A = &wave[0];
    cmplx *E = &wave[params_curr.N];

    cmplx *d_h_d_h_arr = &wave[3 * params_curr.N];

    cmplx *A_data;
    cmplx *E_data;

    if (use_global_memory)
    {
        A_data = &global_memory_buffer[(2 * blockIdx.x) * band_info.max_data_store_size];
        E_data = &global_memory_buffer[(2 * blockIdx.x + 1) * band_info.max_data_store_size];
    }
    else
    {
        A_data = &d_h_d_h_arr[FFT::block_dim.x];
        E_data = &A_data[band_info.max_data_store_size];
    }
    __syncthreads();

    int j, k;
    int current_binary_start_index, base_index;
    int data_index_i, data_index_i1, data_index_tmp;
    int noise_index_i, noise_index_i1, noise_index_tmp;
    double ll, paccept;
    double bi, bi1, tmp;
    int tmp_loc_index, tmp_walker, tmp_band_start_bin_ind, tmp_band_num_bins;
    bool accept;
    __shared__ curandState localState;
    __shared__ double random_val;

    if (threadIdx.x == 0)
    {
        localState = stretch_info.curand_states[blockIdx.x];
    }
    __syncthreads();

    cmplx d_h_d_h_temp = 0.0;

    SingleGalacticBinary curr_binary(
        params_curr.N,
        params_curr.T,
        params_curr.Soms_d,
        params_curr.Sa_a,
        params_curr.Amp,
        params_curr.alpha,
        params_curr.sl1,
        params_curr.kn,
        params_curr.sl2
    );
    
    // NUM_SWAP_SETUPS IS KEY
    for (int band_i = blockIdx.x; band_i < num_swap_setups; band_i += gridDim.x)
    {
        for (int temp_i = band_info.ntemps - 1; temp_i >= 0; temp_i -= 1)
        {
            SingleBand *band_here_i = &bands[band_i * band_info.ntemps + temp_i];
            data_index_i = band_here_i->data_index;
            noise_index_i = band_here_i->noise_index;

            for (int i = threadIdx.x; i < band_here_i->band_data_lengths; i += blockDim.x)
            {
                A_data[i] = data.data_A[band_here_i->data_index * data.data_length + band_here_i->band_start_data_ind + i];
                E_data[i] = data.data_E[band_here_i->data_index * data.data_length + band_here_i->band_start_data_ind + i];
                // if (blockIdx.x == gridDim.x - 1) printf("%d %e, %e\n", i, A_data[i].real(), E_data[i].imag());
            }
            __syncthreads();

            for (int bin_i = 0; bin_i < band_here_i->band_num_bins; bin_i += 1)
            {
                current_binary_start_index = band_here_i->band_start_bin_ind + bin_i;
                // get the parameters to add and remove
                
                curr_binary.amp = band_here_i->gb_params.amp[bin_i];
                curr_binary.f0_ms = band_here_i->gb_params.f0_ms[bin_i];
                curr_binary.fdot = band_here_i->gb_params.fdot0[bin_i];
                curr_binary.phi0 = band_here_i->gb_params.phi0[bin_i];
                curr_binary.cosinc = band_here_i->gb_params.cosinc[bin_i];
                curr_binary.psi = band_here_i->gb_params.psi[bin_i];
                curr_binary.lam = band_here_i->gb_params.lam[bin_i];
                curr_binary.sinbeta = band_here_i->gb_params.sinbeta[bin_i];

                curr_binary.transform();

                // if (threadIdx.x == 0)
                //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

                __syncthreads();
                build_new_single_waveform<FFT>(
                    wave,
                    &start_ind,
                    curr_binary,
                    bin_i);
                __syncthreads();

                for (int i = threadIdx.x; i < params_curr.N; i += blockDim.x)
                {
                    // add to residual with (-)
                    j = (start_ind - band_here_i->band_start_data_ind) + i;
                    if ((j < band_info.max_data_store_size) && (j >= 0))
                    {
                        A_data[j] -= A[i];
                        E_data[j] -= E[i];
                    }
                }
                __syncthreads();
            }
            __syncthreads();
            
            d_h_d_h_temp = 0.0;
            for (int i = threadIdx.x; i < band_here_i->band_data_lengths; i += blockDim.x)
            {
                j = band_here_i->band_start_data_ind + i;
                d_h_d_h_temp += gcmplx::conj(A_data[i]) * A_data[i] / data.psd_A[band_here_i->noise_index * data.data_length + j];
                d_h_d_h_temp += gcmplx::conj(E_data[i]) * E_data[i] / data.psd_E[band_here_i->noise_index * data.data_length + j];
            } 
            __syncthreads();
            d_h_d_h_arr[tid] = 4 * df * d_h_d_h_temp;
            __syncthreads();

            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    d_h_d_h_arr[tid] += d_h_d_h_arr[tid + s];
                }
                __syncthreads();
            }
            __syncthreads();

            ll = -1. / 2. * d_h_d_h_arr[0].real();
            __syncthreads();
            band_here_i->current_like = ll;
            band_here_i->start_like = ll;
            // if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("check 2 %d %d %d %d %e\n", band_i, temp_i, band_here_i->walker_ind, band_here_i->temp_ind, band_here_i->current_like);

        }
        __syncthreads();
        
        // GREATER THAN 0
        for (int temp_i = band_info.ntemps - 1; temp_i > 0; temp_i -= 1)
        {
            SingleBand *band_here_i = &bands[band_i * band_info.ntemps + temp_i];
            SingleBand *band_here_i1 = &bands[band_i * band_info.ntemps + temp_i - 1];
            bi = band_here_i->inv_temp;
            bi1 = band_here_i1->inv_temp;

            // Here we are testing the swaps
            data_index_i = band_here_i->data_index;
            noise_index_i = band_here_i->noise_index;

            data_index_i1 = band_here_i1->data_index;
            noise_index_i1 = band_here_i1->noise_index;

            for (int which = 0; which < 2; which += 1)
            {
                // to make accessible outside ifelse
                SingleBand* band_here_tmp = band_here_i;
                if (which == 0)
                {
                    data_index_tmp = data_index_i;
                    noise_index_tmp = noise_index_i;
                    band_here_tmp = band_here_i1;
                }
                else
                {
                    data_index_tmp = data_index_i1;
                    noise_index_tmp = noise_index_i1;
                    band_here_tmp = band_here_i;
                }
                for (int i = threadIdx.x; i < band_here_tmp->band_data_lengths; i += blockDim.x)
                {
                    A_data[i] = data.data_A[data_index_tmp * data.data_length + band_here_tmp->band_start_data_ind + i];
                    E_data[i] = data.data_E[data_index_tmp * data.data_length + band_here_tmp->band_start_data_ind + i];
                    // if (blockIdx.x == gridDim.x - 1) printf("%d %e, %e\n", i, A_data[i].real(), E_data[i].imag());
                }
                __syncthreads();

                for (int bin_i = 0; bin_i < band_here_tmp->band_num_bins; bin_i += 1)
                {
                    current_binary_start_index = band_here_tmp->band_start_bin_ind + bin_i;
                    // get the parameters to add and remove
                    
                    curr_binary.amp = band_here_tmp->gb_params.amp[bin_i];
                    curr_binary.f0_ms = band_here_tmp->gb_params.f0_ms[bin_i];
                    curr_binary.fdot = band_here_tmp->gb_params.fdot0[bin_i];
                    curr_binary.phi0 = band_here_tmp->gb_params.phi0[bin_i];
                    curr_binary.cosinc = band_here_tmp->gb_params.cosinc[bin_i];
                    curr_binary.psi = band_here_tmp->gb_params.psi[bin_i];
                    curr_binary.lam = band_here_tmp->gb_params.lam[bin_i];
                    curr_binary.sinbeta = band_here_tmp->gb_params.sinbeta[bin_i];

                    curr_binary.transform();

                    // if (threadIdx.x == 0)
                    //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

                    __syncthreads();
                    build_new_single_waveform<FFT>(
                        wave,
                        &start_ind,
                        curr_binary,
                        bin_i);
                    __syncthreads();

                    for (int i = threadIdx.x; i < params_curr.N; i += blockDim.x)
                    {
                        // add to residual with (-)
                        j = (start_ind - band_here_tmp->band_start_data_ind) + i;
                        if ((j < band_info.max_data_store_size) && (j >= 0))
                        {
                            A_data[j] -= A[i];
                            E_data[j] -= E[i];
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
                
                d_h_d_h_temp = 0.0;
                for (int i = threadIdx.x; i < band_here_tmp->band_data_lengths; i += blockDim.x)
                {
                    j = band_here_tmp->band_start_data_ind + i;
                    d_h_d_h_temp += gcmplx::conj(A_data[i]) * A_data[i] / data.psd_A[noise_index_tmp * data.data_length + j];
                    d_h_d_h_temp += gcmplx::conj(E_data[i]) * E_data[i] / data.psd_E[noise_index_tmp * data.data_length + j];
                } 
                __syncthreads();
                d_h_d_h_arr[tid] = 4 * df * d_h_d_h_temp;
                __syncthreads();

                for (unsigned int s = 1; s < blockDim.x; s *= 2)
                {
                    if (tid % (2 * s) == 0)
                    {
                        d_h_d_h_arr[tid] += d_h_d_h_arr[tid + s];
                    }
                    __syncthreads();
                }
                __syncthreads();

                ll = -1. / 2. * d_h_d_h_arr[0].real();
                __syncthreads();
            
                band_here_tmp->swapped_like = ll;
            }
            
            paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);
            __syncthreads();
            if (threadIdx.x == 0)
            {
                random_val = log(curand_uniform_double(&localState));
            }
            __syncthreads();
            accept = paccept > random_val;
            // if ((band_i == 855) && (threadIdx.x == 0)) printf("check 3 %d %d %d %d %d %d %d %d %d %d \n %e %e %e %e %e %e %e %e %d\n\n", band_i, temp_i, band_here_i->loc_index, band_here_i->walker_ind, band_here_i->temp_ind, band_here_i->band_num_bins, band_here_i1->loc_index, band_here_i1->walker_ind, band_here_i1->temp_ind, band_here_i1->band_num_bins, bi, bi1, band_here_i->current_like, band_here_i->swapped_like, band_here_i1->current_like, band_here_i1->swapped_like, paccept, random_val, int(accept));
             __syncthreads();
            band_info.swaps_proposed[band_i * (band_info.ntemps - 1) + temp_i - 1] += 1;
            if ((band_i < min_val) || (band_i > max_val)) accept = false;
            if (accept)
            {
                if (threadIdx.x == 0){

                    band_info.swaps_accepted[band_i * (band_info.ntemps - 1) + temp_i - 1] += 1;
                    // switch the log like values
                    band_here_i->current_like = band_here_i1->swapped_like;
                    band_here_i1->current_like = band_here_i->swapped_like;

                    tmp_loc_index = band_here_i->loc_index;
                    band_here_i->loc_index = band_here_i1->loc_index;
                    band_here_i1->loc_index = tmp_loc_index;

                    GalacticBinaryParams tmp_bin_params = band_here_i->gb_params;
                    band_here_i->gb_params = band_here_i1->gb_params;
                    band_here_i1->gb_params = tmp_bin_params;

                    tmp_band_start_bin_ind = band_here_i->band_start_bin_ind;
                    band_here_i->band_start_bin_ind = band_here_i1->band_start_bin_ind;
                    band_here_i1->band_start_bin_ind = tmp_band_start_bin_ind;

                    tmp_band_num_bins = band_here_i->band_num_bins;
                    band_here_i->band_num_bins = band_here_i1->band_num_bins;
                    band_here_i1->band_num_bins = tmp_band_num_bins;
                }
                __syncthreads();
            }

            if ((temp_i == 1) && (accept))
            {
                // update the cold chain information
                // need to be careful not to overlap with other bands running simultaneous (every 3 or 4 or something)
                for (int bin_i = 0; bin_i < band_here_i->band_num_bins; bin_i += 1)
                {
                    current_binary_start_index = band_here_i->band_start_bin_ind + bin_i;
                    // get the parameters to add and remove
                    
                    curr_binary.amp = band_here_i->gb_params.amp[bin_i];
                    curr_binary.f0_ms = band_here_i->gb_params.f0_ms[bin_i];
                    curr_binary.fdot = band_here_i->gb_params.fdot0[bin_i];
                    curr_binary.phi0 = band_here_i->gb_params.phi0[bin_i];
                    curr_binary.cosinc = band_here_i->gb_params.cosinc[bin_i];
                    curr_binary.psi = band_here_i->gb_params.psi[bin_i];
                    curr_binary.lam = band_here_i->gb_params.lam[bin_i];
                    curr_binary.sinbeta = band_here_i->gb_params.sinbeta[bin_i];

                    curr_binary.transform();

                    // if (threadIdx.x == 0)
                    //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

                    __syncthreads();
                    build_new_single_waveform<FFT>(
                        wave,
                        &start_ind,
                        curr_binary,
                        bin_i);
                    __syncthreads();

                    // remove from residual with (+)
                    for (int i = threadIdx.x; i < band_here_i->gb_params.N; i += blockDim.x)
                    {
                        
                        j = start_ind + i;
                        if ((j < data.data_length) && (j >= 0))
                        {
                            // NEEDS to be i1->update_data_index
                            data.data_A[band_here_i1->update_data_index * data.data_length + j] += A[i];
                            data.data_E[band_here_i1->update_data_index * data.data_length + j] += E[i];
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
                
                // update the cold chain information
                // need to be careful not to overlap with other bands running simultaneous (every 3 or 4 or something)
                for (int bin_i = 0; bin_i < band_here_i1->band_num_bins; bin_i += 1)
                {
                    current_binary_start_index = band_here_i1->band_start_bin_ind + bin_i;
                    // get the parameters to add and remove
                    
                    curr_binary.amp = band_here_i1->gb_params.amp[bin_i];
                    curr_binary.f0_ms = band_here_i1->gb_params.f0_ms[bin_i];
                    curr_binary.fdot = band_here_i1->gb_params.fdot0[bin_i];
                    curr_binary.phi0 = band_here_i1->gb_params.phi0[bin_i];
                    curr_binary.cosinc = band_here_i1->gb_params.cosinc[bin_i];
                    curr_binary.psi = band_here_i1->gb_params.psi[bin_i];
                    curr_binary.lam = band_here_i1->gb_params.lam[bin_i];
                    curr_binary.sinbeta = band_here_i1->gb_params.sinbeta[bin_i];

                    curr_binary.transform();

                    // if (threadIdx.x == 0)
                    //     printf("%e %e %e %e %e %e %e %e %e\n", curr_binary.amp, curr_binary.f0, curr_binary.fdot, curr_binary.fddot, curr_binary.phi0, curr_binary.inc, curr_binary.psi, curr_binary.lam, curr_binary.theta);

                    __syncthreads();
                    build_new_single_waveform<FFT>(
                        wave,
                        &start_ind,
                        curr_binary,
                        bin_i);
                    __syncthreads();

                    // add to residual with (-)
                    for (int i = threadIdx.x; i < band_here_i1->gb_params.N; i += blockDim.x)
                    {
                        
                        j = start_ind + i;
                        if ((j < data.data_length) && (j >= 0))
                        {
                            data.data_A[band_here_i1->update_data_index * data.data_length + j] -= A[i];
                            data.data_E[band_here_i1->update_data_index * data.data_length + j] -= E[i];
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        stretch_info.curand_states[blockIdx.x] = localState;
    }
    __syncthreads();
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void make_tempering_swap_wrap(InputInfo inputs)
{
    using namespace cufftdx;

    if (inputs.device >= 0)
    {
        // set the device
        CUDA_CHECK_AND_EXIT(cudaSetDevice(inputs.device));
    }

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,

    using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    auto memory_size_waveforms = 3 * N * sizeof(cmplx);
    auto memory_size_likelihoods = 1 * FFT::block_dim.x * sizeof(cmplx);
    auto memory_size_data_streams = 2 * inputs.band_info->max_data_store_size * sizeof(cmplx);
    // first is waveforms, second is ll, third is A, E data, fourth is A psd and E psd
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    auto total_memory_size = memory_size_waveforms + memory_size_likelihoods + memory_size_data_streams;

    // Increase max shared memory if needed

    cudaFuncAttributes attr;
    CUDA_CHECK_AND_EXIT(cudaFuncGetAttributes(&attr, make_tempering_swap<FFT>));

    // std::cout << "limit " << attr.maxDynamicSharedSizeBytes << std::endl;

    size_t global_memory_size_per_block, total_global_memory_size, shared_memory_size_mine;
    int num_blocks_per_sm, num_sm; 
    int num_blocks_run;
    cmplx *global_memory_buffer;
    bool use_global_memory;
    if (total_memory_size > 0)
    {
        use_global_memory = true;
        shared_memory_size_mine = memory_size_waveforms + memory_size_likelihoods;

        CUDA_CHECK_AND_EXIT(
            cudaDeviceGetAttribute(&num_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, inputs.device)
        );
        CUDA_CHECK_AND_EXIT(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, inputs.device)
        );
        // std::cout << "check sm " << num_blocks_per_sm << " " << num_sm << std::endl;
        
        global_memory_size_per_block = 2 * inputs.band_info->max_data_store_size * sizeof(cmplx);
        total_global_memory_size = num_blocks_per_sm * num_sm * global_memory_size_per_block;
        num_blocks_run = num_blocks_per_sm * num_sm;
        
        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&global_memory_buffer, total_global_memory_size)
        );
    }
    else
    {
        shared_memory_size_mine = total_memory_size;
        use_global_memory = false;

        num_blocks_run = inputs.num_swap_setups;
    }

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        make_tempering_swap<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    GalacticBinaryParams *params_curr_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&params_curr_d, sizeof(GalacticBinaryParams)));
    
    CUDA_CHECK_AND_EXIT(cudaMemcpy(params_curr_d, inputs.params_curr, sizeof(GalacticBinaryParams), cudaMemcpyHostToDevice));

    DataPackage *data_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&data_d, sizeof(DataPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(data_d, inputs.data, sizeof(DataPackage), cudaMemcpyHostToDevice));

    BandPackage *band_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&band_info_d, sizeof(BandPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(band_info_d, inputs.band_info, sizeof(BandPackage), cudaMemcpyHostToDevice));
    
    MCMCInfo *mcmc_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&mcmc_info_d, sizeof(MCMCInfo)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(mcmc_info_d, inputs.mcmc_info, sizeof(MCMCInfo), cudaMemcpyHostToDevice));
    
    PriorPackage *prior_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&prior_info_d, sizeof(PriorPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(prior_info_d, inputs.prior_info, sizeof(PriorPackage), cudaMemcpyHostToDevice));
    
    int nblocks_curand_setup = std::ceil((num_blocks_run + 32 - 1) / 32);
    // std::cout << "check " << 32 << std::endl; 
    CUDA_CHECK_AND_EXIT(cudaMalloc(&(inputs.stretch_info->curand_states), num_blocks_run * sizeof(curandState)));
    
    // setup the random in-kernel generator
    setup_curand_states<<<nblocks_curand_setup, 32>>>(inputs.stretch_info->curand_states, num_blocks_run);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "check middle" << 32 << std::endl; 

    StretchProposalPackage *stretch_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&stretch_info_d, sizeof(StretchProposalPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(stretch_info_d, inputs.stretch_info, sizeof(StretchProposalPackage), cudaMemcpyHostToDevice));
    
    PeriodicPackage *periodic_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&periodic_info_d, sizeof(PeriodicPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(periodic_info_d, inputs.periodic_info, sizeof(PeriodicPackage), cudaMemcpyHostToDevice));
    
    SingleBand *bands;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&bands, (inputs.band_info)->num_bands * sizeof(SingleBand)));

    // std::cout << "before setup kernel " << num_blocks_run << std::endl; 
    int num_blocks_band_setup = std::ceil(((inputs.band_info)->num_bands + 32 - 1) / 32);
    setup_single_bands<<<num_blocks_band_setup, 32>>>(
        bands, band_info_d, params_curr_d, mcmc_info_d
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "before real kernel " << num_blocks_run << std::endl; 
    
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    make_tempering_swap<FFT><<<num_blocks_run, FFT::block_dim, shared_memory_size_mine>>>(
        data_d,
        band_info_d,
        params_curr_d,
        mcmc_info_d,
        prior_info_d,
        stretch_info_d,
        periodic_info_d,
        bands,
        inputs.num_swap_setups,
        use_global_memory,
        global_memory_buffer,
        inputs.min_val,
        inputs.max_val
    );

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "after real kernel " << 32 << std::endl; 

    extract_single_bands<<<num_blocks_band_setup, 32>>>(
        bands, band_info_d, params_curr_d, mcmc_info_d, true
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    
    // if (inputs.do_synchronize)
    // {
    //     CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // }

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // std::cout << shared_memory_size << std::endl;
    // std::cout << "Success" <<  std::endl;
    CUDA_CHECK_AND_EXIT(cudaFree(params_curr_d));
    CUDA_CHECK_AND_EXIT(cudaFree(data_d));
    CUDA_CHECK_AND_EXIT(cudaFree(band_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(mcmc_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(prior_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(stretch_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree(periodic_info_d));
    CUDA_CHECK_AND_EXIT(cudaFree((inputs.stretch_info->curand_states)));
    CUDA_CHECK_AND_EXIT(cudaFree(bands));

    if (use_global_memory)
    {
        CUDA_CHECK_AND_EXIT(cudaFree(global_memory_buffer));
    }
}

template <unsigned int Arch, unsigned int N>
struct make_tempering_swap_wrap_functor
{
    void operator()(InputInfo inputs) { return make_tempering_swap_wrap<Arch, N>(inputs); }
};

void SharedMemoryMakeTemperingMove(
    DataPackage *data,
    BandPackage *band_info,
    GalacticBinaryParams *params_curr,
    MCMCInfo *mcmc_info,
    PriorPackage *prior_info,
    StretchProposalPackage *stretch_info,
    PeriodicPackage *periodic_info,
    int num_swap_setups,
    int device,
    bool do_synchronize,
    int min_val,
    int max_val
)
{

    InputInfo inputs;

    inputs.data = data;
    inputs.band_info = band_info;
    inputs.params_curr = params_curr;
    inputs.mcmc_info = mcmc_info;
    inputs.prior_info = prior_info;
    inputs.stretch_info = stretch_info;
    inputs.periodic_info = periodic_info;
    inputs.num_swap_setups = num_swap_setups;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.min_val = min_val;
    inputs.max_val = max_val;

    switch (params_curr->N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<make_tempering_swap_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<make_tempering_swap_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<make_tempering_swap_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<make_tempering_swap_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<make_tempering_swap_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<make_tempering_swap_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<make_tempering_swap_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}


//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

const double lisaL = 2.5e9;           // LISA's arm meters
const double lisaLT = lisaL / Clight; // LISA's armn in sec

__device__ void lisanoises(double *Spm, double *Sop, double f, double Soms_d_in, double Sa_a_in, bool return_relative_frequency)
{
    double frq = f;
    // Acceleration noise
    // In acceleration
    double Sa_a = Sa_a_in * (1.0 + pow((0.4e-3 / frq), 2)) * (1.0 + pow((frq / 8e-3), 4));
    // In displacement
    double Sa_d = Sa_a * pow((2.0 * M_PI * frq), (-4.0));
    // In relative frequency unit
    double Sa_nu = Sa_d * pow((2.0 * M_PI * frq / Clight), 2);

    if (return_relative_frequency)
    {
        *Spm = Sa_nu;
    }
    else
    {
        *Spm = Sa_d;
    }

    // Optical Metrology System
    // In displacement
    double Soms_d = Soms_d_in * (1.0 + pow((2.0e-3 / f), 4));
    // In relative frequency unit
    double Soms_nu = Soms_d * pow((2.0 * M_PI * frq / Clight), 2);
    *Sop = Soms_nu;

    if (return_relative_frequency)
    {
        *Sop = Soms_nu;
    }
    else
    {
        *Sop = Soms_d;
    }

    // if ((threadIdx.x == 10) && (blockIdx.x == 0) && (blockIdx.y == 0))
    //     printf("%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e \n", frq, Sa_a_in, Soms_d_in, Sa_a, Sa_d, Sa_nu, *Spm, Soms_d, Soms_nu, *Sop);
}

__device__ double SGal(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_out = (Amp * exp(-(pow(fr, alpha)) * sl1) * (pow(fr, (-7.0 / 3.0))) * 0.5 * (1.0 + tanh(-(fr - kn) * sl2)));
    return Sgal_out;
}

__device__ double GalConf(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_int = SGal(fr, Amp, alpha, sl1, kn, sl2);
    return Sgal_int;
}

__device__ double WDconfusionX(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double t = 4.0 * pow(x, 2) * pow(sin(x), 2);

    double Sg_sens = GalConf(f, Amp, alpha, sl1, kn, sl2);

    // t = 4 * x**2 * xp.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
    return t * Sg_sens;
}

__device__ double WDconfusionAE(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double SgX = WDconfusionX(f, Amp, alpha, sl1, kn, sl2);
    return 1.5 * SgX;
}

__device__ double lisasens(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double Sa_d, Sop;
    bool return_relative_frequency = false;
    lisanoises(&Sa_d, &Sop, f, Soms_d_in, Sa_a_in, return_relative_frequency);

    double ALL_m = sqrt(4.0 * Sa_d + Sop);
    // Average the antenna response
    double AvResp = sqrt(5.);
    // Projection effect
    double Proj = 2.0 / sqrt(3.);
    // Approximative transfert function
    double f0 = 1.0 / (2.0 * lisaLT);
    double a = 0.41;
    double T = sqrt(1. + pow((f / (a * f0)), 2));
    double Sens = pow((AvResp * Proj * T * ALL_m / lisaL), 2);

    if (Amp > 0.0)
    {
        Sens += GalConf(f, Amp, alpha, sl1, kn, sl2);
    }

    return Sens;
}

__device__ double noisepsd_AE(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double Spm, Sop;
    bool return_relative_frequency = true;
    lisanoises(&Spm, &Sop, f, Soms_d_in, Sa_a_in, return_relative_frequency);

    double Sa = (8.0 * (sin(x) * sin(x)) * (2.0 * Spm * (3.0 + 2.0 * cos(x) + cos(2 * x)) + Sop * (2.0 + cos(x))));

    if (Amp > 0.0)
    {
        Sa += WDconfusionAE(f, Amp, alpha, sl1, kn, sl2);
    }

    return Sa;
    //,
}

#define NUM_THREADS_LIKE 256
__global__ void psd_likelihood(double *like_contrib, double *f_arr, cmplx *A_data, cmplx *E_data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    __shared__ double like_vals[NUM_THREADS_LIKE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int data_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    cmplx d_A, d_E;
    double f, Sn_A, Sn_E;
    double inner_product;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    {
        data_index = data_index_all[psd_i];

        A_Soms_d_in = A_Soms_d_in_all[psd_i];
        A_Sa_a_in = A_Sa_a_in_all[psd_i];
        E_Soms_d_in = E_Soms_d_in_all[psd_i];
        E_Sa_a_in = E_Sa_a_in_all[psd_i];
        Amp = Amp_all[psd_i];
        alpha = alpha_all[psd_i];
        sl1 = sl1_all[psd_i];
        kn = kn_all[psd_i];
        sl2 = sl2_all[psd_i];

        for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
        {
            like_vals[i] = 0.0;
        }
        __syncthreads();

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < data_length; i += blockDim.x * gridDim.x)
        {
            d_A = A_data[data_index * data_length + i];
            d_E = E_data[data_index * data_length + i];
            f = f_arr[i];
            if (f == 0.0)
            {
                f = df; // TODO switch this?
            }

            A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
            A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
            E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
            E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
            Sn_A = noisepsd_AE(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
            Sn_E = noisepsd_AE(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

            inner_product = (4.0 * ((gcmplx::conj(d_A) * d_A / Sn_A) + (gcmplx::conj(d_E) * d_E / Sn_E)).real() * df);
            like_vals[tid] += -1.0 / 2.0 * inner_product - log(Sn_A) - log(Sn_E);
            // if ((psd_i == 0) && (i > 10) && (i < 20)) printf("%d %.12e %.12e %.12e %.12e %.12e %.12e %.12e \n", i, inner_product, Sn_A, Sn_E, d_A.real(), d_A.imag(), d_E.real(), d_E.imag());
        }
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            like_contrib[psd_i * num_blocks + bid] = like_vals[0];
        }
        __syncthreads();
    }
}

#define NUM_THREADS_LIKE 256
__global__ void like_sum_from_contrib(double *like_contrib_final, double *like_contrib, int num_blocks_orig, int num_psds)
{
    __shared__ double like_vals[NUM_THREADS_LIKE];
    int tid = threadIdx.x;

    for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    {
        for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
        {
            like_vals[i] = 0.0;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < num_blocks_orig; i += blockDim.x)
        {
            like_vals[tid] += like_contrib[psd_i * num_blocks_orig + i];
        }
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            like_contrib_final[psd_i] = like_vals[0];
        }
        __syncthreads();
    }
}

void psd_likelihood_wrap(double *like_contrib_final, double *f_arr, cmplx *A_data, cmplx *E_data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                         double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    double *like_contrib;

    int num_blocks = std::ceil((data_length + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&like_contrib, num_psds * num_blocks * sizeof(double)));

    dim3 grid(num_blocks, num_psds, 1);

    psd_likelihood<<<grid, NUM_THREADS_LIKE>>>(like_contrib, f_arr, A_data, E_data, data_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, df, data_length, num_data, num_psds);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    dim3 grid_gather(1, num_psds, 1);
    like_sum_from_contrib<<<grid_gather, NUM_THREADS_LIKE>>>(like_contrib_final, like_contrib, num_blocks, num_psds);
    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaFree(like_contrib));
}

#define PDF_NUM_THREADS 32
#define PDF_NDIM 6

__global__
void compute_logpdf(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js,
                    int num_points, int *start_index, int num_components)
{
    int start_index_here, end_index_here, component_here, j;
    __shared__ double point_here[PDF_NDIM];
    __shared__ double log_sum_arr[PDF_NUM_THREADS];
    __shared__ double max_log_sum_arr[PDF_NUM_THREADS];
    __shared__ double max_log_all;
    __shared__ double max_tmp;
    __shared__ double total_log_sum;
    __shared__ double current_log_sum;
    double mean_here[PDF_NDIM];
    double invcov_here[PDF_NDIM][PDF_NDIM];
    double mins_here[PDF_NDIM];
    double maxs_here[PDF_NDIM];
    double point_mapped[PDF_NDIM];
    double diff_from_mean[PDF_NDIM];
    double log_main_part, log_norm_factor, log_weighted_pdf;
    double det_here, log_J_here, weight_here, tmp;
    double kernel_sum = 0.0;
    int tid = threadIdx.x;
    
    for (int i = blockIdx.x; i < num_points; i += gridDim.x)
    {   
        if (tid == 0){total_log_sum = -1e300;}
        __syncthreads();
        for (int k = threadIdx.x; k < PDF_NDIM; k += blockDim.x)
        {
            point_here[k] = points[i * PDF_NDIM + k];
        }
        __syncthreads();

        start_index_here = start_index[i];
        end_index_here = start_index[i + 1];

        while (start_index_here < end_index_here)
        {
            __syncthreads();
            log_sum_arr[tid] = -1e300;
            max_log_sum_arr[tid] = -1e300;
            __syncthreads();

            j = start_index_here + tid;
            __syncthreads();
            if (j < end_index_here)
            {
                // make sure if threads are not used that they do not affect the sum
                component_here = component_index[j];
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    mins_here[k] = mins[k * num_components + component_here];
                    maxs_here[k] = maxs[k * num_components + component_here];
                    mean_here[k] = means[k * num_components + component_here];
                    for (int l = 0; l < PDF_NDIM; l += 1)
                    {
                        invcov_here[k][l] = invcovs[(k * PDF_NDIM + l) * num_components + component_here];
                    }
                }
                det_here = dets[component_here];
                log_J_here = log_Js[component_here];
                weight_here = weights[component_here];
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    point_mapped[k] = ((point_here[k] - mins_here[k]) / (maxs_here[k] - mins_here[k])) * 2. - 1.;
                    diff_from_mean[k] = point_mapped[k] - mean_here[k];
                    // if ((blockIdx.x == 0) && (tid == 0)) printf("%d %d %.10e %.10e\n", component_here, k, point_mapped[k],diff_from_mean[k]);
                }
                // calculate (x-mu)^T * invcov * (x-mu)
                kernel_sum = 0.0;
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    tmp = 0.0;
                    for (int l = 0; l < PDF_NDIM; l += 1)
                    {
                        tmp += invcov_here[k][l] * diff_from_mean[l];
                    }
                    kernel_sum += diff_from_mean[k] * tmp;
                }
                log_main_part = -1./2. * kernel_sum;
                log_norm_factor = (double(PDF_NDIM) / 2.) * log(2 * M_PI) + (1. / 2.) * log(det_here);
                log_weighted_pdf = log(weight_here) + log_norm_factor + log_main_part;

                log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                max_log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, log_weighted_pdf);
                
            }
            __syncthreads();
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    max_log_sum_arr[tid] = max(max_log_sum_arr[tid], max_log_sum_arr[tid + s]);
                }
                __syncthreads();
            }
            __syncthreads();
            // store max in shared value
            if (tid == 0){max_log_all = max_log_sum_arr[tid];}
            __syncthreads();
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, max_log_all);
            
            // subtract max from every value and take exp
            log_sum_arr[tid] = exp(log_sum_arr[tid] - max_log_all);
            __syncthreads();
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    log_sum_arr[tid] += log_sum_arr[tid + s];
                }
                __syncthreads();
            }
            __syncthreads();
            // do it again to add next round if there
            if (tid == 0)
            {
                // finish up initial computation
                current_log_sum = max_log_all + log(log_sum_arr[0]);
                //if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e %.10e\n", component_here, current_log_sum, total_log_sum);

                // start new computation
                // get max
                max_tmp = max(current_log_sum, total_log_sum);
                // subtract max from all values and take exp
                current_log_sum = exp(current_log_sum - max_tmp);
                total_log_sum = exp(total_log_sum - max_tmp);
                // sum values, take log and add back max
                total_log_sum = max_tmp + log(current_log_sum + total_log_sum);
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, total_log_sum);
            }             
            start_index_here += PDF_NUM_THREADS;
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %d\n", start_index_here, end_index_here);
            __syncthreads();
        }
        logpdf_out[i] = total_log_sum;
    }
}

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim)
{
    if (ndim != PDF_NDIM){throw std::invalid_argument("ndim in does not equal NDIM_PDF in GPU code.");}

    compute_logpdf<<<num_points, PDF_NUM_THREADS>>>(logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js,
                    num_points, start_index, num_components);
    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}

CUDA_HOSTDEV
GalacticBinaryParams::GalacticBinaryParams(
    double* amp_,
    double* f0_ms_, 
    double* fdot0_, 
    double* phi0_, 
    double* cosinc_,
    double* psi_, 
    double* lam_,
    double* sinbeta_,
    double* amp_orig_,
    double* f0_ms_orig_, 
    double* fdot0_orig_, 
    double* phi0_orig_, 
    double* cosinc_orig_,
    double* psi_orig_, 
    double* lam_orig_,
    double* sinbeta_orig_,
    double T_, 
    double dt_,
    int N_,
    int num_bin_all_,
    int start_freq_ind_,
    double Soms_d_,
    double Sa_a_, 
    double Amp_,
    double alpha_,
    double sl1_,
    double kn_,
    double sl2_
)
{
    amp = amp_;
    f0_ms = f0_ms_;
    fdot0 = fdot0_;
    phi0 = phi0_;
    cosinc = cosinc_;
    psi = psi_;
    lam = lam_;
    sinbeta = sinbeta_;
    amp_orig = amp_orig_;
    f0_ms_orig = f0_ms_orig_;
    fdot0_orig = fdot0_orig_;
    phi0_orig = phi0_orig_;
    cosinc_orig = cosinc_orig_;
    psi_orig = psi_orig_;
    lam_orig = lam_orig_;
    sinbeta_orig = sinbeta_orig_;
    T = T_;
    dt = dt_;
    N = N_;
    num_bin_all = num_bin_all_;
    start_freq_ind = start_freq_ind_;
    Soms_d = Soms_d_;
    Sa_a = Sa_a_;
    Amp = Amp_;
    alpha = alpha_;
    sl1 = sl1_;
    sl2 = sl2_;
    kn = kn_;
}

DataPackage::DataPackage(
    cmplx* data_A_,
    cmplx* data_E_,
    cmplx* base_data_A_,
    cmplx* base_data_E_,
    double* psd_A_,
    double* psd_E_,
    double* lisasens_A_,
    double* lisasens_E_,
    double df_,
    int data_length_,
    int num_data_,
    int num_psd_
)
{
    data_A = data_A_;
    data_E = data_E_;
    base_data_A = base_data_A_;
    base_data_E = base_data_E_;
    psd_A = psd_A_;
    psd_E = psd_E_;
    lisasens_A = lisasens_A_;
    lisasens_E = lisasens_E_;
    df = df_;
    data_length = data_length_;
    num_data = num_data_;
    num_psd = num_psd_;
}

BandPackage::BandPackage(
    int *loc_index_,
    int *data_index_,
    int *noise_index_,
    int *band_start_bin_ind_,
    int *band_num_bins_,
    int *band_start_data_ind_,
    int *band_data_lengths_,
    int *band_interest_start_data_ind_,
    int *band_interest_data_lengths_,
    int num_bands_,
    int max_data_store_size_,
    double *fmin_allow_,
    double *fmax_allow_,
    int *update_data_index_,
    int ntemps_,
    int *band_ind_,
    int *walker_ind_,
    int *temp_ind_,
    int *swaps_proposed_,
    int *swaps_accepted_
)
{
    loc_index = loc_index_;
    data_index = data_index_;
    noise_index = noise_index_;
    band_start_bin_ind = band_start_bin_ind_;
    band_num_bins = band_num_bins_;
    band_start_data_ind = band_start_data_ind_;
    band_data_lengths = band_data_lengths_;
    band_interest_start_data_ind = band_interest_start_data_ind_;
    band_interest_data_lengths = band_interest_data_lengths_;
    num_bands = num_bands_;
    max_data_store_size = max_data_store_size_;
    fmin_allow = fmin_allow_;
    fmax_allow = fmax_allow_;
    update_data_index = update_data_index_;
    ntemps = ntemps_;
    band_ind = band_ind_;
    walker_ind = walker_ind_;
    temp_ind = temp_ind_;
    swaps_proposed = swaps_proposed_;
    swaps_accepted = swaps_accepted_;
}

MCMCInfo::MCMCInfo(
    cmplx *L_contribution_,
    cmplx *p_contribution_,
    double *prior_all_curr_,
    int *accepted_out_,
    double *band_inv_temperatures_all_,
    bool is_rj_,
    bool phase_maximize_,
    double snr_lim_
)
{
    L_contribution = L_contribution_;
    p_contribution = p_contribution_;
    prior_all_curr = prior_all_curr_;
    accepted_out = accepted_out_;
    band_inv_temperatures_all = band_inv_temperatures_all_;
    is_rj = is_rj_;
    phase_maximize = phase_maximize_;
    snr_lim = snr_lim_;
}

PriorPackage::PriorPackage(
    double rho_star_,
    double f0_min_,
    double f0_max_,
    double fdot_min_,
    double fdot_max_,
    double phi0_min_,
    double phi0_max_,
    double cosinc_min_,
    double cosinc_max_,
    double psi_min_,
    double psi_max_,
    double lam_min_,
    double lam_max_,
    double sinbeta_min_,
    double sinbeta_max_
)
{
    rho_star = rho_star_;
    f0_min = f0_min_;
    f0_max = f0_max_;
    fdot_min = fdot_min_;
    fdot_max = fdot_max_;
    phi0_min = phi0_min_;
    phi0_max = phi0_max_;
    cosinc_min = cosinc_min_;
    cosinc_max = cosinc_max_;
    psi_min = psi_min_;
    psi_max = psi_max_;
    lam_min = lam_min_;
    lam_max = lam_max_;
    sinbeta_min = sinbeta_min_;
    sinbeta_max = sinbeta_max_;
}

CUDA_DEV
double PriorPackage::get_prior_val(
    SingleGalacticBinary gb,
    int num_func,
    double sens_val_prior
)
{
    double prior_val_out = 0.0;

    gb.snr = gb.amp_transform(gb.amp, gb.f0_ms / 1e3, sens_val_prior);
    if ((num_func < 0) || (num_func > 7) || (num_func == 0))
        prior_val_out += get_amp_prior(gb);
    if ((num_func < 0) || (num_func > 7) || (num_func == 1))
        prior_val_out += get_f0_prior(gb.f0_ms);
    if ((num_func < 0) || (num_func > 7) || (num_func == 2))
        prior_val_out += get_fdot_prior(gb.fdot);
    if ((num_func < 0) || (num_func > 7) || (num_func == 3))
        prior_val_out += get_phi0_prior(gb.phi0);
    if ((num_func < 0) || (num_func > 7) || (num_func == 4))
        prior_val_out += get_cosinc_prior(gb.cosinc);
    if ((num_func < 0) || (num_func > 7) || (num_func == 5))
        prior_val_out += get_psi_prior(gb.psi);
    if ((num_func < 0) || (num_func > 7) || (num_func == 6))
        prior_val_out += get_lam_prior(gb.lam);
    if ((num_func < 0) || (num_func > 7) || (num_func == 7))
        prior_val_out += get_sinbeta_prior(gb.sinbeta);

    return prior_val_out;
}

CUDA_DEV double PriorPackage::get_amp_prior(SingleGalacticBinary gb)
{
    
    double rho = gb.snr; // amp * 1e21;
    double Jac = gb.snr / gb.amp;
    if (rho > 0.0)
    {
        return log(abs(Jac)) + log(3. * rho / (4. * pow(rho_star, 2.) * pow((1. + rho / (4 * rho_star)), 5)));
    }
    else {return -INFINITY;}
}

CUDA_DEV double PriorPackage::uniform_dist_logpdf(const double x, const double x_min, const double x_max)
{
    if ((x >= x_min) && (x <= x_max))
    {
        return log(1.0 / (x_max - x_min));
    }
    else
    {
        return -INFINITY;
    }
}

CUDA_DEV double PriorPackage::get_f0_prior(const double f0)
{
    return uniform_dist_logpdf(f0, f0_min, f0_max);
}

CUDA_DEV double PriorPackage::get_fdot_prior(const double fdot)
{
    return uniform_dist_logpdf(fdot, fdot_min, fdot_max);
}

CUDA_DEV double PriorPackage::get_phi0_prior(const double phi0)
{
    return uniform_dist_logpdf(phi0, phi0_min, phi0_max);
}

CUDA_DEV double PriorPackage::get_cosinc_prior(const double cosinc)
{
    return uniform_dist_logpdf(cosinc, cosinc_min, cosinc_max);
}

CUDA_DEV double PriorPackage::get_psi_prior(const double psi)
{
    return uniform_dist_logpdf(psi, psi_min, psi_max);
}

CUDA_DEV double PriorPackage::get_lam_prior(const double lam)
{
    return uniform_dist_logpdf(lam, lam_min, lam_max);
}

CUDA_DEV double PriorPackage::get_sinbeta_prior(const double sinbeta)
{
    return uniform_dist_logpdf(sinbeta, sinbeta_min, sinbeta_max);
}

#define NUM_THREADS_PRIOR_CHECK 32
__global__ void check_prior_vals(double* prior_out, PriorPackage *prior_info, GalacticBinaryParams *gb_params, int num_func)
{
    for (int bin_i = threadIdx.x + blockIdx.x * blockDim.x; bin_i < gb_params->num_bin_all; bin_i += gridDim.x * blockDim.x)
    {
        SingleGalacticBinary gb(
            gb_params->N,
            gb_params->T,
            gb_params->Soms_d,
            gb_params->Sa_a,
            gb_params->Amp,
            gb_params->alpha,
            gb_params->sl1,
            gb_params->kn,
            gb_params->sl2
        );
        
        gb.amp = gb_params->amp[bin_i];
        gb.f0_ms = gb_params->f0_ms[bin_i];
        gb.fdot = gb_params->fdot0[bin_i];
        gb.phi0 = gb_params->phi0[bin_i];
        gb.cosinc = gb_params->cosinc[bin_i];
        gb.psi = gb_params->psi[bin_i];
        gb.lam = gb_params->lam[bin_i];
        gb.sinbeta = gb_params->sinbeta[bin_i];

        double sens_val_prior = -100.0;
        prior_out[bin_i] = prior_info->get_prior_val(gb, num_func, sens_val_prior);
    }
}

void check_prior_vals_wrap(double* prior_out, PriorPackage *prior_info, GalacticBinaryParams *gb_params, int num_func)
{
    GalacticBinaryParams *gb_params_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&gb_params_d, sizeof(GalacticBinaryParams)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(gb_params_d, gb_params, sizeof(GalacticBinaryParams), cudaMemcpyHostToDevice));

    PriorPackage *prior_info_d;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&prior_info_d, sizeof(PriorPackage)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(prior_info_d, prior_info, sizeof(PriorPackage), cudaMemcpyHostToDevice));
    
    int nblocks = std::ceil((gb_params->num_bin_all + NUM_THREADS_PRIOR_CHECK - 1) / NUM_THREADS_PRIOR_CHECK);

    check_prior_vals<<<nblocks, NUM_THREADS_PRIOR_CHECK>>>(prior_out, prior_info_d, gb_params_d, num_func);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaFree(gb_params_d));
    CUDA_CHECK_AND_EXIT(cudaFree(prior_info_d));
}

PeriodicPackage::PeriodicPackage(
    double phi0_period_, double psi_period_, double lam_period_
)
{
    phi0_period = phi0_period_;
    psi_period = psi_period_;
    lam_period = lam_period_;
}

CUDA_HOSTDEV
void wrap_val(double *x, const double x_period)
{
    while(*x < 0.0)
    {
        *x = *x + x_period;
    }
    *x = fmod(*x, x_period);
}

__device__  __host__
int midpoint(int a, int b)
{
    return a + (b-a)/2;
}

__device__ __host__
int eval(double* f_vals, int i, double val, int imin, int imax)
{

    int low = (f_vals[i] <= val);
    int high = (f_vals[i+1] > val);

    if (low && high) {
        return 0;
    } else if (low) {
        return -1;
    } else {
        return 1;
    }
}

// TODO: make this faster by using all threads?
__device__ __host__
int binary_search(double* f_vals, double val, int imin, int imax)
{
    while (imax >= imin) {
        int imid = midpoint(imin, imax);
        int e = eval(f_vals, imid, val, imin, imax);
        if(e == 0) {
            return imid;
        } else if (e < 0) {
            imin = imid;
        } else {         
            imax = imid;
        }
    }

    return -1;
}

CUDA_HOSTDEV
StretchProposalPackage::StretchProposalPackage(
    double* amp_friends_,
    double* f0_friends_, 
    double* fdot0_friends_, 
    double* phi0_friends_, 
    double* cosinc_friends_,
    double* psi_friends_, 
    double* lam_friends_,
    double* sinbeta_friends_,
    int nfriends_,
    int num_friends_init_,
    int num_proposals_,
    double a_,
    int ndim_,
    bool *inds_,
    double *factors_
)
{
    amp_friends = amp_friends_;
    f0_friends = f0_friends_;
    fdot0_friends = fdot0_friends_;
    phi0_friends = phi0_friends_;
    cosinc_friends = cosinc_friends_;
    psi_friends = psi_friends_;
    lam_friends = lam_friends_;
    sinbeta_friends =  sinbeta_friends_;
    nfriends = nfriends_;
    num_friends_init = num_friends_init_;
    num_proposals = num_proposals_;
    a = a_;
    ndim = ndim_;
    inds = inds_;
    factors = factors_;
}

void StretchProposalPackage::dealloc()
{
    return;
}

CUDA_DEV
void StretchProposalPackage::find_friends(SingleGalacticBinary *gb_out, double f_val_in, curandState localState)
{

    int ind_friend;
    
    // lower than minimum
    if (f_val_in <= f0_friends[0])
    {
        ind_friend = 0;
    }
    else if (f_val_in >= f0_friends[num_friends_init - 1])
    {
        ind_friend = num_friends_init - 1;
    }
    else
    {
        ind_friend = binary_search(f0_friends, f_val_in, 0, num_friends_init - 1) - 1;
    }

    int random_ind_change = (((unsigned int)(ceil(curand_uniform_double(&localState) * nfriends))) - 1) - int(nfriends / 2);

    if (ind_friend < int(nfriends / 2))
    {
        ind_friend = int(nfriends / 2) + random_ind_change;
    }
    else if (ind_friend > num_friends_init - 1 - int(nfriends / 2))
    {
        ind_friend = num_friends_init - 1 - int(nfriends / 2) + random_ind_change;
    }
    else
    {
        ind_friend += random_ind_change;
    }
    
    // check boundaries
    if (ind_friend > num_friends_init - 1) ind_friend = num_friends_init - 1;
    if (ind_friend < 0) ind_friend = 0;

    gb_out->amp = amp_friends[ind_friend];
    gb_out->f0_ms = f0_friends[ind_friend];
    gb_out->fdot = fdot0_friends[ind_friend];
    gb_out->phi0 = phi0_friends[ind_friend];
    gb_out->cosinc = cosinc_friends[ind_friend];
    gb_out->psi = psi_friends[ind_friend];
    gb_out->lam = lam_friends[ind_friend];
    gb_out->sinbeta = sinbeta_friends[ind_friend];
}

CUDA_DEV
void StretchProposalPackage::get_proposal(SingleGalacticBinary *gb_prop, double *factors, curandState localState, const SingleGalacticBinary gb_in, const PeriodicPackage periodic_info)
{
    
    double zz = pow(
        (a - 1.0) * curand_uniform_double(&localState) + 1.0
    , 2.0) / a;

    SingleGalacticBinary gb_friend(gb_in.N, gb_in.T, gb_in.Soms_d, gb_in.Sa_a, gb_in.Amp, gb_in.alpha, gb_in.sl1, gb_in.kn, gb_in.sl2);
    
    find_friends(&gb_friend, gb_in.f0_ms, localState);

    direct_change(&(gb_prop->amp), gb_in.amp, gb_friend.amp, zz);
    direct_change(&(gb_prop->f0_ms), gb_in.f0_ms, gb_friend.f0_ms, zz);
    direct_change(&(gb_prop->fdot), gb_in.fdot, gb_friend.fdot, zz);
    wrap_change(&(gb_prop->phi0), gb_in.phi0, gb_friend.phi0, zz, periodic_info.phi0_period);
    direct_change(&(gb_prop->cosinc), gb_in.cosinc, gb_friend.cosinc, zz);
    wrap_change(&(gb_prop->psi), gb_in.psi, gb_friend.psi, zz, periodic_info.psi_period);
    wrap_change(&(gb_prop->lam), gb_in.lam, gb_friend.lam, zz, periodic_info.lam_period);
    direct_change(&(gb_prop->sinbeta), gb_in.sinbeta, gb_friend.sinbeta, zz);
    *factors = (ndim - 1.0) * log(zz);
    // if ((blockIdx.x == 122) && (threadIdx.x == 0)) printf("CHECK: %.12e %.12e\n", zz, *factors);

}

CUDA_HOSTDEV 
void StretchProposalPackage::direct_change(double *x_prop, const double x_curr, const double x_friend, const double fraction)
{
    double diff = x_curr - x_friend;
    *x_prop = x_friend + fraction * diff;
}

CUDA_HOSTDEV 
void StretchProposalPackage::wrap_change(double *x_prop, const double x_curr, const double x_friend, const double fraction, const double period)
{
    double diff = x_curr - x_friend;

    if (abs(diff) > period / 2.)
    {
        if (diff > 0.0)
        {
            diff = x_curr - (x_friend + period);
        }
        else
        {
            diff = x_curr - (x_friend - period);
        }
    }
    double tmp = (x_friend + fraction * diff);
    wrap_val(&tmp, period);
    *x_prop = tmp;
}

CUDA_DEV
SingleGalacticBinary::SingleGalacticBinary(const int N_, const double Tobs_, const double Soms_d_, const double Sa_a_, const double Amp_, const double alpha_, const double sl1_, const double kn_, const double sl2_)
{
    Soms_d = Soms_d_;
    Sa_a = Sa_a_;
    Amp = Amp_;
    alpha = alpha_;
    sl1 = sl1_;
    sl2 = sl2_;
    kn = kn_;
    T = Tobs_;
    N = N_;
}

CUDA_DEV 
void SingleGalacticBinary::transform()
{
    f0 = f0_transform();
    inc = cosinc_transform();
    theta = sinbeta_transform();
    fddot = 0.0;
    // must be after f0
    // amp = amp_transform();
}

CUDA_DEV 
double  SingleGalacticBinary::amp_transform(double amp, double f0, double Sn_f)
{
    double f_star = 1 / (2. * M_PI * lisaL) * Clight;
    if (Sn_f <= 0.0){
        Sn_f = lisasens(f0, Soms_d * Soms_d, Sa_a * Sa_a, Amp, alpha, sl1, kn, sl2);
    }
    double factor = 1./2. * sqrt((T * pow(sin(f0 / f_star), 2.)) / Sn_f);
    return amp * factor;
}

CUDA_DEV 
double SingleGalacticBinary::f0_transform()
{
   return f0_ms / 1e3;
}

CUDA_DEV 
double SingleGalacticBinary::cosinc_transform()
{
    return acos(cosinc);
}  

CUDA_DEV 
double SingleGalacticBinary::sinbeta_transform()
{
    double beta_tmp = asin(sinbeta);
    double theta_tmp = M_PI / 2. - beta_tmp;
    return theta_tmp;
}

// CUDA_HOSTDEV
// SingleBand::SingleBand(){return;}

CUDA_HOSTDEV
void SingleBand::setup(
    int loc_index_,
    int data_index_,
    int noise_index_,
    int band_start_bin_ind_,
    int band_num_bins_,
    int band_start_data_ind_,
    int band_data_lengths_,
    int band_interest_start_data_ind_,
    int band_interest_data_lengths_,
    int max_data_store_size_,
    double fmin_allow_,
    double fmax_allow_,
    int update_data_index_,
    double inv_temp_,
    int band_ind_,
    int walker_ind_,
    int temp_ind_,
    GalacticBinaryParams *gb_params_all
)
{
    loc_index = loc_index_;
    data_index = data_index_;
    noise_index = noise_index_;
    band_start_bin_ind = band_start_bin_ind_;
    band_num_bins = band_num_bins_;
    band_start_data_ind = band_start_data_ind_;
    band_data_lengths = band_data_lengths_;
    band_interest_start_data_ind = band_interest_start_data_ind_;
    band_interest_data_lengths = band_interest_data_lengths_;
    max_data_store_size = max_data_store_size_;
    fmin_allow = fmin_allow_;
    fmax_allow = fmax_allow_;
    update_data_index = update_data_index_;
    inv_temp = inv_temp_;
    band_ind = band_ind_;
    walker_ind = walker_ind_;
    temp_ind = temp_ind_;
    gb_params = GalacticBinaryParams(
        &(gb_params_all->amp[band_start_bin_ind]),
        &(gb_params_all->f0_ms[band_start_bin_ind]),
        &(gb_params_all->fdot0[band_start_bin_ind]),
        &(gb_params_all->phi0[band_start_bin_ind]),
        &(gb_params_all->cosinc[band_start_bin_ind]),
        &(gb_params_all->psi[band_start_bin_ind]),
        &(gb_params_all->lam[band_start_bin_ind]),
        &(gb_params_all->sinbeta[band_start_bin_ind]),
        &(gb_params_all->amp_orig[band_start_bin_ind]),
        &(gb_params_all->f0_ms_orig[band_start_bin_ind]),
        &(gb_params_all->fdot0_orig[band_start_bin_ind]),
        &(gb_params_all->phi0_orig[band_start_bin_ind]),
        &(gb_params_all->cosinc_orig[band_start_bin_ind]),
        &(gb_params_all->psi_orig[band_start_bin_ind]),
        &(gb_params_all->lam_orig[band_start_bin_ind]),
        &(gb_params_all->sinbeta_orig[band_start_bin_ind]),
        gb_params_all->T, 
        gb_params_all->dt,
        gb_params_all->N,
        gb_params_all->num_bin_all,
        gb_params_all->start_freq_ind,
        gb_params_all->Soms_d,
        gb_params_all->Sa_a, 
        gb_params_all->Amp,
        gb_params_all->alpha,
        gb_params_all->sl1,
        gb_params_all->kn,
        gb_params_all->sl2
    );
}



#define NUM_THREADS_LIKE 256
__global__ void get_psd_val(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int noise_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    double f, Sn_A, Sn_E;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int f_i = blockIdx.x * blockDim.x + threadIdx.x; f_i < num_f; f_i += gridDim.x * blockDim.x)
    {
        noise_index = noise_index_all[f_i];

        A_Soms_d_in = A_Soms_d_in_all[noise_index];
        A_Sa_a_in = A_Sa_a_in_all[noise_index];
        E_Soms_d_in = E_Soms_d_in_all[noise_index];
        E_Sa_a_in = E_Sa_a_in_all[noise_index];
        Amp = Amp_all[noise_index];
        alpha = alpha_all[noise_index];
        sl1 = sl1_all[noise_index];
        kn = kn_all[noise_index];
        sl2 = sl2_all[noise_index];
        f = f_arr[f_i];
        
        A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
        A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
        E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
        E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
        Sn_A = noisepsd_AE(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        Sn_E = noisepsd_AE(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

        // if (Sn_A != Sn_A)
        // {
        //     printf("BADDDDD: %d %e %e %e %e %e %e %e %e\n", f_i, f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        // }

        Sn_A_out[f_i] = Sn_A;
        Sn_E_out[f_i] = Sn_E;
    }
}

void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{

    int num_blocks = std::ceil((num_f + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    get_psd_val<<<num_blocks, NUM_THREADS_LIKE>>>(Sn_A_out, Sn_E_out, f_arr, noise_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, num_f);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}




#define NUM_THREADS_LIKE 256
__global__ void get_lisasens_val(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int noise_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    double f, Sn_A, Sn_E;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int f_i = blockIdx.x * blockDim.x + threadIdx.x; f_i < num_f; f_i += gridDim.x * blockDim.x)
    {
        noise_index = noise_index_all[f_i];

        A_Soms_d_in = A_Soms_d_in_all[noise_index];
        A_Sa_a_in = A_Sa_a_in_all[noise_index];
        E_Soms_d_in = E_Soms_d_in_all[noise_index];
        E_Sa_a_in = E_Sa_a_in_all[noise_index];
        Amp = Amp_all[noise_index];
        alpha = alpha_all[noise_index];
        sl1 = sl1_all[noise_index];
        kn = kn_all[noise_index];
        sl2 = sl2_all[noise_index];
        f = f_arr[f_i];
        
        A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
        A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
        E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
        E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
        Sn_A = lisasens(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        Sn_E = lisasens(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

        // if (Sn_A != Sn_A)
        // {
        //     printf("BADDDDD: %d %e %e %e %e %e %e %e %e\n", f_i, f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        // }

        Sn_A_out[f_i] = Sn_A;
        Sn_E_out[f_i] = Sn_E;
    }
}

void get_lisasens_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{

    int num_blocks = std::ceil((num_f + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    get_lisasens_val<<<num_blocks, NUM_THREADS_LIKE>>>(Sn_A_out, Sn_E_out, f_arr, noise_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, num_f);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}


