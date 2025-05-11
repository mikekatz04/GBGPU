#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

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

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void make_move(
    cmplx *L_contribution,
    cmplx *p_contribution,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *params_curr,
    double *params_prop,
    double *prior_all_curr,
    double *prior_all_prop,
    double *factors_all,
    double *random_val_all,
    int *band_start_bin_ind,
    int *band_num_bins,
    int *band_start_data_ind,
    int *band_data_lengths,
    double *band_inv_temperatures_all,
    bool *accepted_out,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int num_bands,
    int max_band_data_length,
    bool is_rj,
    double snr_lim)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    unsigned int start_ind_add = 0;
    unsigned int start_ind_remove = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

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

    int this_band_start_index, this_band_length, j, k;
    int this_band_start_bin_ind, this_band_num_bin;
    int this_band_data_index, this_band_noise_index;
    double this_binary_inv_temp;

    double amp_prop, f0_prop, fdot0_prop, fddot0_prop, phi0_prop, iota_prop, psi_prop, lam_prop, theta_prop;
    double amp_curr, f0_curr, fdot0_curr, fddot0_curr, phi0_curr, iota_curr, psi_curr, lam_curr, theta_curr;

    int current_binary_start_index, base_index;
    double ll_diff, lp_diff;
    double prior_curr, prior_prop, factors, lnpdiff, random_val;
    bool accept;

    cmplx d_h_remove_temp = 0.0;
    cmplx d_h_add_temp = 0.0;
    cmplx remove_remove_temp = 0.0;
    cmplx add_add_temp = 0.0;
    cmplx add_remove_temp = 0.0;

    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_add_lower;
    int total_i_vals;

    cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
    int real_ind, real_ind_add, real_ind_remove;
    cmplx d_A, d_E;
    double n_A, n_E;
    double opt_snr, det_snr;

    for (int band_i = blockIdx.x; band_i < num_bands; band_i += gridDim.x)
    {
        this_band_start_index = band_start_data_ind[band_i]; // overall index to which binary
        this_band_length = band_data_lengths[band_i];
        this_band_start_bin_ind = band_start_bin_ind[band_i];
        this_band_num_bin = band_num_bins[band_i];
        this_band_data_index = data_index[band_i];
        this_band_noise_index = noise_index[band_i];

        for (int bin_i = 0; bin_i < this_band_num_bin; bin_i += 1)
        {
            current_binary_start_index = this_band_start_bin_ind + bin_i;
            base_index = current_binary_start_index * 9;

            this_binary_inv_temp = band_inv_temperatures_all[current_binary_start_index];

            prior_curr = prior_all_curr[current_binary_start_index];
            prior_prop = prior_all_prop[current_binary_start_index];

            lp_diff = prior_prop - prior_curr;
            factors = factors_all[current_binary_start_index];
            random_val = random_val_all[current_binary_start_index];

            // get the parameters to add and remove
            amp_curr = params_curr[base_index + 0];
            f0_curr = params_curr[base_index + 1];
            fdot0_curr = params_curr[base_index + 2];
            fddot0_curr = params_curr[base_index + 3];
            phi0_curr = params_curr[base_index + 4];
            iota_curr = params_curr[base_index + 5];
            psi_curr = params_curr[base_index + 6];
            lam_curr = params_curr[base_index + 7];
            theta_curr = params_curr[base_index + 8];

            amp_prop = params_prop[base_index + 0];
            f0_prop = params_prop[base_index + 1];
            fdot0_prop = params_prop[base_index + 2];
            fddot0_prop = params_prop[base_index + 3];
            phi0_prop = params_prop[base_index + 4];
            iota_prop = params_prop[base_index + 5];
            psi_prop = params_prop[base_index + 6];
            lam_prop = params_prop[base_index + 7];
            theta_prop = params_prop[base_index + 8];

            __syncthreads();
            build_single_waveform<FFT>(
                wave_remove,
                &start_ind_remove,
                amp_curr,
                f0_curr,
                fdot0_curr,
                fddot0_curr,
                phi0_curr,
                iota_curr,
                psi_curr,
                lam_curr,
                theta_curr,
                T,
                dt,
                N,
                bin_i);
            __syncthreads();
            build_single_waveform<FFT>(
                wave_add,
                &start_ind_add,
                amp_prop,
                f0_prop,
                fdot0_prop,
                fddot0_prop,
                phi0_prop,
                iota_prop,
                psi_prop,
                lam_prop,
                theta_prop,
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
            d_h_remove_temp = 0.0;
            d_h_add_temp = 0.0;
            remove_remove_temp = 0.0;
            add_add_temp = 0.0;
            add_remove_temp = 0.0;

            if (total_i_vals < 2 * N)
            {
                for (int i = threadIdx.x;
                     i < total_i_vals;
                     i += blockDim.x)
                {

                    j = lower_start_ind + i;

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

                    // if ((bin_i == 0)){
                    // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, this_band_noise_index * data_length + j, this_band_data_index * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
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

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

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

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

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

            ll_diff = -1. / 2. * (-2. * d_h_add_arr[0] + 2. * d_h_remove_arr[0] - 2. * add_remove_arr[0] + add_add_arr[0] + remove_remove_arr[0]).real();
            __syncthreads();

            // determine detailed balance with tempering on the Likelihood term
            lnpdiff = factors + (this_binary_inv_temp * ll_diff) + lp_diff;

            // accept or reject
            accept = lnpdiff > random_val;
            // accept = false;
            if ((is_rj) && (amp_prop / amp_curr > 1e10))
            {
                det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
                opt_snr = sqrt(add_add_arr[0].real());
                // put an snr limit on rj
                if ((opt_snr < snr_lim) || (det_snr < snr_lim) || (abs(1.0 - det_snr / opt_snr) > 0.5))
                {
                    // if ((this_band_data_index == 0) && (threadIdx.x == 0)) printf("NIXED %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                    accept = false;
                }
                else
                {
                    //if ((this_binary_inv_temp < 0.005) & (threadIdx.x == 0)) printf("KEPT  %e %e %e %e %e\n", this_binary_inv_temp, snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                }
            }

            // if ((is_rj) && (amp_prop / amp_curr > 1e10) && (accept))
            // {
            //     det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
            //     opt_snr = sqrt(add_add_arr[0].real());
            //     if ((threadIdx.x == 0))
            //         printf("SNR info  %e %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr), this_binary_inv_temp);
            // }

            // if ((blockIdx.x == 0) && (threadIdx.x == 0))printf("%d %d %.12e %.12e %.12e %.12e %e %e %e %e %e %e\n", bin_i, accept, f0_prop, f0_curr, wave_remove[0].real(), wave_add[0].real(), ll_diff, this_binary_inv_temp, lp_diff, factors, lnpdiff, random_val);

            // readout if it was accepted
            accepted_out[current_binary_start_index] = accept;

            if (accept)
            {
                if (tid == 0)
                {
                    L_contribution[band_i] += ll_diff;
                    p_contribution[band_i] += lp_diff;
                }
                __syncthreads();
                // change current Likelihood

                for (int i = threadIdx.x;
                     i < N;
                     i += blockDim.x)
                {

                    j = start_ind_remove + i - start_freq_ind;

                    h_A = A_remove[i];
                    h_E = E_remove[i];

                    // if (i == 0) printf("start: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());

                    // data_A[this_band_data_index * data_length + j] += h_A;
                    // data_E[this_band_data_index * data_length + j] += h_E;

                    atomicAddComplex(&data_A[this_band_data_index * data_length + j], h_A);
                    atomicAddComplex(&data_E[this_band_data_index * data_length + j], h_E);
                    // if (i == 0) printf("remove: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                }
                __syncthreads();

                for (int i = threadIdx.x;
                     i < N;
                     i += blockDim.x)
                {

                    j = start_ind_add + i - start_freq_ind;

                    h_A = A_add[i];
                    h_E = E_add[i];

                    // data_A[this_band_data_index * data_length + j] -= h_A;
                    // data_E[this_band_data_index * data_length + j] -= h_E;

                    atomicAddComplex(&data_A[this_band_data_index * data_length + j], -h_A);
                    atomicAddComplex(&data_E[this_band_data_index * data_length + j], -h_E);

                    // if (i == 0) printf("add: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                }
                __syncthreads();
                // do not need to adjust data as this one is already in there
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void make_move_wrap(InputInfo inputs)
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

    // first is waveforms, second is ll, third is A, E data, fourth is A psd and E psd
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 3 * N * sizeof(cmplx) + 5 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        make_move<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    make_move<FFT><<<inputs.num_bands, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.L_contribution,
        inputs.p_contribution,
        inputs.data_A,
        inputs.data_E,
        inputs.noise_A,
        inputs.noise_E,
        inputs.data_index,
        inputs.noise_index,
        inputs.params_curr,
        inputs.params_prop,
        inputs.prior_all_curr,
        inputs.prior_all_prop,
        inputs.factors_all,
        inputs.random_val_all,
        inputs.band_start_bin_ind,
        inputs.band_num_bins,
        inputs.band_start_data_ind,
        inputs.band_data_lengths,
        inputs.band_inv_temperatures_all,
        inputs.accepted_out,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length,
        inputs.num_bands,
        inputs.max_data_store_size,
        inputs.is_rj,
        inputs.snr_lim);

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
struct make_move_wrap_functor
{
    void operator()(InputInfo inputs) { return make_move_wrap<Arch, N>(inputs); }
};

void SharedMemoryMakeMove(
    cmplx *L_contribution,
    cmplx *p_contribution,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *params_curr,
    double *params_prop,
    double *prior_all_curr,
    double *prior_all_prop,
    double *factors_all,
    double *random_val_all,
    int *band_start_bin_ind,
    int *band_num_bins,
    int *band_start_data_ind,
    int *band_data_lengths,
    double *band_inv_temperatures_all,
    bool *accepted_out,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int num_bands,
    int max_data_store_size,
    int device,
    bool do_synchronize,
    bool is_rj,
    double snr_lim)
{

    InputInfo inputs;

    inputs.L_contribution = L_contribution;
    inputs.p_contribution = p_contribution;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
    inputs.data_index = data_index;
    inputs.noise_index = noise_index;
    inputs.params_curr = params_curr;
    inputs.params_prop = params_prop;
    inputs.prior_all_curr = prior_all_curr;
    inputs.prior_all_prop = prior_all_prop;
    inputs.factors_all = factors_all;
    inputs.random_val_all = random_val_all;
    inputs.band_start_bin_ind = band_start_bin_ind;
    inputs.band_num_bins = band_num_bins;
    inputs.band_start_data_ind = band_start_data_ind;
    inputs.band_data_lengths = band_data_lengths;
    inputs.band_inv_temperatures_all = band_inv_temperatures_all;
    inputs.accepted_out = accepted_out;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.num_bands = num_bands;
    inputs.max_data_store_size = max_data_store_size;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.is_rj = is_rj;
    inputs.snr_lim = snr_lim;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<make_move_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<make_move_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<make_move_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<make_move_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<make_move_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<make_move_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<make_move_wrap_functor, 2048>(inputs);
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
__launch_bounds__(FFT::max_threads_per_block) __global__ void make_move_together(
    cmplx *L_contribution,
    cmplx *p_contribution,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *params_curr,
    double *params_prop,
    double *prior_all_curr,
    double *prior_all_prop,
    double *factors_all,
    double *random_val_all,
    int *band_start_bin_ind,
    int *band_num_bins,
    int *band_start_data_ind,
    int *band_data_lengths,
    double *band_inv_temperatures_all,
    bool *accepted_out,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int num_bands,
    int max_band_data_length,
    bool is_rj,
    double snr_lim)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    unsigned int start_ind_add = 0;
    unsigned int start_ind_remove = 0;

    extern __shared__ unsigned char shared_mem[];

    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    cmplx *tmp = (cmplx *)shared_mem;
    cmplx *A_diff = &tmp[0];
    cmplx *E_diff = &tmp[max_band_data_length];

    cmplx *d_h_r_a = &tmp[2 * max_band_data_length];
    cmplx *r_a_r_a = &d_h_r_a[FFT::block_dim.x];
    
    int this_band_start_index, this_band_length, j, k;
    int this_band_start_bin_ind, this_band_num_bin;
    int this_band_data_index, this_band_noise_index;
    double this_binary_inv_temp;

    double amp_prop, f0_prop, fdot0_prop, fddot0_prop, phi0_prop, iota_prop, psi_prop, lam_prop, theta_prop;
    double amp_curr, f0_curr, fdot0_curr, fddot0_curr, phi0_curr, iota_curr, psi_curr, lam_curr, theta_curr;

    int current_binary_start_index, base_index;
    double ll_diff, lp_diff;
    double prior_curr, prior_prop, factors, lnpdiff, random_val;
    bool accept;

    cmplx d_h_remove_temp = 0.0;
    cmplx d_h_add_temp = 0.0;
    cmplx remove_remove_temp = 0.0;
    cmplx add_add_temp = 0.0;
    cmplx add_remove_temp = 0.0;

    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_add_lower;
    int total_i_vals;

    cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
    int real_ind, real_ind_add, real_ind_remove;
    cmplx d_A, d_E;
    double n_A, n_E;
    double opt_snr, det_snr;
    double tmp_prior_curr, tmp_prior_prop;

    for (int band_i = blockIdx.x; band_i < num_bands; band_i += gridDim.x)
    {
        tmp_prior_curr = 0.0;
        tmp_prior_prop = 
        this_band_start_index = band_start_data_ind[band_i]; // overall index to which binary
        this_band_length = band_data_lengths[band_i];
        this_band_start_bin_ind = band_start_bin_ind[band_i];
        this_band_num_bin = band_num_bins[band_i];
        this_band_data_index = data_index[band_i];
        this_band_noise_index = noise_index[band_i];

        for (int bin_i = 0; bin_i < this_band_num_bin; bin_i += 1)
        {
            current_binary_start_index = this_band_start_bin_ind + bin_i;
            base_index = current_binary_start_index * 9;

            this_binary_inv_temp = band_inv_temperatures_all[current_binary_start_index];

            prior_curr = prior_all_curr[current_binary_start_index];
            prior_prop = prior_all_prop[current_binary_start_index];

            lp_diff = prior_prop - prior_curr;
            factors = factors_all[current_binary_start_index];
            random_val = random_val_all[current_binary_start_index];

            // get the parameters to add and remove
            amp_curr = params_curr[base_index + 0];
            f0_curr = params_curr[base_index + 1];
            fdot0_curr = params_curr[base_index + 2];
            fddot0_curr = params_curr[base_index + 3];
            phi0_curr = params_curr[base_index + 4];
            iota_curr = params_curr[base_index + 5];
            psi_curr = params_curr[base_index + 6];
            lam_curr = params_curr[base_index + 7];
            theta_curr = params_curr[base_index + 8];

            amp_prop = params_prop[base_index + 0];
            f0_prop = params_prop[base_index + 1];
            fdot0_prop = params_prop[base_index + 2];
            fddot0_prop = params_prop[base_index + 3];
            phi0_prop = params_prop[base_index + 4];
            iota_prop = params_prop[base_index + 5];
            psi_prop = params_prop[base_index + 6];
            lam_prop = params_prop[base_index + 7];
            theta_prop = params_prop[base_index + 8];

            __syncthreads();
            build_single_waveform<FFT>(
                wave_remove,
                &start_ind_remove,
                amp_curr,
                f0_curr,
                fdot0_curr,
                fddot0_curr,
                phi0_curr,
                iota_curr,
                psi_curr,
                lam_curr,
                theta_curr,
                T,
                dt,
                N,
                bin_i);
            __syncthreads();
            build_single_waveform<FFT>(
                wave_add,
                &start_ind_add,
                amp_prop,
                f0_prop,
                fdot0_prop,
                fddot0_prop,
                phi0_prop,
                iota_prop,
                psi_prop,
                lam_prop,
                theta_prop,
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
            d_h_remove_temp = 0.0;
            d_h_add_temp = 0.0;
            remove_remove_temp = 0.0;
            add_add_temp = 0.0;
            add_remove_temp = 0.0;

            if (total_i_vals < 2 * N)
            {
                for (int i = threadIdx.x;
                     i < total_i_vals;
                     i += blockDim.x)
                {

                    j = lower_start_ind + i;

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

                    // if ((bin_i == 0)){
                    // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, this_band_noise_index * data_length + j, this_band_data_index * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
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

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

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

                    n_A = noise_A[this_band_noise_index * data_length + j];
                    n_E = noise_E[this_band_noise_index * data_length + j];

                    d_A = data_A[this_band_data_index * data_length + j];
                    d_E = data_E[this_band_data_index * data_length + j];

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

            ll_diff = -1. / 2. * (-2. * d_h_add_arr[0] + 2. * d_h_remove_arr[0] - 2. * add_remove_arr[0] + add_add_arr[0] + remove_remove_arr[0]).real();
            __syncthreads();

            // determine detailed balance with tempering on the Likelihood term
            lnpdiff = factors + (this_binary_inv_temp * ll_diff) + lp_diff;

            // accept or reject
            accept = lnpdiff > random_val;
            // accept = false;
            if ((is_rj) && (amp_prop / amp_curr > 1e10))
            {
                det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
                opt_snr = sqrt(add_add_arr[0].real());
                // put an snr limit on rj
                if ((opt_snr < snr_lim) || (det_snr < snr_lim) || (abs(1.0 - det_snr / opt_snr) > 0.5))
                {
                    // if ((this_band_data_index == 0) && (threadIdx.x == 0)) printf("NIXED %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                    accept = false;
                }
                else
                {
                    //if ((this_binary_inv_temp < 0.005) & (threadIdx.x == 0)) printf("KEPT  %e %e %e %e %e\n", this_binary_inv_temp, snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr));
                }
            }

            // if ((is_rj) && (amp_prop / amp_curr > 1e10) && (accept))
            // {
            //     det_snr = ((d_h_add_arr[0].real() + add_remove_arr[0].real()) / sqrt(add_add_arr[0].real()));
            //     opt_snr = sqrt(add_add_arr[0].real());
            //     if ((threadIdx.x == 0))
            //         printf("SNR info  %e %e %e %e %e\n", snr_lim, opt_snr, det_snr, abs(1.0 - det_snr / opt_snr), this_binary_inv_temp);
            // }

            // if ((blockIdx.x == 0) && (threadIdx.x == 0))printf("%d %d %.12e %.12e %.12e %.12e %e %e %e %e %e %e\n", bin_i, accept, f0_prop, f0_curr, wave_remove[0].real(), wave_add[0].real(), ll_diff, this_binary_inv_temp, lp_diff, factors, lnpdiff, random_val);

            // readout if it was accepted
            accepted_out[current_binary_start_index] = accept;

            if (accept)
            {
                if (tid == 0)
                {
                    L_contribution[band_i] += ll_diff;
                    p_contribution[band_i] += lp_diff;
                }
                __syncthreads();
                // change current Likelihood

                for (int i = threadIdx.x;
                     i < N;
                     i += blockDim.x)
                {

                    j = start_ind_remove + i - start_freq_ind;

                    h_A = A_remove[i];
                    h_E = E_remove[i];

                    // if (i == 0) printf("start: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());

                    // data_A[this_band_data_index * data_length + j] += h_A;
                    // data_E[this_band_data_index * data_length + j] += h_E;

                    atomicAddComplex(&data_A[this_band_data_index * data_length + j], h_A);
                    atomicAddComplex(&data_E[this_band_data_index * data_length + j], h_E);
                    // if (i == 0) printf("remove: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                }
                __syncthreads();

                for (int i = threadIdx.x;
                     i < N;
                     i += blockDim.x)
                {

                    j = start_ind_add + i - start_freq_ind;

                    h_A = A_add[i];
                    h_E = E_add[i];

                    // data_A[this_band_data_index * data_length + j] -= h_A;
                    // data_E[this_band_data_index * data_length + j] -= h_E;

                    atomicAddComplex(&data_A[this_band_data_index * data_length + j], -h_A);
                    atomicAddComplex(&data_E[this_band_data_index * data_length + j], -h_E);

                    // if (i == 0) printf("add: %d %.12e %.12e %.12e %.12e : %.12e %.12e %.12e %.12e\n\n", bin_i, data_A[this_band_data_index * data_length + j].real(), data_A[this_band_data_index * data_length + j].imag(), data_E[this_band_data_index * data_length + j].real(), data_E[this_band_data_index * data_length + j].imag(), h_A.real(), h_A.imag(), h_E.real(), h_E.imag());
                }
                __syncthreads();
                // do not need to adjust data as this one is already in there
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void make_move_wrap(InputInfo inputs)
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

    // 
    auto shared_memory_size_mine = 2 * inputs.max_data_store_size * sizeof(cmplx) + 2 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        make_move<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    make_move<FFT><<<inputs.num_bands, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.L_contribution,
        inputs.p_contribution,
        inputs.data_A,
        inputs.data_E,
        inputs.noise_A,
        inputs.noise_E,
        inputs.data_index,
        inputs.noise_index,
        inputs.params_curr,
        inputs.params_prop,
        inputs.prior_all_curr,
        inputs.prior_all_prop,
        inputs.factors_all,
        inputs.random_val_all,
        inputs.band_start_bin_ind,
        inputs.band_num_bins,
        inputs.band_start_data_ind,
        inputs.band_data_lengths,
        inputs.band_inv_temperatures_all,
        inputs.accepted_out,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_ind,
        inputs.data_length,
        inputs.num_bands,
        inputs.max_data_store_size,
        inputs.is_rj,
        inputs.snr_lim);

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
struct make_move_wrap_functor
{
    void operator()(InputInfo inputs) { return make_move_wrap<Arch, N>(inputs); }
};

void SharedMemoryMakeMove(
    cmplx *L_contribution,
    cmplx *p_contribution,
    cmplx *data_A,
    cmplx *data_E,
    double *noise_A,
    double *noise_E,
    int *data_index,
    int *noise_index,
    double *params_curr,
    double *params_prop,
    double *prior_all_curr,
    double *prior_all_prop,
    double *factors_all,
    double *random_val_all,
    int *band_start_bin_ind,
    int *band_num_bins,
    int *band_start_data_ind,
    int *band_data_lengths,
    double *band_inv_temperatures_all,
    bool *accepted_out,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int num_bands,
    int max_data_store_size,
    int device,
    bool do_synchronize,
    bool is_rj,
    double snr_lim)
{

    InputInfo inputs;

    inputs.L_contribution = L_contribution;
    inputs.p_contribution = p_contribution;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
    inputs.data_index = data_index;
    inputs.noise_index = noise_index;
    inputs.params_curr = params_curr;
    inputs.params_prop = params_prop;
    inputs.prior_all_curr = prior_all_curr;
    inputs.prior_all_prop = prior_all_prop;
    inputs.factors_all = factors_all;
    inputs.random_val_all = random_val_all;
    inputs.band_start_bin_ind = band_start_bin_ind;
    inputs.band_num_bins = band_num_bins;
    inputs.band_start_data_ind = band_start_data_ind;
    inputs.band_data_lengths = band_data_lengths;
    inputs.band_inv_temperatures_all = band_inv_temperatures_all;
    inputs.accepted_out = accepted_out;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_ind = start_freq_ind;
    inputs.data_length = data_length;
    inputs.num_bands = num_bands;
    inputs.max_data_store_size = max_data_store_size;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.is_rj = is_rj;
    inputs.snr_lim = snr_lim;

    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<make_move_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<make_move_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<make_move_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<make_move_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<make_move_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<make_move_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<make_move_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
}

const double lisaL = 2.5e9;           // LISA's arm meters
const double lisaLT = lisaL / Clight; // LISA's armn in sec

__device__ void lisanoises(double *Spm, double *Sop, double f, double Soms_d_in, double Sa_a_in)
{
    double frq = f;
    // Acceleration noise
    // In acceleration
    double Sa_a = Sa_a_in * (1.0 + pow((0.4e-3 / frq), 2)) * (1.0 + pow((frq / 8e-3), 4));
    // In displacement
    double Sa_d = Sa_a * pow((2.0 * M_PI * frq), (-4.0));
    // In relative frequency unit
    double Sa_nu = Sa_d * pow((2.0 * M_PI * frq / Clight), 2);
    *Spm = Sa_nu;

    // Optical Metrology System
    // In displacement
    double Soms_d = Soms_d_in * (1.0 + pow((2.0e-3 / f), 4));
    // In relative frequency unit
    double Soms_nu = Soms_d * pow((2.0 * M_PI * frq / Clight), 2);
    *Sop = Soms_nu;

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

__device__ double noisepsd_AE(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double Spm, Sop;
    lisanoises(&Spm, &Sop, f, Soms_d_in, Sa_a_in);

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