#include <stdio.h>
#include <iostream>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <curand_kernel.h>
#include "block_io.hpp"
#include "common.hpp"
#endif

#include "SharedMemoryGBGPU.hpp"
#include "LISA.h"
#include "global.h"
#include "math.h"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

// Recursive FFT function
vector<cmplx> fft(const vector<cmplx>& x) {
    int N = x.size();

    // Base case: if the input size is 1, return the input as is
    if (N <= 1) {
        return x;
    }

    // Divide the input into even and odd indexed elements
    vector<cmplx> even(N / 2);
    vector<cmplx> odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    // Recursively compute the FFT of even and odd indexed elements
    vector<cmplx> q = fft(even);
    vector<cmplx> r = fft(odd);

    // Combine the results
    vector<cmplx> y(N);
    for (int i = 0; i < N / 2; ++i) {
        double angle = -2 * M_PI * i / N;
        cmplx w(cos(angle), sin(angle));
        y[i] = q[i] + w * r[i];
        y[i + N / 2] = q[i] - w * r[i];
    }
    return y;
}

CUDA_DEVICE
void spacecraft(double P1[3], double P2[3], double P3[3], double t)
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

CUDA_DEVICE
__inline__ void get_eplus(double eplus[], double u[], double v[])
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

CUDA_DEVICE
__inline__ void get_ecross(double ecross[], double u[], double v[])
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

CUDA_DEVICE
__inline__ void AET_from_XYZ_swap(cmplx *X_in, cmplx *Y_in, cmplx *Z_in)
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

#ifdef __CUDACC__
template <class FFT>
 
#endif
CUDA_DEVICE
void build_single_waveform(
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
    int bin_i,
    int tdi_channel_setup)
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

#ifdef __CUDACC__
    int start1 = threadIdx.x;
    int incr1 = blockDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif
    // construct slow part
    for (int i = start1; i < N; i += incr1)
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

    CUDA_SYNCTHREADS;

    #ifdef __CUDACC__
    FFT().execute(reinterpret_cast<void *>(X));
    FFT().execute(reinterpret_cast<void *>(Y));
    FFT().execute(reinterpret_cast<void *>(Z));
    #else
    std::vector<cmplx> vecX(X, X + N);
    std::vector<cmplx> vecY(Y, Y + N);
    std::vector<cmplx> vecZ(Z, Z + N);
    std::vector<cmplx> vecX_fft = fft(vecX);
    std::vector<cmplx> vecY_fft = fft(vecY);
    std::vector<cmplx> vecZ_fft = fft(vecZ);

    for (int i = 0; i < N; i += 1)
    {
        X[i] = vecX_fft[i];
        Y[i] = vecY_fft[i];
        Z[i] = vecZ_fft[i];
    }
    #endif

    CUDA_SYNCTHREADS;

    for (int i = start1; i < N; i += incr1)
    {
        X[i] *= amp;
        Y[i] *= amp;
        Z[i] *= amp;
    }

    CUDA_SYNCTHREADS;

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
    for (int i = start1; i < N_over_2; i += incr1)
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
    if ((tdi_channel_setup == TDI_CHANNEL_SETUP_AET) || (tdi_channel_setup == TDI_CHANNEL_SETUP_AE))
    {
        for (int i = start1; i < N; i += incr1)
        {
            AET_from_XYZ_swap(&X[i], &Y[i], &Z[i]);
        }
    }

    CUDA_SYNCTHREADS;

    double fmin = (q - ((double)N) / 2.) / T;
    *start_ind = (int)rint(fmin * T);

    return;
}

#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) 
__global__
#endif
void get_waveform(
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
    int num_bin_all,
    int tdi_channel_setup)
{
    using complex_type = cmplx;

    int nchannels = 3;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE)
        nchannels = 2;

    unsigned int start_ind = 0;

#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    cmplx *wave = (cmplx *)shared_mem;
#else
    cmplx wave_tmp[3 * N];
    cmplx *wave = &wave_tmp[0];
#endif

#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif

    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {

#ifdef __CUDACC__
        build_single_waveform<FFT>(
#else
        build_single_waveform(
#endif
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
            bin_i,
            tdi_channel_setup);

#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int incr2 = blockDim.x;
#else
        int start2 = 0;
        int incr2 = 1;
#endif
        for (int i = start2; i < N; i += incr2)
        {
            for (int j = 0; j < nchannels; j += 1)
            {
                tdi_out[(bin_i * nchannels + j) * N + i] = wave[j * N + i];
            }
        }

        CUDA_SYNCTHREADS;
    }
    //
}

#ifdef __CUDACC__
// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void get_waveform_wrap(InputInfo inputs)
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
        inputs.num_bin_all,
        inputs.tdi_channel_setup);
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
struct get_waveform_wrap_functor
{
    void operator()(InputInfo inputs) { return get_waveform_wrap<Arch, N>(inputs); }
};
#endif

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
    int num_bin_all, 
    int tdi_channel_setup)
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
    inputs.tdi_channel_setup = tdi_channel_setup;

#ifdef __CUDACC__
    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<get_waveform_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<get_waveform_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<get_waveform_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<get_waveform_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<get_waveform_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<get_waveform_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<get_waveform_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }

    // const unsigned int arch = example::get_cuda_device_arch();
    // simple_block_fft<800>(x);
#else
    get_waveform(
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
        inputs.num_bin_all,
        inputs.tdi_channel_setup);
#endif
}

//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////

CUDA_DEVICE
void add_inner_product_contribution(
    cmplx *contrib_h1_h2, // d_h if <data length | template length>, else ignored
    cmplx *contrib_h1_h1, // h_h if <data length | template length>, else a_b
    cmplx *contrib_h2_h2,
    cmplx *h1, // or template 1 if added as template
    cmplx *h2, // or template 2 if data is a template
    int i_1, // 
    int i_2, //
    int array_type_1,
    int array_type_2,
    double *noise,
    int noise_ind,
    int noise_i,
    int data_ind, // ignored if array types are both templates
    int tdi_channel_setup,
    int data_length,
    int N,
    int num_data,
    int num_noise
)
{
    int nchannels = 3;
    double multi_factor = 1.0;
    cmplx h1_i, h2_i;
    double n;
    int noise_ind_now, data_ind_now, template_ind_now;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE) nchannels = 2;

    if ((tdi_channel_setup == TDI_CHANNEL_SETUP_AE) || (tdi_channel_setup == TDI_CHANNEL_SETUP_AET))
    {
        for (int chan = 0; chan < nchannels; chan += 1)
        {
            noise_ind_now = (noise_ind * nchannels + chan) * data_length + noise_i;
            
            if ((noise_ind_now >= nchannels * data_length * num_noise) | (noise_i > data_length))
            {
                printf("Above full noise range.%d, %d, %d, %d\n", noise_ind_now, nchannels * data_length * num_noise, noise_i, data_length);
                continue;
            }
            n = noise[noise_ind_now];

            if (array_type_1 == ARRAY_TYPE_DATA)
            {
                data_ind_now = (data_ind * nchannels + chan) * data_length + i_1;
                if ((data_ind_now >= nchannels * data_length * num_data) | (i_1 > data_length))
                {
                    printf("Above full data range. %d, %d, %d, %d\n", data_ind_now, nchannels * data_length * num_data, i_1, data_length);
                    continue;
                }
                h1_i = h1[data_ind_now];
            }
            else
            {
                template_ind_now = chan * N + i_1;
                if ((template_ind_now >= nchannels * N) | (i_1 > N))
                {
                    printf("Above full template range. %d, %d, %d, %d\n", template_ind_now, nchannels * N, i_1, N);
                    continue;
                }
                h1_i = h1[template_ind_now];
            }

            if (array_type_2 == ARRAY_TYPE_DATA)
            {
                data_ind_now = (data_ind * nchannels + chan) * data_length + i_2;
                if ((data_ind_now >= nchannels * data_length * num_data) | (i_2 > data_length))
                {
                    printf("Above full data range. %d, %d, %d, %d\n", data_ind_now, nchannels * data_length * num_data, i_1, data_length);
                    continue;
                }
                h2_i = h2[data_ind_now];
            }
            else
            {
                template_ind_now = chan * N + i_2;
                if ((template_ind_now >= nchannels * N) | (i_2 > N))
                {
                    printf("Above full template range. %d, %d, %d, %d\n", template_ind_now, nchannels * N, i_2, N);
                    continue;
                }
                h2_i = h2[template_ind_now];
            }

            *contrib_h1_h1 += (gcmplx::conj(h1_i) * h1_i * n); // n is invC
            *contrib_h2_h2 += (gcmplx::conj(h2_i) * h2_i * n); // n is invC
            *contrib_h1_h2 += (gcmplx::conj(h1_i) * h2_i * n); // n is invC
        }
    }
    else
    {
        for (int chan_1 = 0; chan_1 < 3; chan_1 += 1)
        {
            for (int chan_2 = 0; chan_2 <= chan_1; chan_2 += 1)
            {
                if (chan_1 == chan_2)
                {
                    multi_factor = 1.0; // PSD
                }
                else
                {
                    multi_factor = -2.0; // CSD
                }
                // nchannels has to be 3 here
                if (array_type_1 == ARRAY_TYPE_DATA)
                {
                    data_ind_now = (data_ind * nchannels + chan_1) * data_length + i_1;
                    if ((data_ind_now >= nchannels * data_length * num_data) | (i_1 > data_length))
                    {
                        printf("Above full data range. %d, %d, %d, %d\n", data_ind_now, nchannels * data_length * num_data, i_1, data_length);
                        continue;
                    }
                    h1_i = h1[data_ind_now];
                }
                else
                {
                    template_ind_now = chan_1 * N + i_1;
                    if ((template_ind_now >= nchannels * N) | (i_1 > N))
                    {
                        printf("Above full template range. %d, %d, %d, %d\n", template_ind_now, N * nchannels, i_1, N);
                        continue;
                    }
                    h1_i = h1[template_ind_now];
                }

                if (array_type_2 == ARRAY_TYPE_DATA)
                {
                    data_ind_now = (data_ind * nchannels + chan_2) * data_length + i_2;
                    if ((data_ind_now >= nchannels * data_length * num_data) | (i_2 > data_length))
                    {
                        printf("Above full data range. %d, %d, %d, %d\n", data_ind_now, nchannels * data_length * num_data, i_2, data_length);
                        continue;
                    }
                    h2_i = h2[data_ind_now];
                }
                else
                {
                    template_ind_now = chan_2 * N + i_2;
                    if ((template_ind_now >= nchannels * N) | (i_2 > N))
                    {
                        printf("Above full template range. %d, %d, %d, %d\n", template_ind_now, N * nchannels, i_2, N);
                        continue;
                    }
                    h2_i = h2[template_ind_now];
                }

                noise_ind_now = ((noise_ind * 3 + chan_1) * 3 + chan_2) * data_length + noise_i;
                if ((noise_ind_now >= nchannels * nchannels * data_length * num_noise) | (noise_i > data_length))
                {
                    printf("Above full noise range.%d, %d, %d, %d\n", noise_ind_now, nchannels * data_length * num_noise, noise_i, data_length);
                    continue;
                }
                n = noise[noise_ind_now];

                // TODO: check
                // multi_factor lets us skip off-diagonal double counting need
                *contrib_h1_h1 += (gcmplx::conj(h1_i) * h1_i * n); // n is invC
                *contrib_h2_h2 += (gcmplx::conj(h2_i) * h2_i * n); // n is invC
                *contrib_h1_h2 += (gcmplx::conj(h1_i) * h2_i * n); // n is invC
            }
        }
    }
}

#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ 
#endif
void get_ll(
    cmplx *d_h,
    cmplx *h_h,
    cmplx *data,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int num_data, 
    int num_noise)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    cmplx *wave = (cmplx *)shared_mem;
    cmplx *d_h_temp = &wave[3 * N];
    cmplx *h_h_temp = &d_h_temp[FFT::block_dim.x];
#else
    cmplx wave_tmp[3 * N];
    cmplx *wave = &wave_tmp[0];
    cmplx _d_h_temp[1];
    cmplx *d_h_temp = &_d_h_temp[0];
    cmplx _h_h_temp[1];
    cmplx *h_h_temp = &_d_h_temp[0];
#endif
    int nchannels = 3;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE)
        nchannels = 2;

    double df = 1. / T;

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    cmplx tmp1, tmp2;
    int data_ind, noise_ind;
    cmplx _ignore_this = 0.0;
    cmplx _ignore_this_2 = 0.0;
    double multi_factor = 1.0;
    int jj = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d, h;
    double n;
    int start_freq_ind;

#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif
    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {

        data_ind = data_index[bin_i];
        noise_ind = noise_index[bin_i];
        start_freq_ind = start_freq_inds[data_ind];

#ifdef __CUDACC__
        build_single_waveform<FFT>(
#else
        build_single_waveform(
#endif
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
            bin_i,
            tdi_channel_setup);

        CUDA_SYNCTHREADS;
        tmp1 = 0.0;
        tmp2 = 0.0;
        multi_factor = 1.0;

#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int incr2 = blockDim.x;
#else
        int start2 = 0;
        int incr2 = 1;
#endif
        for (int i = start2; i < N; i += incr2)
        {
            jj = i + start_ind - start_freq_ind;
            add_inner_product_contribution(
                &tmp1, &_ignore_this, &tmp2,
                data, wave, 
                jj, i, 
                ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                noise, noise_ind, jj,
                data_ind, tdi_channel_setup, data_length, N,
                num_data, num_noise
            );
            
        }
        CUDA_SYNCTHREADS;

        d_h_temp[tid] = tmp1;
        h_h_temp[tid] = tmp2;
        // if (((bin_i == 10) || (bin_i == 400))) printf("%d %d  %e %e %e %e \n", bin_i, tid, d_h_temp[tid].real(), d_h_temp[tid].imag(), h_h_temp[tid].real(), h_h_temp[tid].imag());
        CUDA_SYNCTHREADS;
#ifdef __CUDACC__
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                d_h_temp[tid] += d_h_temp[tid + s];
                h_h_temp[tid] += h_h_temp[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                // printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
            }
            CUDA_SYNCTHREADS;
        }
        CUDA_SYNCTHREADS;
#endif
        if (tid == 0)
        {
            d_h[bin_i] = 4.0 * df * d_h_temp[0];
            h_h[bin_i] = 4.0 * df * h_h_temp[0];
        }
        CUDA_SYNCTHREADS;
    }
}

#ifdef __CUDACC__
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
        inputs.data_arr,
        inputs.noise,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup, 
        inputs.num_data,
        inputs.num_noise);

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
#endif

void SharedMemoryLikeComp(
    cmplx *d_h,
    cmplx *h_h,
    cmplx *data,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise)
{

    InputInfo inputs;
    inputs.d_h = d_h;
    inputs.h_h = h_h;
    inputs.data_arr = data;
    inputs.noise = noise;
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
    inputs.start_freq_inds = start_freq_inds;
    inputs.data_length = data_length;
    inputs.tdi_channel_setup = tdi_channel_setup;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.num_data = num_data;
    inputs.num_noise = num_noise;

#ifdef __CUDACC__
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
#else
    get_ll(
        inputs.d_h,
        inputs.h_h,
        inputs.data_arr,
        inputs.noise,
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
        inputs.data_length,
        inputs.tdi_channel_setup, 
        inputs.num_data,
        inputs.num_noise);

#endif
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

#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ 
#endif
void get_swap_ll_diff(
    cmplx *d_h_remove,
    cmplx *d_h_add,
    cmplx *remove_remove,
    cmplx *add_add,
    cmplx *add_remove,
    cmplx *data,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int num_data,
    int num_noise)
{
    using complex_type = cmplx;

    unsigned int start_ind_add = 0;
    unsigned int start_ind_remove = 0;

#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    cmplx *wave_add = (cmplx *)shared_mem;
    cmplx *wave_remove = &wave_add[3 * N];

    cmplx *d_h_remove_arr = &wave_remove[3 * N];
    cmplx *d_h_add_arr = &d_h_remove_arr[FFT::block_dim.x];
    cmplx *remove_remove_arr = &d_h_add_arr[FFT::block_dim.x];
    cmplx *add_add_arr = &remove_remove_arr[FFT::block_dim.x];
    cmplx *add_remove_arr = &add_add_arr[FFT::block_dim.x];

#else
    cmplx _wave_add[3 * N];
    cmplx *wave_add = &_wave_add[0];
    cmplx _wave_remove[3 * N];
    cmplx *wave_remove = &_wave_remove[0];
    cmplx _d_h_remove_arr[1];
    cmplx *d_h_remove_arr = &_d_h_remove_arr[0];
    cmplx _d_h_add_arr[1];
    cmplx *d_h_add_arr = &_d_h_add_arr[0];
    cmplx _remove_remove_arr[1];
    cmplx *remove_remove_arr = &_remove_remove_arr[0];
    cmplx _add_add_arr[1];
    cmplx *add_add_arr = &_add_add_arr[0];
    cmplx _add_remove_arr[1];
    cmplx *add_remove_arr = &_add_remove_arr[0];
#endif
    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;
    int nchannels = 3;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE)
        nchannels = 2;

    double df = 1. / T;
#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif
    cmplx d_h_remove_temp = 0.0;
    cmplx d_h_add_temp = 0.0;
    cmplx remove_remove_temp = 0.0;
    cmplx add_add_temp = 0.0;
    cmplx add_remove_temp = 0.0;
    cmplx _ignore_this = 0.0;
    cmplx _ignore_this_2 = 0.0;
    int data_ind, noise_ind;

    int start_freq_ind;
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

#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif
    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {

        data_ind = data_index[bin_i];
        noise_ind = noise_index[bin_i];
        start_freq_ind = start_freq_inds[data_ind];

        d_h_remove_temp = 0.0;
        d_h_add_temp = 0.0;
        remove_remove_temp = 0.0;
        add_add_temp = 0.0;
        add_remove_temp = 0.0;

#ifdef __CUDACC__
        build_single_waveform<FFT>(
#else
        build_single_waveform(
#endif
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
            bin_i,
            tdi_channel_setup);
        CUDA_SYNCTHREADS;
#ifdef __CUDACC__
        build_single_waveform<FFT>(
#else
        build_single_waveform(
#endif
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
            bin_i,
            tdi_channel_setup);
        CUDA_SYNCTHREADS;

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
        // printf("ECK %d %d \n", total_i_vals, 2 * N);
        CUDA_SYNCTHREADS;
#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int incr2 = blockDim.x;
#else
        int start2 = 0;
        int incr2 = 1;
#endif
        if (total_i_vals < 2 * N)
        {

            for (int i = start2;
                 i < total_i_vals;
                 i += incr2)
            {

                j = lower_start_ind + i;                
                // if ((bin_i == 0)){
                // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, noise_ind * data_length + j, data_ind * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
                // }

                // if ((bin_i == 0)) printf("%d %e %e %e %e %e %e %e %e %e %e %e %e \n", tid, d_h_remove_temp.real(), d_h_remove_temp.imag(), d_h_add_temp.real(), d_h_add_temp.imag(), add_add_temp.real(), add_add_temp.imag(), remove_remove_temp.real(), remove_remove_temp.imag(), add_remove_temp.real(), add_remove_temp.imag());

                if (j < upper_start_ind)
                {
                    real_ind = i;
                    if (is_add_lower)
                    {
        
                        add_inner_product_contribution(
                            &d_h_add_temp, &_ignore_this, &add_add_temp,
                            data, wave_add, 
                            j, real_ind, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );

                        // // get <d|h> term
                        // d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                        // d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

                        // // <h|h>
                        // add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                        // add_add_temp += gcmplx::conj(h_E) * h_E / n_E;
                    }
                    else
                    {
                        add_inner_product_contribution(
                            &d_h_remove_temp, &_ignore_this, &remove_remove_temp,
                            data, wave_remove, 
                            j, real_ind, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );

                        // h_A = A_remove[real_ind];
                        // h_E = E_remove[real_ind];

                        // // get <d|h> term
                        // d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                        // d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

                        // // <h|h>
                        // remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                        // remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

                        // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                    }
                }
                else if (j >= lower_end_ind)
                {
                    real_ind = j - upper_start_ind;
                    if (!is_add_lower)
                    {
                        add_inner_product_contribution(
                            &d_h_add_temp, &_ignore_this, &add_add_temp,
                            data, wave_add, 
                            j, real_ind, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
                        // h_A_add = A_add[real_ind];
                        // h_E_add = E_add[real_ind];

                        // get <d|h> term
                        // d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                        // d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                        // // <h|h>
                        // add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                        // add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;
                    }
                    else
                    {
                        add_inner_product_contribution(
                            &d_h_remove_temp, &_ignore_this, &remove_remove_temp,
                            data, wave_remove, 
                            j, real_ind, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
                        
                        // h_A_remove = A_remove[real_ind];
                        // h_E_remove = E_remove[real_ind];

                        // get <d|h> term
                        // d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                        // d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                        // // <h|h>
                        // remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                        // remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;
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

                    // h_A_add = A_add[real_ind_add];
                    // h_E_add = E_add[real_ind_add];

                    // // get <d|h> term
                    // d_h_add_temp += gcmplx::conj(d_A) * h_A_add / n_A;
                    // d_h_add_temp += gcmplx::conj(d_E) * h_E_add / n_E;

                    // // <h|h>
                    // add_add_temp += gcmplx::conj(h_A_add) * h_A_add / n_A;
                    // add_add_temp += gcmplx::conj(h_E_add) * h_E_add / n_E;

                    add_inner_product_contribution(
                        &d_h_add_temp, &_ignore_this, &_ignore_this_2,
                        data, wave_add, 
                        j, real_ind_add, 
                        ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                        noise, noise_ind, j,
                        data_ind, tdi_channel_setup, data_length, N,
                        num_data, num_noise
                    );
                        
                    if (!is_add_lower)
                    {
                        real_ind_remove = i;
                    }
                    else
                    {
                        real_ind_remove = j - upper_start_ind;
                    }

                    // h_A_remove = A_remove[real_ind_remove];
                    // h_E_remove = E_remove[real_ind_remove];

                    // // get <d|h> term
                    // d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / n_A;
                    // d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / n_E;

                    // // <h|h>
                    // remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / n_A;
                    // remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / n_E;

                    add_inner_product_contribution(
                            &d_h_remove_temp, &_ignore_this, &_ignore_this_2,
                            data, wave_remove, 
                            j, real_ind_remove, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
                    
                    // add_remove_temp += gcmplx::conj(h_A_remove) * h_A_add / n_A;
                    // add_remove_temp += gcmplx::conj(h_E_remove) * h_E_add / n_E;

                    add_inner_product_contribution(
                        &add_remove_temp, &add_add_temp, &remove_remove_temp,
                        wave_add, wave_remove, 
                        real_ind_add, real_ind_remove, 
                        ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                        noise, noise_ind, j,
                        data_ind, tdi_channel_setup, data_length, N,
                        num_data, num_noise
                    );
                }
            }
        }
        else
        {
            for (int i = start2;
                 i < N;
                 i += incr2)
            {

                j = start_ind_remove + i - start_freq_ind;
                // n_A = noise_A[noise_ind * data_length + j];
                // n_E = noise_E[noise_ind * data_length + j];

                // d_A = data_A[data_ind * data_length + j];
                // d_E = data_E[data_ind * data_length + j];

                // // if ((bin_i == num_bin - 1))printf("CHECK remove: %d %e %e  \n", i, n_A, d_A.real());
                // //  calculate h term
                // h_A = A_remove[i];
                // h_E = E_remove[i];

                // // get <d|h> term
                // d_h_remove_temp += gcmplx::conj(d_A) * h_A / n_A;
                // d_h_remove_temp += gcmplx::conj(d_E) * h_E / n_E;

                // // <h|h>
                // remove_remove_temp += gcmplx::conj(h_A) * h_A / n_A;
                // remove_remove_temp += gcmplx::conj(h_E) * h_E / n_E;

                add_inner_product_contribution(
                            &d_h_remove_temp, &_ignore_this, &remove_remove_temp,
                            data, wave_remove, 
                            j, i, 
                            ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
            }

            for (int i = start2;
                 i < N;
                 i += incr2)
            {

                j = start_ind_add + i - start_freq_ind;

                // n_A = noise_A[noise_ind * data_length + j];
                // n_E = noise_E[noise_ind * data_length + j];

                // d_A = data_A[data_ind * data_length + j];
                // d_E = data_E[data_ind * data_length + j];

                // // if ((bin_i == 0))printf("CHECK add: %d %d %e %e %e %e  \n", i, j, n_A, d_A.real(), n_E, d_E.real());
                // //  calculate h term
                // h_A = A_add[i];
                // h_E = E_add[i];

                // // get <d|h> term
                // d_h_add_temp += gcmplx::conj(d_A) * h_A / n_A;
                // d_h_add_temp += gcmplx::conj(d_E) * h_E / n_E;

                // // <h|h>
                // add_add_temp += gcmplx::conj(h_A) * h_A / n_A;
                // add_add_temp += gcmplx::conj(h_E) * h_E / n_E;

                add_inner_product_contribution(
                    &d_h_add_temp, &_ignore_this, &add_add_temp,
                    data, wave_add, 
                    j, i, 
                    ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                    noise, noise_ind, j,
                    data_ind, tdi_channel_setup, data_length, N,
                    num_data, num_noise
                );
                    
            }
        }

        CUDA_SYNCTHREADS;
        d_h_remove_arr[tid] = d_h_remove_temp;

        d_h_add_arr[tid] = d_h_add_temp;
        add_add_arr[tid] = add_add_temp;
        remove_remove_arr[tid] = remove_remove_temp;
        add_remove_arr[tid] = add_remove_temp;
        CUDA_SYNCTHREADS;
#ifdef __CUDACC__
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
            CUDA_SYNCTHREADS;
        }
        CUDA_SYNCTHREADS;
#endif
        if (tid == 0)
        {
            d_h_remove[bin_i] = 4.0 * df * d_h_remove_arr[0];
            d_h_add[bin_i] = 4.0 * df * d_h_add_arr[0];
            add_add[bin_i] = 4.0 * df * add_add_arr[0];
            remove_remove[bin_i] = 4.0 * df * remove_remove_arr[0];
            add_remove[bin_i] = 4.0 * df * add_remove_arr[0];
        }
        CUDA_SYNCTHREADS;

        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}

#ifdef __CUDACC__
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
        inputs.data_arr,
        inputs.noise,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup,
        inputs.num_data,
        inputs.num_noise);

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
#endif

void SharedMemorySwapLikeComp(
    cmplx *d_h_remove,
    cmplx *d_h_add,
    cmplx *remove_remove,
    cmplx *add_add,
    cmplx *add_remove,
    cmplx *data,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise)
{

    InputInfo inputs;
    inputs.d_h_remove = d_h_remove;
    inputs.d_h_add = d_h_add;
    inputs.remove_remove = remove_remove;
    inputs.add_add = add_add;
    inputs.add_remove = add_remove;
    inputs.data_arr = data;
    inputs.noise = noise;
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
    inputs.start_freq_inds = start_freq_inds;
    inputs.data_length = data_length;
    inputs.tdi_channel_setup = tdi_channel_setup;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.num_data = num_data;
    inputs.num_noise = num_noise;

#ifdef __CUDACC__
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
#else
    get_swap_ll_diff(
        inputs.d_h_remove,
        inputs.d_h_add,
        inputs.remove_remove,
        inputs.add_add,
        inputs.add_remove,
        inputs.data_arr,
        inputs.noise,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup,
        inputs.num_data,
        inputs.num_noise);

#endif
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
#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ 
#endif
void get_chi_squared(
    cmplx *h1_h1,
    cmplx *h2_h2,
    cmplx *h1_h2,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int num_data, 
    int num_noise)
{
    using complex_type = cmplx;

    unsigned int start_ind_1 = 0;
    unsigned int start_ind_2 = 0;

    
#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    cmplx *wave_1 = (cmplx *)shared_mem;
    cmplx *wave_2 = &wave_1[3 * N];
    cmplx *h1_h1_arr = &wave_2[3 * N];
    cmplx *h2_h2_arr = &h1_h1_arr[FFT::block_dim.x];
    cmplx *h1_h2_arr = &h2_h2_arr[FFT::block_dim.x];

#else
    cmplx _wave_1[3 * N];
    cmplx *wave_1 = &_wave_1[0];
    cmplx _wave_2[3 * N];
    cmplx *wave_2 = &_wave_2[0];
    cmplx _h1_h1_arr[1];
    cmplx *h1_h1_arr = &_h1_h1_arr[0];
    cmplx _h2_h2_arr[1];
    cmplx *h2_h2_arr = &h2_h2_arr[0];
    cmplx _h1_h2_arr[1];
    cmplx *h1_h2_arr = &h1_h2_arr[0];
#endif
    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
#ifdef __CUDACC__
    int tid = threadIdx.x;
#else       
    int tid = 0;
#endif
    cmplx h1_h1_tmp = 0.0;
    cmplx h2_h2_tmp = 0.0;
    cmplx h1_h2_tmp = 0.0;

    int noise_ind;

    int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
    bool is_1_lower;
    int total_i_vals;

    cmplx h, h_1, h_2;
    int real_ind, real_ind_1, real_ind_2;
    int start_freq_ind;
    int jj = 0;
    int j = 0;
    int last_row_end;
    int output_ind, output_ind2;
    cmplx _ignore_this = 0.0;
    cmplx _ignore_this_2 = 0.0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);
    double n;
#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;

    int start2 = blockIdx.y;
    int incr2 = gridDim.y;    
#else
    int start1 = 0;
    int incr1 = 1;

    int start2 = 0;
    int incr2 = 1;
#endif
    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {
        for (int bin_j = start2 + bin_i + 1; bin_j < num_bin_all; bin_j += incr2)
        {
            h1_h1_tmp = 0.0;
            h1_h2_tmp = 0.0;
            h2_h2_tmp = 0.0;
            noise_ind = noise_index[bin_i]; // must be the same
            start_freq_ind = start_freq_inds[noise_ind];
            // get index into upper triangular array (without diagonal)
            last_row_end = bin_i * (bin_i + 1) / 2;
            output_ind = num_bin_all * bin_i + bin_j - int(((bin_i + 2) * (bin_i + 1)) / 2);
            // output_ind2 = last_row_end + bin_j; // INDEX SO DOES NOT HAVE +1 

#ifdef __CUDACC__
            build_single_waveform<FFT>(
#else
            build_single_waveform(
#endif
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
                bin_i, tdi_channel_setup);
            CUDA_SYNCTHREADS;
#ifdef __CUDACC__
            build_single_waveform<FFT>(
#else
            build_single_waveform(
#endif
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
                bin_i, tdi_channel_setup);
            CUDA_SYNCTHREADS;

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

#ifdef __CUDACC__
            int start3 = threadIdx.x;
            int incr3 = blockDim.x;
#else
            int start3 = 0;
            int incr3 = 1;
#endif
            CUDA_SYNCTHREADS;
            if (total_i_vals < 2 * N)
            {
                for (int i = start3;
                    i < total_i_vals;
                    i += incr3)
                {

                    j = lower_start_ind + i;

                    // if ((bin_i == 0)){
                    // printf("%d %d %d %d %d %d %d %e %e %e %e %e %e\n", i, j, noise_ind, data_ind, noise_ind * data_length + j, data_ind * data_length + j, data_length, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E);
                    // }

                    // if ((bin_i == 0)) printf("%d %e %e %e %e %e %e %e %e %e %e %e %e \n", tid, d_h_2_temp.real(), d_h_2_temp.imag(), d_h_1_temp.real(), d_h_1_temp.imag(), h2_h2_temp.real(), h2_h2_temp.imag(), h1_h1_temp.real(), h1_h1_temp.imag(), h1_h2_temp.real(), h1_h2_temp.imag());

                    if (j < upper_start_ind)
                    {
                        real_ind = i;
                        if (is_1_lower)
                        {

                            // h_A = A_1[real_ind];
                            // h_E = E_1[real_ind];
                            
                            add_inner_product_contribution(
                                &_ignore_this, &h1_h1_tmp, &_ignore_this_2,
                                wave_1, wave_1, 
                                real_ind, real_ind, 
                                ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                                noise, noise_ind, j,
                                -1, tdi_channel_setup, data_length, N,
                                num_data, num_noise
                            );
                            
                            // <h|h>
                            // h2_h2_temp += gcmplx::conj(h_A) * h_A / n_A;
                            // h2_h2_temp += gcmplx::conj(h_E) * h_E / n_E;
                        }
                        else
                        {
                            // h_A = A_2[real_ind];
                            // h_E = E_2[real_ind];

                            // // <h|h>
                            // h1_h1_temp += gcmplx::conj(h_A) * h_A / n_A;
                            // h1_h1_temp += gcmplx::conj(h_E) * h_E / n_E;

                            add_inner_product_contribution(
                                &_ignore_this, &h2_h2_tmp, &_ignore_this_2,
                                wave_2, wave_2, 
                                real_ind, real_ind, 
                                ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                                noise, noise_ind, j,
                                -1, tdi_channel_setup, data_length, N,
                                num_data, num_noise
                            );

                            // if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                        }
                    }
                    else if (j >= lower_end_ind)
                    {
                        real_ind = j - upper_start_ind;
                        if (!is_1_lower)
                        {

                            // h_A_1 = A_1[real_ind];
                            // h_E_1 = E_1[real_ind];

                            // <h|h>
                            add_inner_product_contribution(
                                &_ignore_this, &h1_h1_tmp, &_ignore_this_2,
                                wave_1, wave_1, 
                                real_ind, real_ind, 
                                ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                                noise, noise_ind, j,
                                -1, tdi_channel_setup, data_length, N,
                                num_data, num_noise
                            );
                            
                            // h2_h2_temp += gcmplx::conj(h_A_1) * h_A_1 / n_A;
                            // h2_h2_temp += gcmplx::conj(h_E_1) * h_E_1 / n_E;
                        }
                        else
                        {
                            // h_A_2 = A_2[real_ind];
                            // h_E_2 = E_2[real_ind];

                            // // <h|h>
                            // h1_h1_temp += gcmplx::conj(h_A_2) * h_A_2 / n_A;
                            // h1_h1_temp += gcmplx::conj(h_E_2) * h_E_2 / n_E;

                            add_inner_product_contribution(
                                &_ignore_this, &h2_h2_tmp, &_ignore_this_2,
                                wave_2, wave_2, 
                                real_ind, real_ind, 
                                ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                                noise, noise_ind, j,
                                -1, tdi_channel_setup, data_length, N,
                                num_data, num_noise
                            );

                            
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

                        // h_A_1 = A_1[real_ind_1];
                        // h_E_1 = E_1[real_ind_1];

                        // // <h|h>
                        // h2_h2_temp += gcmplx::conj(h_A_1) * h_A_1 / n_A;
                        // h2_h2_temp += gcmplx::conj(h_E_1) * h_E_1 / n_E;

                        if (!is_1_lower)
                        {
                            real_ind_2 = i;
                        }
                        else
                        {
                            real_ind_2 = j - upper_start_ind;
                        }

                        add_inner_product_contribution(
                            &h1_h2_tmp, &h1_h1_tmp, &h2_h2_tmp,
                            wave_1, wave_2, 
                            real_ind_1, real_ind_2, 
                            ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, j,
                            -1, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
                       
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
                CUDA_SYNCTHREADS;
                continue;
            }

            CUDA_SYNCTHREADS;
            h2_h2_arr[tid] = h2_h2_tmp;
            h1_h1_arr[tid] = h1_h1_tmp;
            h1_h2_arr[tid] = h1_h2_tmp;
            CUDA_SYNCTHREADS;

#ifdef __CUDACC__
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
                CUDA_SYNCTHREADS;
            }
            CUDA_SYNCTHREADS;
#endif
            if (tid == 0)
            {
                h2_h2[output_ind] = 4.0 * df * h2_h2_arr[0];
                h1_h1[output_ind] = 4.0 * df * h1_h1_arr[0];
                h1_h2[output_ind] = 4.0 * df * h1_h2_arr[0];
            }
            CUDA_SYNCTHREADS;

        }
        // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}


#ifdef __CUDACC__ 
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
        inputs.noise,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup,
        inputs.num_data, 
        inputs.num_noise);

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
#endif

void SharedMemoryChiSquaredComp(
    cmplx *h1_h1,
    cmplx *h2_h2,
    cmplx *h1_h2,
    double *noise,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data, 
    int num_noise)
{

    InputInfo inputs;
    inputs.h1_h1 = h1_h1;
    inputs.h2_h2 = h2_h2;
    inputs.h1_h2 = h1_h2;
    inputs.noise = noise;

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
    inputs.start_freq_inds = start_freq_inds;
    inputs.data_length = data_length;
    inputs.tdi_channel_setup = tdi_channel_setup;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.num_data = num_data;
    inputs.num_noise = num_noise;

#ifdef __CUDACC__
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
#else
    get_chi_squared(
        inputs.h1_h1,
        inputs.h2_h2,
        inputs.h1_h2,
        inputs.noise,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup,
        inputs.num_data, 
        inputs.num_noise);
#endif
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

#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ 
#endif
void generate_global_template(
    cmplx *tmplt, 
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    int nchannels = 3;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE)
        nchannels = 2;

#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    cmplx *wave = (cmplx *)shared_mem; // N * nchannels length

#else
    cmplx _wave[3 * N];
    cmplx *wave = &_wave[0];
    
#endif
    // auto this_block_data = tdi_out
    //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif
    
    int template_ind;
    double factor;
    int start_freq_ind;
    int jj = 0;
#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif
    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {

        template_ind = template_index[bin_i];
        factor = factors[bin_i];
        start_freq_ind = start_freq_inds[template_ind];
        
        // printf("%d %d %d \n", template_ind, start_freq_ind, bin_i);
        
#ifdef __CUDACC__
        build_single_waveform<FFT>(
#else
        build_single_waveform(
#endif
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
            bin_i,
            tdi_channel_setup);

        CUDA_SYNCTHREADS;
#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int incr2 = blockDim.x;
#else
        int start2 = 0;
        int incr2 = 1;
#endif
        for (int chan = 0; chan < nchannels; chan += 1)
        {
            for (int i = start2; i < N; i += incr2)
            {
                jj = i + start_ind - start_freq_ind;
#ifdef __CUDACC__
                atomicAddComplex(&tmplt[(template_ind * nchannels + chan) * data_length + jj], factor * wave[chan * N + i]);
#else           
                tmplt[(template_ind * nchannels + chan) * data_length + jj] = factor * wave[chan * N + i];
#endif
            }
        }
        CUDA_SYNCTHREADS;
    }
}


#ifdef __CUDACC__
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
        inputs.data_arr,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup);

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
#endif

void SharedMemoryGenerateGlobal(
    cmplx *data,
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
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize)
{

    InputInfo inputs;
    inputs.data_arr = data;
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
    inputs.start_freq_inds = start_freq_inds;
    inputs.data_length = data_length;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.tdi_channel_setup = tdi_channel_setup;

#ifdef __CUDACC__
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
#else
    generate_global_template(
        inputs.data_arr,
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
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup);

#endif
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


#ifdef __CUDACC__
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ 
#endif
void get_fstat_ll(
    cmplx *M_mat,
    cmplx *N_arr,
    cmplx *data,
    double *noise,
    int *data_index,
    int *noise_index,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int num_data, 
    int num_noise)
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

#ifdef __CUDACC__
    extern __shared__ unsigned char shared_mem[];
    // NEEDS to be 4 * 3 * N in length for 4 waveforms
    cmplx *wave = (cmplx *)shared_mem;
    cmplx *M_temp = &wave[4 * 3 * N];
    cmplx *N_temp = &M_temp[FFT::block_dim.x];

#else
    cmplx wave_tmp[4 * 3 * N];
    cmplx *wave = &wave_tmp[0];

    cmplx _M_temp[1] = {0.0};
    cmplx *M_temp = &_M_temp[0];
    cmplx _N_temp[1] = {0.0};
    cmplx *N_temp = &_N_temp[0];
    
#endif
    int nchannels = 3;
    if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE)
        nchannels = 2;

    double df = 1. / T;
#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    cmplx tmp1, tmp2;
    int data_ind, noise_ind;
    cmplx _ignore_this = 0.0;
    cmplx _ignore_this_2 = 0.0;
    double multi_factor = 1.0;
    int jj = 0;
    // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

    cmplx d, h;
    double n;
    int start_freq_ind;

    double iota_arr[4] = {M_PI / 2.0, M_PI / 2.0, M_PI / 2.0, M_PI / 2.0};
    double psi_arr[4] = {0.0, M_PI / 4.0, 0.0, M_PI/ 4.0};
    double A_arr[4] = {2., 2., 2., 2.};
    double phase_arr[4] = {0.0, M_PI, 3. * M_PI / 2., M_PI / 2.};

    // Now need to calculate the Filters A^{i} (Cornish & Crowder '05)
    // A^{1} -> iota = pi/2, psi = 0,    A = 2, phase = 0
    // A^{2} -> iota = pi/2, psi = pi/4, A = 2, phase = pi
    // A^{3} -> iota = pi/2, psi = 0,    A = 2, phase = 3*pi/2
    // A^{4} -> iota = pi/2, psi = pi/4, A = 2, phase = pi/2

    double psi_tmp, iota_tmp, phase_tmp, A_tmp; 

#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int incr1 = gridDim.x;
#else
    int start1 = 0;
    int incr1 = 1;
#endif
    for (int bin_i = start1; bin_i < num_bin_all; bin_i += incr1)
    {

        data_ind = data_index[bin_i];
        noise_ind = noise_index[bin_i];
        start_freq_ind = start_freq_inds[data_ind];

        for (int ii = 0; ii < 4; ii += 1)
        {
            CUDA_SYNCTHREADS;
#ifdef __CUDACC__
            build_single_waveform<FFT>(
#else
            build_single_waveform(
#endif
                &wave[ii * 3 * N],
                &start_ind,
                A_arr[ii],
                f0[bin_i],
                fdot0[bin_i],
                fddot0[bin_i],
                phase_arr[ii],
                iota_arr[ii],
                psi_arr[ii],
                lam[bin_i],
                theta[bin_i],
                T,
                dt,
                N,
                bin_i,
                tdi_channel_setup
            );
            CUDA_SYNCTHREADS;
        }
    

#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int incr2 = blockDim.x;
#else
        int start2 = 0;
        int incr2 = 1;
#endif
        for (int ii = 0; ii < 4; ii += 1)
        {
            CUDA_SYNCTHREADS;
            tmp2 = 0.0;

            // if ((tid == 20) && (bin_i == 0))
            //     printf("%d %d %e %e %e %e\n", ii, bin_i, wave[tid].real(), wave[tid].imag(), wave[N + tid].real(), wave[N + tid].imag(), wave[2 * N + tid].real(), wave[2 * N + tid].imag(), A_arr[ii], phase_arr[ii], iota_arr[ii], psi_arr[ii]);
            
            for (int i = start2; i < N; i += incr2)
            {
                jj = i + start_ind - start_freq_ind;
                for (int chan = 0; chan < nchannels; chan += 1)
                {
                    
                    add_inner_product_contribution(
                        &tmp2, &_ignore_this, &_ignore_this_2, 
                        data, &wave[(ii * 3 * N) + chan * N], 
                        jj, i, 
                        ARRAY_TYPE_DATA, ARRAY_TYPE_TEMPLATE,
                        noise, noise_ind, jj,
                        data_ind, tdi_channel_setup, data_length, N,
                        num_data, num_noise
                    );
                }
            }
            CUDA_SYNCTHREADS;
            N_temp[tid] = tmp2;

#ifdef __CUDACC__
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    N_temp[tid] += N_temp[tid + s];
                }
                CUDA_SYNCTHREADS;
            }
            CUDA_SYNCTHREADS;
#endif
            CUDA_SYNCTHREADS;
            if (tid == 0)
            {
                N_arr[bin_i * 4 + ii] = 4.0 * df * N_temp[0];
            }
            CUDA_SYNCTHREADS;
            for (int kk = ii; kk < 4; kk += 1)
            {
                CUDA_SYNCTHREADS;
                tmp1 = 0.0;
                multi_factor = 1.0;
                
                CUDA_SYNCTHREADS;
                for (int i = start2; i < N; i += incr2)
                {
                    jj = i + start_ind - start_freq_ind;
                    for (int chan = 0; chan < nchannels; chan += 1)
                    {
                        add_inner_product_contribution(
                            &tmp1, &_ignore_this, &_ignore_this_2, 
                            &wave[(ii * 3 * N) + chan * N], &wave[(kk * 3 * N) + chan * N], 
                            i, i,
                            ARRAY_TYPE_TEMPLATE, ARRAY_TYPE_TEMPLATE,
                            noise, noise_ind, jj,
                            data_ind, tdi_channel_setup, data_length, N,
                            num_data, num_noise
                        );
                    }
                }
                
                M_temp[tid] = tmp1;

                        // if (((bin_i == 10) || (bin_i == 400))) printf("%d %d  %e %e %e %e \n", bin_i, tid, d_h_temp[tid].real(), d_h_temp[tid].imag(), h_h_temp[tid].real(), h_h_temp[tid].imag());
                CUDA_SYNCTHREADS;
#ifdef __CUDACC__
                for (unsigned int s = 1; s < blockDim.x; s *= 2)
                {
                    if (tid % (2 * s) == 0)
                    {
                        M_temp[tid] += M_temp[tid + s];
                    }
                    CUDA_SYNCTHREADS;
                }
                CUDA_SYNCTHREADS;
                if (tid == 0)
                {
                    M_mat[(bin_i * 4 + ii) * 4 + kk] = 4.0 * df * M_temp[0];
                    M_mat[(bin_i * 4 + kk) * 4 + ii] = 4.0 * df * M_temp[0];
                }
                CUDA_SYNCTHREADS;
#endif
            }
            CUDA_SYNCTHREADS;
        }
    }
}

#ifdef __CUDACC__
// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template <unsigned int Arch, unsigned int N>
void get_fstat_ll_wrap(InputInfo inputs)
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

    // first is 4 waveforms, second is M_temp and N_temp
    auto shared_memory_size_mine = 4 * 3 * N * sizeof(cmplx) + 2 * FFT::block_dim.x * sizeof(cmplx);
    // std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_fstat_ll<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    // std::cout << (int) FFT::block_dim.x << std::endl;
    //  Invokes kernel with FFT::block_dim threads in CUDA block
    get_fstat_ll<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.M_mat,
        inputs.N_arr,
        inputs.data_arr,
        inputs.noise,
        inputs.data_index,
        inputs.noise_index,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup, 
        inputs.num_data,
        inputs.num_noise);

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
struct get_fstat_ll_wrap_functor
{
    void operator()(InputInfo inputs) { return get_fstat_ll_wrap<Arch, N>(inputs); }
};
#endif

void SharedMemoryFstatLikeComp(
    cmplx *M_mat,
    cmplx *N_arr,
    cmplx *data,
    double *noise,
    int *data_index,
    int *noise_index,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise)
{

    InputInfo inputs;
    inputs.M_mat = M_mat;
    inputs.N_arr = N_arr;
    inputs.data_arr = data;
    inputs.noise = noise;
    inputs.data_index = data_index;
    inputs.noise_index = noise_index;
    inputs.f0 = f0;
    inputs.fdot0 = fdot0;
    inputs.fddot0 = fddot0;
    inputs.lam = lam;
    inputs.theta = theta;
    inputs.T = T;
    inputs.dt = dt;
    inputs.N = N;
    inputs.num_bin_all = num_bin_all;
    inputs.start_freq_inds = start_freq_inds;
    inputs.data_length = data_length;
    inputs.tdi_channel_setup = tdi_channel_setup;
    inputs.device = device;
    inputs.do_synchronize = do_synchronize;
    inputs.num_data = num_data;
    inputs.num_noise = num_noise;

#ifdef __CUDACC__
    switch (N)
    {
    // All SM supported by cuFFTDx
    case 32:
        example::sm_runner<get_fstat_ll_wrap_functor, 32>(inputs);
        return;
    case 64:
        example::sm_runner<get_fstat_ll_wrap_functor, 64>(inputs);
        return;
    case 128:
        example::sm_runner<get_fstat_ll_wrap_functor, 128>(inputs);
        return;
    case 256:
        example::sm_runner<get_fstat_ll_wrap_functor, 256>(inputs);
        return;
    case 512:
        example::sm_runner<get_fstat_ll_wrap_functor, 512>(inputs);
        return;
    case 1024:
        example::sm_runner<get_fstat_ll_wrap_functor, 1024>(inputs);
        return;
    case 2048:
        example::sm_runner<get_fstat_ll_wrap_functor, 2048>(inputs);
        return;

    default:
    {
        printf("CHECKING\n");
        throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
    }
    }
#else
    get_fstat_ll(
        inputs.M_mat,
        inputs.N_arr,
        inputs.data_arr,
        inputs.noise,
        inputs.data_index,
        inputs.noise_index,
        inputs.f0,
        inputs.fdot0,
        inputs.fddot0,
        inputs.lam,
        inputs.theta,
        inputs.T,
        inputs.dt,
        inputs.N,
        inputs.num_bin_all,
        inputs.start_freq_inds,
        inputs.data_length,
        inputs.tdi_channel_setup, 
        inputs.num_data,
        inputs.num_noise);

#endif
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