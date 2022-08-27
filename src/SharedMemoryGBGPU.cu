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

__device__
void spacecraft(double P1[3], double P2[3], double P3[3], double t)
{
        // """Compute space craft positions as a function of time"""
        //# kappa and lambda are constants determined in the Constants.h file

        //# angular quantities defining orbit
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


template<class FFT>
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
    int bin_i
)
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
        for (int j = 0; j < 6; j+= 1)
        {
            kdotr[j] = 0.0;
        }

        #pragma unroll
        for (int j = 0; j < 3; j+= 1)
        {
            kdotP[j] = 0.0;
            Aij[j] = 0.0;
        }
        
        #pragma unroll
        for (int j = 0; j < 3; j+= 1)
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
        for (int j = 0; j < 3; j+= 1)
        {
            xi_tmp = t - kdotP[j];
            xi[j] = xi_tmp;
            fi = (
                f0
                + fdot0 * xi_tmp
                + 1 / 2.0 * fddot0 * (xi_tmp * xi_tmp)
            );
            fonfs[j] = fi / fstar;
        }

        #pragma unroll
        for (int j = 0; j < 3; j+= 1)
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
            argS = (
                phi0
                + (om - df) * t
                + M_PI * fdot0 * (xi_tmp * xi_tmp)
                + 1. / 3. * M_PI * fddot0 * (xi_tmp * xi_tmp * xi_tmp)
            );

            kdotP[j] = om * kdotP[j] - argS;
        }

        #pragma unroll
        for (int j = 0; j < 6; j += 1)
        {

            s = s_all[j];
            A_ind = j % 3;
            arg_ij = 0.5 * fonfs[s] * (1 + kdotr[j]);

            Gs[j] = (
                0.25
                * sin(arg_ij)
                / arg_ij
                * exp(-I * (arg_ij + kdotP[s]))
                * Aij[A_ind]
            );

        }

        f = (
            f0
            + fdot0 * t
            + 1 / 2.0 * fddot0 * (t * t)
        );

        omL = f / fstar;
        SomL = sin(omL);
        fctr = gcmplx::exp(-I * omL);
        fctr2 = 4.0 * omL * SomL * fctr / amp;

        // order is (12, 23, 31, 21, 32, 13)
        
        // Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        tmp_X = fctr2 * ((Gs[3] - Gs[2] + (Gs[0] - Gs[5]) * fctr));

        //tmp_X = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t);  
        //tmp2_X = complex_type {tmp_X.real(), tmp_X.imag()};
        X[i] = tmp_X;

        tmp_Y = fctr2 * ((Gs[4] - Gs[0] + (Gs[1] - Gs[3]) * fctr));
        //tmp2_Y = complex_type {tmp_Y.real(), tmp_Y.imag()};
        Y[i] = tmp_Y;

        tmp_Z = fctr2 * ((Gs[5] - Gs[1] + (Gs[2] - Gs[4]) * fctr));
        //tmp_Z = sin(2. * M_PI * f0 * t) + I * cos(2. * M_PI * f0 * t); 
        //tmp2_Z = complex_type {tmp_Z.real(), tmp_Z.imag()};
        Z[i] = tmp_Z;
    }
    
    __syncthreads();

    FFT().execute(reinterpret_cast<void*>(X));
    FFT().execute(reinterpret_cast<void*>(Y));
    FFT().execute(reinterpret_cast<void*>(Z));

    __syncthreads();

    for (int i = threadIdx.x; i < N; i+= blockDim.x)
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

    //if (tdi2)
    //{
    //    omegaL = 2. * M_PI * f0_out * (Larm / Clight);
    //    tdi2_factor = 2.0 * I * sin(2. * omegaL) * exp(-2. * I * omegaL);
    //    fctr3 *= tdi2_factor;
    //}
    
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
    

    /*__syncthreads();
    if ((bin_i == 0) && (threadIdx.x == 0))
    {
        for (int jj = 0; jj < N; jj += 20)
        {
            printf("%d %.18e %.18e,\n", jj, X[jj].real(), X[jj].imag());
        }
    }*/
    // convert to A, E, T (they sit in X,Y,Z)
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        AET_from_XYZ_swap(&X[i], &Y[i], &Z[i]);
    }
    
    __syncthreads();

    double fmin = (q - ((double) N) / 2.) / T;
    *start_ind = (int) rint(fmin * T);

    return;
}

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_waveform(
    cmplx* tdi_out,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all
) 
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__  unsigned char shared_mem[];

    //auto this_block_data = tdi_out
 //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    //example::io<FFT>::load_to_smem(this_block_data, shared_mem);
    cmplx *wave = (cmplx*)shared_mem;

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
            bin_i
        );

        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            for (int j = 0; j < 3; j += 1)
            {
                tdi_out[(bin_i * 3 + j) * N + i] = wave[j * N + i];
            }
        }

        __syncthreads();

        /*if ((threadIdx.x == 0) && (bin_i == 80))
        {
            printf("CHECK: %d %e %e %e %e\n", bin_i, wave[0].real(), wave[0].imag(), tdi_out[(bin_i * 3 + 0) * N + 0].real(), tdi_out[(bin_i * 3 + 0) * N + 0].imag());
        }*/
        
        //example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}




// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<unsigned int Arch, unsigned int N>
void simple_block_fft(InputInfo inputs) {
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
 
    using FFT          = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    auto shared_memory_size_mine = 3 * N * sizeof(cmplx);
    //std::cout << "input [1st FFT]:\n" << shared_memory_size_mine << std::endl;
    //for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //    std::cout << data[i].x << " " << data[i].y << std::endl;
    //}

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_waveform<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    //std::cout << (int) FFT::block_dim.x << std::endl;
    // Invokes kernel with FFT::block_dim threads in CUDA block
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
        inputs.num_bin_all
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    //std::cout << "output [1st FFT]:\n";
    //for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //    std::cout << data[i].x << " " << data[i].y << std::endl;
    //}

    //std::cout << shared_memory_size << std::endl;
    //std::cout << "Success" <<  std::endl;

}

template<unsigned int Arch, unsigned int N>
struct simple_block_fft_functor {
    void operator()(InputInfo inputs) { return simple_block_fft<Arch, N>(inputs); }
};
void SharedMemoryWaveComp(
    cmplx* tdi_out,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all
) 
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
    
    switch (N) {
            // All SM supported by cuFFTDx
            case 32: example::sm_runner<simple_block_fft_functor, 32>(inputs); return;
            case 64: example::sm_runner<simple_block_fft_functor, 64>(inputs); return;
            case 128: example::sm_runner<simple_block_fft_functor, 128>(inputs); return;
            case 256: example::sm_runner<simple_block_fft_functor, 256>(inputs); return;
            case 512: example::sm_runner<simple_block_fft_functor, 512>(inputs); return;
            case 1024: example::sm_runner<simple_block_fft_functor, 1024>(inputs); return;
            case 2048: example::sm_runner<simple_block_fft_functor, 2048>(inputs); return;

            default: {
                throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
            }
        }
   
    //const unsigned int arch = example::get_cuda_device_arch();
    //simple_block_fft<800>(x);
}


//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////
//////////////////




template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void get_ll(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind
) 
{
    using complex_type = cmplx;

    unsigned int start_ind = 0;

    extern __shared__  unsigned char shared_mem[];

    //auto this_block_data = tdi_out
 //+ cufftdx::size_of<FFT>::value * FFT::ffts_per_block * blockIdx.x;

    double df = 1. / T;
    int tid = threadIdx.x;
    cmplx *wave = (cmplx*)shared_mem;
    cmplx *A = &wave[0];
    cmplx *E = &wave[N];

    cmplx *d_h_temp = &wave[3 * N];
    cmplx *h_h_temp = &d_h_temp[FFT::block_dim.x];
    cmplx tmp1, tmp2;

    int jj = 0;
    //example::io<FFT>::load_to_smem(this_block_data, shared_mem);
    
    cmplx d_A, d_E, h_A, h_E, n_A, n_E;
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
            bin_i
        );

        __syncthreads();
        tmp1 = 0.0;
        tmp2 = 0.0;
        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            jj = i + start_ind - start_freq_ind;
            d_A = data_A[jj];
            d_E = data_E[jj];
            n_A = noise_E[jj];
            n_E = noise_A[jj];

            
            
            h_A = A[i];
            h_E = E[i];

            tmp1 += (
                gcmplx::conj(d_A) * h_A / n_A
                + gcmplx::conj(d_E) * h_E / n_E
            );

            tmp2 += (
                gcmplx::conj(h_A) * h_A / n_A
                + gcmplx::conj(h_E) * h_E / n_E
            );
            __syncthreads();
        }

        d_h_temp[tid] = tmp1;
        h_h_temp[tid] = tmp2;
        //if (((bin_i == 10) || (bin_i == 400))) printf("%d %d  %e %e %e %e \n", bin_i, tid, d_h_temp[tid].real(), d_h_temp[tid].imag(), h_h_temp[tid].real(), h_h_temp[tid].imag());
        __syncthreads();
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                d_h_temp[tid] += d_h_temp[tid + s];
                h_h_temp[tid] += h_h_temp[tid + s];
                //if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
                //printf("%d %d %d %d %.18e %.18e %.18e %.18e %.18e %.18e %d\n", bin_i, channel_i, s, tid, sdata[tid].real(), sdata[tid].imag(), tmp.real(), tmp.imag(), sdata[tid + s].real(), sdata[tid + s].imag(), s + tid);
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
        
        //example::io<FFT>::store_from_smem(shared_mem, this_block_data);
    }
    //
}


// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C double precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<unsigned int Arch, unsigned int N>
void get_ll_wrap(InputInfo inputs) {
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
 
    using FFT          = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<double>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<Arch>());
    using complex_type = cmplx;

    // Allocate managed memory for input/output
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(cmplx);

    // Shared memory must fit input data and must be big enough to run FFT
    auto shared_memory_size = std::max((unsigned int)FFT::shared_memory_size, (unsigned int)size_bytes);

    // first is waveforms, second is d_h_temp and h_h_temp
    auto shared_memory_size_mine = 3 * N * sizeof(cmplx) + 2 * FFT::block_dim.x * sizeof(cmplx);
    //std::cout << "input [1st FFT]:\n" << size  << "  " << size_bytes << "  " << FFT::shared_memory_size << std::endl;
    //for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //    std::cout << data[i].x << " " << data[i].y << std::endl;
    //}

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        get_waveform<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size_mine));

    //std::cout << (int) FFT::block_dim.x << std::endl;
    // Invokes kernel with FFT::block_dim threads in CUDA block
    get_ll<FFT><<<inputs.num_bin_all, FFT::block_dim, shared_memory_size_mine>>>(
        inputs.d_h,
        inputs.h_h,
        inputs.data_A,
        inputs.data_E,
        inputs.noise_A,
        inputs.noise_E,
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
        inputs.start_freq_ind
    );

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    //std::cout << "output [1st FFT]:\n";
    //for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //    std::cout << data[i].x << " " << data[i].y << std::endl;
    //}

    //std::cout << shared_memory_size << std::endl;
    //std::cout << "Success" <<  std::endl;

}

template<unsigned int Arch, unsigned int N>
struct get_ll_wrap_functor {
    void operator()(InputInfo inputs) { return get_ll_wrap<Arch, N>(inputs); }
};


void SharedMemoryLikeComp(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind
) 
{

    InputInfo inputs;
    inputs.d_h = d_h;
    inputs.h_h = h_h;
    inputs.data_A = data_A;
    inputs.data_E = data_E;
    inputs.noise_A = noise_A;
    inputs.noise_E = noise_E;
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
    
    switch (N) {
            // All SM supported by cuFFTDx
            case 32: example::sm_runner<get_ll_wrap_functor, 32>(inputs); return;
            case 64: example::sm_runner<get_ll_wrap_functor, 64>(inputs); return;
            case 128: example::sm_runner<get_ll_wrap_functor, 128>(inputs); return;
            case 256: example::sm_runner<get_ll_wrap_functor, 256>(inputs); return;
            case 512: example::sm_runner<get_ll_wrap_functor, 512>(inputs); return;
            case 1024: example::sm_runner<get_ll_wrap_functor, 1024>(inputs); return;
            case 2048: example::sm_runner<get_ll_wrap_functor, 2048>(inputs); return;

            default: {
                throw std::invalid_argument("N must be a multiple of 2 between 32 and 2048.");
            }
        }
   
    //const unsigned int arch = example::get_cuda_device_arch();
    //simple_block_fft<800>(x);
}
