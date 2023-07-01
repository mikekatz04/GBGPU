#ifndef __SHAREDMEMORY_GBGPU_HPP__
#define __SHAREDMEMORY_GBGPU_HPP__

#include "global.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class GalacticBinaryParams{
    public:
        double* amp; 
        double* f0; 
        double* fdot0; 
        double* fddot0; 
        double* phi0; 
        double* iota;
        double* psi; 
        double* lam;
        double* theta;
        double T; 
        double dt;
        int N;
        int num_bin_all;
        int start_freq_ind;

        GalacticBinaryParams(
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
        );

};

class DataPackage{
    public:
        cmplx* data_A; 
        cmplx* data_E; 
        double* psd_A; 
        double* psd_E; 
        double df; 
        int data_length;
        int num_data;
        int num_psd;

        DataPackage(
            cmplx* data_A,
            cmplx* data_E,
            double* psd_A,
            double* psd_E,
            double df,
            int data_length,
            int num_data,
            int num_psd
        );

};

class BandPackage{
    public:
        int *data_index;
        int *noise_index;
        int *band_start_bin_ind;
        int *band_num_bins;
        int *band_start_data_ind;
        int *band_data_lengths;
        int num_bands;
        int max_data_store_size;

        BandPackage(
            int *data_index,
            int *noise_index,
            int *band_start_bin_ind,
            int *band_num_bins,
            int *band_start_data_ind,
            int *band_data_lengths,
            int num_bands,
            int max_data_store_size
        );
};

class MCMCInfo{
    public:
        cmplx *L_contribution;
        cmplx *p_contribution;
        double *prior_all_curr;
        double *factors_all;
        double *random_val_all;
        bool *accepted_out;
        double *band_inv_temperatures_all;
        bool is_rj;
        double snr_lim;

        MCMCInfo(
            cmplx *L_contribution,
            cmplx *p_contribution,
            double *prior_all_curr,
            double *factors_all,
            double *random_val_all,
            bool *accepted_out,
            double *band_inv_temperatures_all,
            bool is_rj,
            double snr_lim
        );
};

class PriorPackage{
    public:
        double f0_min, f0_max;
        double fdot_min, fdot_max;
        double phi0_min, phi0_max;
        double cosinc_min, cosinc_max;
        double psi_min, psi_max;
        double lam_min, lam_max;
        double sinbeta_min, sinbeta_max;

        PriorPackage(double f0_min, double f0_max, double fdot_min, double fdot_max, double phi0_min, double phi0_max, double cosinc_min, double cosinc_max, double psi_min, double psi_max, double lam_min, double lam_max, double sinbeta_min, double sinbeta_max);
        
        CUDA_HOSTDEV 
        double get_prior_val(
            const double amp, 
            const double f0, 
            const double fdot, 
            const double phi0, 
            const double cosinc,
            const double psi,
            const double lam,
            const double sinbeta
        );
        CUDA_HOSTDEV double get_amp_prior(const double amp);
        CUDA_HOSTDEV double get_f0_prior(const double f0);
        CUDA_HOSTDEV double get_fdot_prior(const double fdot);
        CUDA_HOSTDEV double get_phi0_prior(const double phi0);
        CUDA_HOSTDEV double get_cosinc_prior(const double cosinc);
        CUDA_HOSTDEV double get_psi_prior(const double psi);
        CUDA_HOSTDEV double get_lam_prior(const double lam);
        CUDA_HOSTDEV double get_sinbeta_prior(const double sinbeta);
};

class PeriodicPackage{
    public:
        double phi0_period;
        double psi_period;
        double lam_period;
    
        PeriodicPackage(double phi0_period, double psi_period, double lam_period);
         
        CUDA_HOSTDEV void wrap(double *x, double x_period);
};


class TransformPackage{
    public:
    
        TransformPackage();
         
        CUDA_HOSTDEV void transform(double *amp, double *f0, double *fdot, double *fddot, double *phi0, double *inc, double *psi, double *lam, double *beta);
        CUDA_HOSTDEV void amp_transform(double *amp, const double f0);
        CUDA_HOSTDEV void f0_transform(double *f0);
        CUDA_HOSTDEV void cosinc_transform(double *inc);
        CUDA_HOSTDEV void sinbeta_transform(double *beta);

};


class StretchProposalPackage{
    public:
        double* amp_friends; 
        double* f0_friends; 
        double* fdot0_friends; 
        double* phi0_friends; 
        double* iota_friends;
        double* psi_friends; 
        double* lam_friends;
        double* beta_friends;
        double* z;
        double* factors;
    
        StretchProposalPackage(
            double* amp_friends,
            double* f0_friends, 
            double* fdot0_friends, 
            double* phi0_friends, 
            double* iota_friends,
            double* psi_friends, 
            double* lam_friends,
            double* beta_friends,
            double* z,
            double* factors
        );
         
        CUDA_HOSTDEV void direct_change(double *x_prop, const double x_curr, const double x_friend, const double fraction);
        CUDA_HOSTDEV void wrap_change(double *x_prop, const double x_curr, const double x_friend, const double fraction, const double period, void (*wrap_func)(double *));
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
);

typedef struct InputInfoTag{
    double* amp;
    double* f0;
    double* fdot0;
    double* fddot0;
    double* phi0;
    double* iota;
    double* psi;
    double* lam;
    double* theta;
    double T;
    double dt;
    int N;
    int num_bin_all;
    cmplx *tdi_out;
    cmplx *d_h;
    cmplx *h_h;
    cmplx *data_A; 
    cmplx *data_E;
    double *noise_A;
    double *noise_E;
    int *data_index;
    int *noise_index;
    int start_freq_ind;
    int data_length;
    cmplx* d_h_remove;
    cmplx* d_h_add;
    cmplx* remove_remove;
    cmplx* add_add;
    cmplx* add_remove;
    double* amp_add;
    double* f0_add; 
    double* fdot0_add; 
    double* fddot0_add; 
    double* phi0_add; 
    double* iota_add;
    double* psi_add; 
    double* lam_add;
    double* theta_add;
    double* amp_remove; 
    double* f0_remove; 
    double* fdot0_remove; 
    double* fddot0_remove; 
    double* phi0_remove; 
    double* iota_remove;
    double* psi_remove; 
    double* lam_remove;
    double* theta_remove;
    int device;
    bool do_synchronize;
    double* factors;
    cmplx *L_contribution;
    cmplx *p_contribution;
    GalacticBinaryParams *params_curr;
    GalacticBinaryParams *params_prop;
    double *prior_all_curr;
    double *prior_all_prop;
    double *factors_all;
    double *random_val_all;
    int *band_start_bin_ind;
    int *band_num_bins;
    int *band_start_data_ind;
    int *band_data_lengths;
    double *band_inv_temperatures_all;
    bool *accepted_out;
    int num_bands;
    int max_data_store_size;
    bool is_rj;
    double snr_lim;
    DataPackage *data;
    BandPackage *band_info;
    MCMCInfo *mcmc_info;
    PriorPackage *prior_info;
} InputInfo; 

void SharedMemoryLikeComp(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    int* data_index,
    int* noise_index,
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
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);


void SharedMemorySwapLikeComp(
    cmplx* d_h_remove,
    cmplx* d_h_add,
    cmplx* remove_remove,
    cmplx* add_add,
    cmplx* add_remove,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    int* data_index,
    int* noise_index,
    double* amp_add, 
    double* f0_add, 
    double* fdot0_add, 
    double* fddot0_add, 
    double* phi0_add, 
    double* iota_add,
    double* psi_add, 
    double* lam_add,
    double* theta_add,
    double* amp_remove, 
    double* f0_remove, 
    double* fdot0_remove, 
    double* fddot0_remove, 
    double* phi0_remove, 
    double* iota_remove,
    double* psi_remove, 
    double* lam_remove,
    double* theta_remove,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);


void SharedMemoryGenerateGlobal(
    cmplx* data_A,
    cmplx* data_E,
    int* data_index,
    double* factors,
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
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);

void specialty_piece_wise_likelihoods_wrap(
    double* lnL,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    int* data_index,
    int* noise_index,
    int* start_inds,
    int* lengths,
    double df, 
    int num_parts,
    int start_freq_ind,
    int data_length,
    bool do_synchronize
);

void SharedMemoryMakeMove(
    DataPackage *data,
    BandPackage *band_info,
    GalacticBinaryParams *params_curr,
    GalacticBinaryParams *params_prop,
    MCMCInfo *mcmc_info,
    int device,
    bool do_synchronize
);

void SharedMemoryMakeNewMove(
    DataPackage *data,
    BandPackage *band_info,
    GalacticBinaryParams *params_curr,
    GalacticBinaryParams *params_prop,
    MCMCInfo *mcmc_info,
    PriorPackage *prior_info,
    int device,
    bool do_synchronize
);

void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* A_data, cmplx* E_data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds);

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim);

#endif // __SHAREDMEMORY_GBGPU_HPP__