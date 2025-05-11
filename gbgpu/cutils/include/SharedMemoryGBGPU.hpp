#ifndef __SHAREDMEMORY_GBGPU_HPP__
#define __SHAREDMEMORY_GBGPU_HPP__

#include "global.h"

#ifdef __CUDACC__
#include <curand_kernel.h>

#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CURANDSTATE curandState
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CURANDSTATE void*
#endif

class SingleGalacticBinary{
    public:
        double snr; 
        double f0_ms; 
        double fdot; 
        double fddot; 
        double phi0; 
        double cosinc;
        double psi; 
        double lam;
        double sinbeta;
        double f0;
        double inc;
        double theta;
        double amp;
        double T; 
        double dt;
        int N;
        int num_bin_all;
        int start_freq_ind;
        double Soms_d;
        double Sa_a;
        double Amp;
        double alpha;
        double sl1;
        double kn;
        double sl2;

    CUDA_DEV SingleGalacticBinary(const int N_, const double Tobs_, const double Soms_d_, const double Sa_a_, const double Amp_, const double alpha_, const double sl1_, const double kn_, const double sl2_);
    CUDA_DEV void transform();
    CUDA_DEV double amp_transform();
    CUDA_DEV double f0_transform();
    CUDA_DEV double cosinc_transform();
    CUDA_DEV double sinbeta_transform();
};

class GalacticBinaryParams{
    public:
        double* snr; 
        double* f0_ms; 
        double* fdot0; 
        double* phi0; 
        double* cosinc;
        double* psi; 
        double* lam;
        double* sinbeta;
        double* snr_orig; 
        double* f0_ms_orig; 
        double* fdot0_orig; 
        double* phi0_orig; 
        double* cosinc_orig;
        double* psi_orig; 
        double* lam_orig;
        double* sinbeta_orig;
        double T; 
        double dt;
        int N;
        int num_bin_all;
        int start_freq_ind;
        double Soms_d;
        double Sa_a;
        double Amp;
        double alpha;
        double sl1;
        double kn;
        double sl2;

        CUDA_HOSTDEV
        GalacticBinaryParams(
            double* snr,
            double* f0_ms, 
            double* fdot0, 
            double* phi0, 
            double* cosinc,
            double* psi, 
            double* lam,
            double* sinbeta,
            double* snr_orig,
            double* f0_ms_orig, 
            double* fdot0_orig, 
            double* phi0_orig, 
            double* cosinc_orig,
            double* psi_orig, 
            double* lam_orig,
            double* sinbeta_orig,
            double T, 
            double dt,
            int N,
            int num_bin_all,
            int start_freq_ind,
            double Soms_d,
            double Sa_a, 
            double Amp,
            double alpha,
            double sl1,
            double kn,
            double sl2
        );

};

class SingleBand{
    public:
        int loc_index;
        int data_index;
        int noise_index;
        int band_start_bin_ind;
        int band_num_bins;
        int band_start_data_ind;
        int band_data_lengths;
        int band_interest_start_data_ind;
        int band_interest_data_lengths;
        int max_data_store_size;
        double fmin_allow;
        double fmax_allow;
        int update_data_index;
        double start_like;
        double current_like;
        double swapped_like;
        double inv_temp;
        int band_ind;
        int walker_ind;
        int temp_ind;
        GalacticBinaryParams gb_params;

        // CUDA_HOSTDEV SingleBand();
        CUDA_HOSTDEV void setup(
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
            double inv_temp,
            int band_ind,
            int walker_ind,
            int temp_ind,
            GalacticBinaryParams *gb_params_all
        );
};

class DataPackage{
    public:
        cmplx* data_A; 
        cmplx* data_E; 
        cmplx* base_data_A; 
        cmplx* base_data_E; 
        double* psd_A; 
        double* psd_E; 
        double df; 
        int data_length;
        int num_data;
        int num_psd;

        DataPackage(
            cmplx* data_A,
            cmplx* data_E,
            cmplx* base_data_A,
            cmplx* base_data_E,
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
        int *loc_index;
        int *data_index;
        int *noise_index;
        int *band_start_bin_ind;
        int *band_num_bins;
        int *band_start_data_ind;
        int *band_data_lengths;
        int *band_interest_start_data_ind;
        int *band_interest_data_lengths;
        int num_bands;
        int max_data_store_size;
        double *fmin_allow;
        double *fmax_allow;
        int *update_data_index;
        int ntemps;
        int *band_ind;
        int *walker_ind;
        int *temp_ind;
        int *swaps_proposed;
        int *swaps_accepted;

        BandPackage(
            int *loc_index,
            int *data_index,
            int *noise_index,
            int *band_start_bin_ind,
            int *band_num_bins,
            int *band_start_data_ind,
            int *band_data_lengths,
            int *band_interest_start_data_ind,
            int *band_interest_data_lengths,
            int num_bands,
            int max_data_store_size,
            double *fmin_allow,
            double *fmax_allow,
            int *update_data_index,
            int ntemps,
            int *band_ind,
            int *walker_ind,
            int *temp_ind,
            int *swaps_proposed,
            int *swaps_accepted
        );
};

class MCMCInfo{
    public:
        cmplx *L_contribution;
        cmplx *p_contribution;
        double *prior_all_curr;
        int *accepted_out;
        double *band_inv_temperatures_all;
        bool is_rj;
        double snr_lim;

        MCMCInfo(
            cmplx *L_contribution,
            cmplx *p_contribution,
            double *prior_all_curr,
            int *accepted_out,
            double *band_inv_temperatures_all,
            bool is_rj,
            double snr_lim
        );
};

class PriorPackage{
    public:
        double rho_star;
        double f0_min, f0_max;
        double fdot_min, fdot_max;
        double phi0_min, phi0_max;
        double cosinc_min, cosinc_max;
        double psi_min, psi_max;
        double lam_min, lam_max;
        double sinbeta_min, sinbeta_max;

        PriorPackage(double rho_star, double f0_min, double f0_max, double fdot_min, double fdot_max, double phi0_min, double phi0_max, double cosinc_min, double cosinc_max, double psi_min, double psi_max, double lam_min, double lam_max, double sinbeta_min, double sinbeta_max);
        
        CUDA_HOSTDEV 
        double get_prior_val(
            const SingleGalacticBinary gb_in, int num_func
        );
        CUDA_HOSTDEV double get_snr_prior(const double snr);
        CUDA_HOSTDEV double get_f0_prior(const double f0);
        CUDA_HOSTDEV double get_fdot_prior(const double fdot);
        CUDA_HOSTDEV double get_phi0_prior(const double phi0);
        CUDA_HOSTDEV double get_cosinc_prior(const double cosinc);
        CUDA_HOSTDEV double get_psi_prior(const double psi);
        CUDA_HOSTDEV double get_lam_prior(const double lam);
        CUDA_HOSTDEV double get_sinbeta_prior(const double sinbeta);
        CUDA_HOSTDEV double uniform_dist_logpdf(const double x, const double x_min, const double x_max);
};

class PeriodicPackage{
    public:
        double phi0_period;
        double psi_period;
        double lam_period;
    
        PeriodicPackage(double phi0_period, double psi_period, double lam_period);
};


class StretchProposalPackage{
    public:
        double* snr_friends; 
        double* f0_friends; 
        double* fdot0_friends; 
        double* phi0_friends; 
        double* cosinc_friends;
        double* psi_friends; 
        double* lam_friends;
        double* sinbeta_friends;
        int nfriends;
        int num_friends_init;
        int num_proposals;
        int ndim;
        double a;
        CURANDSTATE *curand_states;
        bool *inds;
        double *factors;
    
        StretchProposalPackage(
            double* amp_friends,
            double* f0_friends, 
            double* fdot0_friends, 
            double* phi0_friends, 
            double* iota_friends,
            double* psi_friends, 
            double* lam_friends,
            double* beta_friends,
            int nfriends,
            int num_friends_init,
            int num_proposals,
            double a,
            int ndim,
            bool *inds,
            double *factors
        );
         
        void dealloc();
        void find_friends(SingleGalacticBinary *gb_out, double f_val_in, CURANDSTATE localState);
        CUDA_DEV void get_proposal(SingleGalacticBinary *gb_prop, double *factors, CURANDSTATE localState, const SingleGalacticBinary gb_in, const PeriodicPackage periodic_info);
        CUDA_HOSTDEV void direct_change(double *x_prop, const double x_curr, const double x_friend, const double fraction);
        CUDA_HOSTDEV void wrap_change(double *x_prop, const double x_curr, const double x_friend, const double fraction, const double period);
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
    double *prior_all_curr;
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
    StretchProposalPackage *stretch_info;
    PeriodicPackage *periodic_info;
    int num_swap_setups;
    int min_val;
    int max_val;
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
    MCMCInfo *mcmc_info,
    int device,
    bool do_synchronize
);

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
);

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
);

void check_prior_vals_wrap(double* prior_out, PriorPackage *prior_info, GalacticBinaryParams *gb_params, int num_func);

void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* A_data, cmplx* E_data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds);

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim);

#endif // __SHAREDMEMORY_GBGPU_HPP__