#ifndef __SHAREDMEMORY_GBGPU_HPP__
#define __SHAREDMEMORY_GBGPU_HPP__

#include "global.h"

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
        double *prior_all_prop;
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
            double *prior_all_prop,
            double *factors_all,
            double *random_val_all,
            bool *accepted_out,
            double *band_inv_temperatures_all,
            bool is_rj,
            double snr_lim
        );
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
    int device,
    bool do_synchronize
);

void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* A_data, cmplx* E_data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds);

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim);

#endif // __SHAREDMEMORY_GBGPU_HPP__