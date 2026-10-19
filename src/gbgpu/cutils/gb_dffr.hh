#ifndef __GB_DFFR_HH__
#define __GB_DFFR_HH__

// gb_dffr.hh -- host entry points of the DFFR response of galactic binaries, defined in gb_dffr.cu
// on LAT's DFFR tile (lat_dffr_kernels.hh) with GBDFFRSource (gb_dffr_source.hh). The binding
// exposes them as the gb_dffr_* methods of GBComputationGroupWrap, next to the gb_stft_* methods of
// the existing response, and the consumers are the same: get_ll, fill_global, swap_ll, the
// F-statistic and both gradients. The response reads the kinematics table of the grid instead of
// the orbits, and the TDI generation (1 or 2) instead of a TDIConfig. rows_per_pass of the
// two-template consumers is 32 or 16 on the GPU; the CPU uses one row.

#include "gbt_global.h"  // cmplx
#include "Detector.hpp"  // KinematicsTable
#include "domains.hpp"   // STFTFresnel, STFTDomain

void gb_dffr_get_ll_wrap(
    cmplx *d_h_out, cmplx *h_h_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int *start_freq_inds);

void gb_dffr_fill_global_wrap(
    cmplx *template_fill,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int *start_freq_inds);

void gb_dffr_swap_ll_wrap(
    cmplx *d_h_add_out, cmplx *d_h_remove_out,
    cmplx *add_add_out, cmplx *remove_remove_out, cmplx *add_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int rows_per_pass, int *start_freq_inds);

void gb_dffr_get_fstat_ll_wrap(
    double *N_re_out, double *N_im_out,
    double *M_re_out, double *M_im_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int rows_per_pass, int *start_freq_inds);

void gb_dffr_get_ll_grad_wrap(
    double *grad_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int *start_freq_inds);

void gb_dffr_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    int rows_per_pass, int *start_freq_inds);

// The Fisher matrices (num_bin, num_derivs, num_derivs) with the estimator of
// gb_stft_information_matrix: steps param_eps[inds[d]], four-point stencil or two-point with
// easy_central_difference. rows_per_pass is 4 or 8 on the GPU; the CPU uses one row.
void gb_dffr_information_matrix_wrap(
    double *info_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *noise_index_all, int *inds, double *param_eps,
    int num_bin, int nparams, int num_derivs, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays, double chirp_rate_floor,
    bool easy_central_difference, int rows_per_pass, int *start_freq_inds);

// Diagnostics: the compact template (num_bin, 3, NT, 2 n_side_bins + 1) with the band starts
// (num_bin, NT), and every bin-independent quantity of the segment table.
void gb_dffr_fill_template_wrap(
    cmplx *template_out, int *start_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays,
    double chirp_rate_floor);

void gb_dffr_segment_table_wrap(
    double *effective_frequency, double *chirp_rate, cmplx *reference_phasor,
    cmplx *link_amplitude, cmplx *envelope_slope, cmplx *emission_phasor,
    double *emission_frequency, cmplx *tdi_weights, int *start_freq_ind,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays,
    double chirp_rate_floor);

#endif  // __GB_DFFR_HH__
