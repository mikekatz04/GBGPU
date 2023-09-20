import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t

from gbgpu.utils.pointeradjust import pointer_adjust, wrapper
from libcpp cimport bool
assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "SharedMemoryGBGPU.hpp":
    ctypedef void* cmplx 'cmplx'
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
    ) except+

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
    ) except+

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
    ) except+


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
    ) except+

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
    ) except+

    #void SharedMemoryMakeMove(
    #    DataPackageWrap *data,
    #    BandPackageWrap *band_info,
    #    GalacticBinaryParamsWrap *params_curr,
    #    GalacticBinaryParamsWrap *params_prop,
    #    MCMCInfoWrap *mcmc_info,
    #    int device,
    #    bool do_synchronize
    # ) except+

    void SharedMemoryMakeNewMove(
        DataPackageWrap *data,
        BandPackageWrap *band_info,
        GalacticBinaryParamsWrap *params_curr,
        MCMCInfoWrap *mcmc_info,
        PriorPackageWrap *prior_info,
        StretchProposalPackageWrap *stretch_info,
        PeriodicPackageWrap *periodic_info,
        int device,
        bool do_synchronize
    ) except +


    void SharedMemoryMakeTemperingMove(
        DataPackageWrap *data,
        BandPackageWrap *band_info,
        GalacticBinaryParamsWrap *params_curr,
        MCMCInfoWrap *mcmc_info,
        PriorPackageWrap *prior_info,
        StretchProposalPackageWrap *stretch_info,
        PeriodicPackageWrap *periodic_info,
        int num_swap_setups,
        int device,
        bool do_synchronize,
        int min_val,
        int max_val
    ) except +

    void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* A_data, cmplx* E_data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds) except+
        
    void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim) except + 

    cdef cppclass GalacticBinaryParamsWrap "GalacticBinaryParams":
        GalacticBinaryParamsWrap(
            double* amp,
            double* f0, 
            double* fdot0, 
            double* phi0, 
            double* cosinc,
            double* psi, 
            double* lam,
            double* sinbeta,
            double* amp_orig,
            double* f0_orig, 
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
        )
    cdef cppclass DataPackageWrap "DataPackage":
        DataPackageWrap(
            cmplx* data_A,
            cmplx* data_E,
            cmplx* base_data_A,
            cmplx* base_data_E,
            double* psd_A,
            double* psd_E,
            double df,
            int data_length,
            int num_data,
            int num_psd,
        )
    
    cdef cppclass BandPackageWrap "BandPackage":
        BandPackageWrap(
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
        )

    cdef cppclass MCMCInfoWrap "MCMCInfo":
        MCMCInfoWrap(
            cmplx *L_contribution,
            cmplx *p_contribution,
            double *prior_all_curr,
            int *accepted_out,
            double *band_inv_temperatures_all,
            bool is_rj,
            double snr_lim
        )
    cdef cppclass PriorPackageWrap "PriorPackage":
        PriorPackageWrap(double rho_star, double f0_min, double f0_max, double fdot_min, double fdot_max, double phi0_min, double phi0_max, double cosinc_min, double cosinc_max, double psi_min, double psi_max, double lam_min, double lam_max, double sinbeta_min, double sinbeta_max);

    cdef cppclass PeriodicPackageWrap "PeriodicPackage":
        PeriodicPackageWrap(double phi0_period, double psi_period, double lam_period);

    cdef cppclass StretchProposalPackageWrap "StretchProposalPackage":
        StretchProposalPackageWrap(
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

    void check_prior_vals_wrap(double* prior_out, PriorPackageWrap *prior_info, GalacticBinaryParamsWrap *gb_params, int num_func);
    void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f) except+

cdef class pyStretchProposalPackage:
    cdef StretchProposalPackageWrap *g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            amp_friends,
            f0_friends, 
            fdot0_friends,
            phi0_friends, 
            iota_friends,
            psi_friends, 
            lam_friends,
            beta_friends,
            nfriends,
            num_friends_init,
            num_proposals,
            a,
            ndim,
            inds,
            factors
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t amp_friends_in = amp_friends
        cdef size_t f0_friends_in = f0_friends
        cdef size_t fdot0_friends_in = fdot0_friends
        cdef size_t phi0_friends_in = phi0_friends
        cdef size_t iota_friends_in = iota_friends
        cdef size_t psi_friends_in = psi_friends
        cdef size_t lam_friends_in = lam_friends
        cdef size_t beta_friends_in = beta_friends
        cdef size_t inds_in = inds
        cdef size_t factors_in = factors
        
        self.g = new StretchProposalPackageWrap(
            <double*> amp_friends_in,
            <double*> f0_friends_in, 
            <double*> fdot0_friends_in, 
            <double*> phi0_friends_in, 
            <double*> iota_friends_in,
            <double*> psi_friends_in, 
            <double*> lam_friends_in,
            <double*> beta_friends_in,
            nfriends,
            num_friends_init,
            num_proposals,
            a,
            ndim,
            <bool *> inds_in,
            <double *> factors_in
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g


cdef class pyPeriodicPackage:
    cdef PeriodicPackageWrap *g

    def __cinit__(self, 
        phi0_period, psi_period, lam_period
    ):
       
        self.g = new PeriodicPackageWrap(
            phi0_period, psi_period, lam_period
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g


cdef class pyPriorPackage:
    cdef PriorPackageWrap *g

    def __cinit__(self, 
        rho_star, f0_min, f0_max, fdot_min, fdot_max, phi0_min, phi0_max, cosinc_min, cosinc_max, psi_min, psi_max, lam_min, lam_max, sinbeta_min, sinbeta_max
    ):
    
        self.g = new PriorPackageWrap(
            rho_star, f0_min, f0_max, fdot_min, fdot_max, phi0_min, phi0_max, cosinc_min, cosinc_max, psi_min, psi_max, lam_min, lam_max, sinbeta_min, sinbeta_max
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g


cdef class pyMCMCInfo:
    cdef MCMCInfoWrap *g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            L_contribution,
            p_contribution,
            prior_all_curr,
            accepted_out,
            band_inv_temperatures_all,
            is_rj,
            snr_lim
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t L_contribution_in = L_contribution
        cdef size_t p_contribution_in = p_contribution
        cdef size_t prior_all_curr_in = prior_all_curr
        cdef size_t accepted_out_in = accepted_out
        cdef size_t band_inv_temperatures_all_in = band_inv_temperatures_all

        self.g = new MCMCInfoWrap(
            <cmplx *>L_contribution_in,
            <cmplx *>p_contribution_in,
            <double *>prior_all_curr_in,
            <int *>accepted_out_in,
            <double *>band_inv_temperatures_all_in,
            is_rj,
            snr_lim
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g



cdef class pyBandPackage:
    cdef BandPackageWrap *g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            loc_index,
            data_index,
            noise_index,
            band_start_bin_ind,
            band_num_bins,
            band_start_data_ind,
            band_data_lengths,
            band_interest_start_data_ind,
            band_interest_data_lengths,
            num_bands,
            max_data_store_size,
            fmin_allow,
            fmax_allow,
            update_data_index,
            ntemps,
            band_ind,
            walker_ind,
            temp_ind,
            swaps_proposed,
            swaps_accepted
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t loc_index_in = loc_index
        cdef size_t data_index_in = data_index
        cdef size_t noise_index_in = noise_index
        cdef size_t band_start_bin_ind_in = band_start_bin_ind
        cdef size_t band_num_bins_in = band_num_bins
        cdef size_t band_start_data_ind_in = band_start_data_ind
        cdef size_t band_data_lengths_in = band_data_lengths
        cdef size_t band_interest_start_data_ind_in = band_interest_start_data_ind
        cdef size_t band_interest_data_lengths_in = band_interest_data_lengths
        cdef size_t fmin_allow_in = fmin_allow
        cdef size_t fmax_allow_in = fmax_allow
        cdef size_t update_data_index_in = update_data_index
        cdef size_t band_ind_in = band_ind
        cdef size_t walker_ind_in = walker_ind
        cdef size_t temp_ind_in = temp_ind
        cdef size_t swaps_proposed_in = swaps_proposed
        cdef size_t swaps_accepted_in = swaps_accepted

        self.g = new BandPackageWrap(
            <int *>loc_index_in,
            <int *>data_index_in,
            <int *>noise_index_in,
            <int *>band_start_bin_ind_in,
            <int *>band_num_bins_in,
            <int *>band_start_data_ind_in,
            <int *>band_data_lengths_in,
            <int *>band_interest_start_data_ind_in,
            <int *>band_interest_data_lengths_in,
            num_bands,
            max_data_store_size,
            <double *>fmin_allow_in,
            <double *>fmax_allow_in,
            <int *>update_data_index_in,
            ntemps,
            <int *>band_ind_in,
            <int *>walker_ind_in,
            <int *>temp_ind_in,
            <int *>swaps_proposed_in,
            <int *>swaps_accepted_in
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g


cdef class pyDataPackage:
    cdef  DataPackageWrap*g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            data_A,
            data_E,
            base_data_A,
            base_data_E,
            psd_A,
            psd_E,
            df,
            data_length,
            num_data,
            num_psd,
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t data_A_in = data_A
        cdef size_t data_E_in = data_E
        cdef size_t base_data_A_in = base_data_A
        cdef size_t base_data_E_in = base_data_E
        cdef size_t psd_A_in = psd_A
        cdef size_t psd_E_in = psd_E

        self.g = new DataPackageWrap(
            <cmplx*> data_A_in,
            <cmplx*> data_E_in,
            <cmplx*> base_data_A_in,
            <cmplx*> base_data_E_in,
            <double*> psd_A_in,
            <double*> psd_E_in,
            df,
            data_length,
            num_data,
            num_psd,
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g



cdef class pyGalacticBinaryParams:
    cdef GalacticBinaryParamsWrap *g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            amp, 
            f0, 
            fdot0, 
            phi0, 
            cosinc, 
            psi, 
            lam, 
            sinbeta,
            amp_orig, 
            f0_orig, 
            fdot0_orig, 
            phi0_orig, 
            cosinc_orig, 
            psi_orig, 
            lam_orig, 
            sinbeta_orig,
            T,
            dt, 
            N,
            num_bin_all,
            start_freq_ind,
            Soms_d,
            Sa_a, 
            Amp,
            alpha,
            sl1,
            kn,
            sl2
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t amp_in = amp
        cdef size_t f0_in = f0
        cdef size_t fdot0_in = fdot0
        cdef size_t phi0_in = phi0
        cdef size_t cosinc_in = cosinc
        cdef size_t psi_in = psi
        cdef size_t lam_in = lam
        cdef size_t sinbeta_in = sinbeta
        
        cdef size_t amp_orig_in = amp_orig
        cdef size_t f0_orig_in = f0_orig
        cdef size_t fdot0_orig_in = fdot0_orig
        cdef size_t phi0_orig_in = phi0_orig
        cdef size_t cosinc_orig_in = cosinc_orig
        cdef size_t psi_orig_in = psi_orig
        cdef size_t lam_orig_in = lam_orig
        cdef size_t sinbeta_orig_in = sinbeta_orig

        self.g = new GalacticBinaryParamsWrap(
            <double *>amp_in, 
            <double *>f0_in, 
            <double *>fdot0_in, 
            <double *>phi0_in, 
            <double *>cosinc_in,
            <double *>psi_in, 
            <double *>lam_in,
            <double *>sinbeta_in,
            <double *>amp_orig_in, 
            <double *>f0_orig_in, 
            <double *>fdot0_orig_in, 
            <double *>phi0_orig_in, 
            <double *>cosinc_orig_in,
            <double *>psi_orig_in, 
            <double *>lam_orig_in,
            <double *>sinbeta_orig_in,
            T, 
            dt,
            N,
            num_bin_all,
            start_freq_ind,
            Soms_d,
            Sa_a, 
            Amp,
            alpha,
            sl1,
            kn,
            sl2
        )

    def g_in(self):
        return <uintptr_t>self.g

    def __dealloc__(self):
        if self.g:
            del self.g


@pointer_adjust
def SharedMemoryWaveComp_wrap(
    tdi_out,
    amp, 
    f0, 
    fdot0, 
    fddot0, 
    phi0, 
    iota, 
    psi, 
    lam, 
    theta,
    T,
    dt, 
    N,
    num_bin_all
):
    cdef size_t tdi_out_in = tdi_out
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta

    SharedMemoryWaveComp(
        <cmplx *> tdi_out_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all
    )


@pointer_adjust
def SharedMemoryLikeComp_wrap(
        d_h,
        h_h,
        data_A,
        data_E,
        noise_A,
        noise_E,
        data_index, 
        noise_index,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta

    SharedMemoryLikeComp(
        <cmplx *> d_h_in,
        <cmplx *> h_h_in,
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <double *> noise_A_in,
        <double *> noise_E_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    )



@pointer_adjust
def SharedMemorySwapLikeComp_wrap(
        d_h_remove,
        d_h_add,
        remove_remove,
        add_add,
        add_remove,
        data_A,
        data_E,
        noise_A,
        noise_E,
        data_index, 
        noise_index,
        amp_add, 
        f0_add, 
        fdot0_add, 
        fddot0_add, 
        phi0_add, 
        iota_add, 
        psi_add, 
        lam_add, 
        theta_add,
        amp_remove, 
        f0_remove, 
        fdot0_remove, 
        fddot0_remove, 
        phi0_remove, 
        iota_remove, 
        psi_remove, 
        lam_remove, 
        theta_remove,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t d_h_remove_in = d_h_remove
    cdef size_t d_h_add_in = d_h_add
    cdef size_t remove_remove_in = remove_remove
    cdef size_t add_add_in = add_add
    cdef size_t add_remove_in = add_remove
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_add_in = amp_add
    cdef size_t f0_add_in = f0_add
    cdef size_t fdot0_add_in = fdot0_add
    cdef size_t fddot0_add_in = fddot0_add
    cdef size_t phi0_add_in = phi0_add
    cdef size_t iota_add_in = iota_add
    cdef size_t psi_add_in = psi_add
    cdef size_t lam_add_in = lam_add
    cdef size_t theta_add_in = theta_add
    cdef size_t amp_remove_in = amp_remove
    cdef size_t f0_remove_in = f0_remove
    cdef size_t fdot0_remove_in = fdot0_remove
    cdef size_t fddot0_remove_in = fddot0_remove
    cdef size_t phi0_remove_in = phi0_remove
    cdef size_t iota_remove_in = iota_remove
    cdef size_t psi_remove_in = psi_remove
    cdef size_t lam_remove_in = lam_remove
    cdef size_t theta_remove_in = theta_remove

    SharedMemorySwapLikeComp(
        <cmplx *> d_h_remove_in,
        <cmplx *> d_h_add_in,
        <cmplx *> remove_remove_in,
        <cmplx *> add_add_in,
        <cmplx *> add_remove_in,
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <double *> noise_A_in,
        <double *> noise_E_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_add_in, 
        <double *>f0_add_in, 
        <double *>fdot0_add_in, 
        <double *>fddot0_add_in, 
        <double *>phi0_add_in, 
        <double *>iota_add_in,
        <double *>psi_add_in, 
        <double *>lam_add_in,
        <double *>theta_add_in,
        <double *>amp_remove_in, 
        <double *>f0_remove_in, 
        <double *>fdot0_remove_in, 
        <double *>fddot0_remove_in, 
        <double *>phi0_remove_in, 
        <double *>iota_remove_in,
        <double *>psi_remove_in, 
        <double *>lam_remove_in,
        <double *>theta_remove_in,
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    )


@pointer_adjust
def SharedMemoryGenerateGlobal_wrap(
        data_A,
        data_E,
        data_index,
        factors,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t data_index_in = data_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t factors_in = factors

    SharedMemoryGenerateGlobal(
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <int*> data_index_in,
        <double *> factors_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    )

@pointer_adjust
def specialty_piece_wise_likelihoods(
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
        data_length,
        do_synchronize
    ):

    cdef size_t lnL_in = lnL
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t start_inds_in = start_inds
    cdef size_t lengths_in = lengths

    specialty_piece_wise_likelihoods_wrap(
        <double*> lnL_in,
        <cmplx*> data_A_in,
        <cmplx*> data_E_in,
        <double*> noise_A_in,
        <double*> noise_E_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <int*> start_inds_in,
        <int*> lengths_in,
        df, 
        num_parts,
        start_freq_ind,
        data_length,
        do_synchronize
    )   


# @pointer_adjust
# def SharedMemoryMakeMove_wrap(
#         data,
#        band_info,
#        params_curr,
#        params_prop,
#        mcmc_info,
#        device,
#        do_synchronize
#    ):

#    cdef size_t params_curr_in = params_curr.g_in()
#    cdef size_t params_prop_in = params_prop.g_in()
#    cdef size_t data_in = data.g_in()
#    cdef size_t band_info_in = band_info.g_in()
#    cdef size_t mcmc_info_in = mcmc_info.g_in()

#    SharedMemoryMakeMove(
#        <DataPackageWrap *>data_in,
#        <BandPackageWrap *>band_info_in,
#        <GalacticBinaryParamsWrap *>params_curr_in,
#        <GalacticBinaryParamsWrap *>params_prop_in,
#        <MCMCInfoWrap *>mcmc_info_in,
#        device,
#        do_synchronize,
#    )


@pointer_adjust
def SharedMemoryMakeNewMove_wrap(
        data,
        band_info,
        params_curr,
        mcmc_info,
        prior_info,
        stretch_info,
        periodic_info,
        device,
        do_synchronize
    ):

    cdef size_t params_curr_in = params_curr.g_in()
    cdef size_t data_in = data.g_in()
    cdef size_t band_info_in = band_info.g_in()
    cdef size_t mcmc_info_in = mcmc_info.g_in()
    cdef size_t prior_info_in = prior_info.g_in()
    cdef size_t stretch_info_in = stretch_info.g_in()
    cdef size_t periodic_info_in = periodic_info.g_in()

    SharedMemoryMakeNewMove(
        <DataPackageWrap *>data_in,
        <BandPackageWrap *>band_info_in,
        <GalacticBinaryParamsWrap *>params_curr_in,
        <MCMCInfoWrap *>mcmc_info_in,
        <PriorPackageWrap *>prior_info_in,
        <StretchProposalPackageWrap *> stretch_info_in,
        <PeriodicPackageWrap *> periodic_info_in,
        device,
        do_synchronize,
    )

@pointer_adjust
def SharedMemoryMakeTemperingMove_wrap(
        data,
        band_info,
        params_curr,
        mcmc_info,
        prior_info,
        stretch_info,
        periodic_info,
        num_swap_setups,
        device,
        do_synchronize,
        min_val,
        max_val
    ):

    cdef size_t params_curr_in = params_curr.g_in()
    cdef size_t data_in = data.g_in()
    cdef size_t band_info_in = band_info.g_in()
    cdef size_t mcmc_info_in = mcmc_info.g_in()
    cdef size_t prior_info_in = prior_info.g_in()
    cdef size_t stretch_info_in = stretch_info.g_in()
    cdef size_t periodic_info_in = periodic_info.g_in()

    SharedMemoryMakeTemperingMove(
        <DataPackageWrap *>data_in,
        <BandPackageWrap *>band_info_in,
        <GalacticBinaryParamsWrap *>params_curr_in,
        <MCMCInfoWrap *>mcmc_info_in,
        <PriorPackageWrap *>prior_info_in,
        <StretchProposalPackageWrap *> stretch_info_in,
        <PeriodicPackageWrap *> periodic_info_in,
        num_swap_setups,
        device,
        do_synchronize,
        min_val,
        max_val
    )

@pointer_adjust
def check_prior_vals(prior_vals, prior_info, gb_params, num_func):
    cdef size_t prior_vals_in = prior_vals
    cdef size_t gb_params_in = gb_params.g_in()
    cdef size_t prior_info_in = prior_info.g_in()

    check_prior_vals_wrap(<double*> prior_vals_in, <PriorPackageWrap *>prior_info_in, <GalacticBinaryParamsWrap *>gb_params_in, num_func)


@pointer_adjust
def psd_likelihood(like_contrib_final, f_arr, A_data, E_data, data_index_all,  A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, 
                     Amp_all,  alpha_all,  sl1_all,  kn_all, sl2_all, df, data_length, num_data, num_psds):

    cdef size_t like_contrib_final_in = like_contrib_final
    cdef size_t f_arr_in = f_arr
    cdef size_t A_data_in = A_data
    cdef size_t E_data_in = E_data
    cdef size_t data_index_all_in = data_index_all
    cdef size_t A_Soms_d_in_all_in = A_Soms_d_in_all
    cdef size_t A_Sa_a_in_all_in = A_Sa_a_in_all
    cdef size_t E_Soms_d_in_all_in = E_Soms_d_in_all
    cdef size_t E_Sa_a_in_all_in = E_Sa_a_in_all
    cdef size_t Amp_all_in = Amp_all
    cdef size_t alpha_all_in = alpha_all
    cdef size_t sl1_all_in = sl1_all
    cdef size_t kn_all_in = kn_all
    cdef size_t sl2_all_in = sl2_all

    psd_likelihood_wrap(<double*> like_contrib_final_in, <double*> f_arr_in, <cmplx*> A_data_in, <cmplx*> E_data_in, <int*> data_index_all_in, <double*> A_Soms_d_in_all_in, <double*> A_Sa_a_in_all_in, <double*> E_Soms_d_in_all_in, <double*> E_Sa_a_in_all_in, 
                    <double*> Amp_all_in, <double*> alpha_all_in, <double*> sl1_all_in, <double*> kn_all_in, <double*> sl2_all_in, df, data_length, num_data, num_psds)

@pointer_adjust
def compute_logpdf(logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js, 
                    num_points, start_index, num_components, ndim):

    cdef size_t logpdf_out_in = logpdf_out
    cdef size_t component_index_in = component_index
    cdef size_t points_in = points
    cdef size_t weights_in = weights
    cdef size_t mins_in = mins
    cdef size_t maxs_in = maxs
    cdef size_t means_in = means
    cdef size_t invcovs_in = invcovs
    cdef size_t dets_in = dets
    cdef size_t log_Js_in = log_Js
    cdef size_t start_index_in = start_index

    compute_logpdf_wrap(<double*>logpdf_out_in, <int *>component_index_in, <double*>points_in,
                    <double *>weights_in, <double*>mins_in, <double*>maxs_in, <double*>means_in, <double*>invcovs_in, <double*>dets_in, <double *>log_Js_in, 
                    num_points, <int *>start_index_in, num_components, ndim)


@pointer_adjust
def get_psd_val(Sn_A_out, Sn_E_out, f_arr, noise_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, num_f):
    
    cdef size_t Sn_A_out_in = Sn_A_out
    cdef size_t Sn_E_out_in = Sn_E_out
    cdef size_t f_arr_in = f_arr
    cdef size_t noise_index_all_in = noise_index_all
    cdef size_t A_Soms_d_in_all_in = A_Soms_d_in_all
    cdef size_t A_Sa_a_in_all_in = A_Sa_a_in_all
    cdef size_t E_Soms_d_in_all_in = E_Soms_d_in_all
    cdef size_t E_Sa_a_in_all_in = E_Sa_a_in_all
    cdef size_t Amp_all_in = Amp_all
    cdef size_t alpha_all_in = alpha_all
    cdef size_t sl1_all_in = sl1_all
    cdef size_t kn_all_in = kn_all
    cdef size_t sl2_all_in = sl2_all

    get_psd_val_wrap(<double *>Sn_A_out_in, <double *>Sn_E_out_in, <double *>f_arr_in, <int *>noise_index_all_in, <double *>A_Soms_d_in_all_in, <double *>A_Sa_a_in_all_in, <double *>E_Soms_d_in_all_in, <double *>E_Sa_a_in_all_in,
                               <double *>Amp_all_in, <double *>alpha_all_in, <double *>sl1_all_in, <double *>kn_all_in, <double *>sl2_all_in, num_f)