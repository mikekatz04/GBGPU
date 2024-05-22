from gbgpu.utils.constants import *


def get_settings(copy_settings_file=False):

    # general settings
    dt = 15.0
    Tobs = 4.0 * YEAR

    N_data = int(Tobs / dt)
    Tobs = N_data * dt
    df = 1.0 / Tobs
    oversample = 4

    waveform_kwargs = dict(
        dt=dt,
        T=Tobs,
        N=None,
        oversample=oversample,
        use_c_implementation=True,
    )

    base_string = "run_test_1_new_CHECK"
    main_dir = "./"
    population_directory_list = [main_dir + f"Realization_{i}/" for i in range(1, 4)]

    triples_setup_directory = main_dir + "populations_for_search/"
    search_dir = main_dir + "search_info/"
    evidence_dir = main_dir + "evidence_info/"
    pe_dir = main_dir + "pe_info/"
    status_file_base = "status_file"
    bad_file = main_dir + base_string + "_bad_file.txt"
    
    directory_info = dict(
        base_string=base_string,
        population_directory_list=population_directory_list,
        triples_setup_directory=triples_setup_directory,
        main_dir=main_dir,
        search_dir=search_dir,
        evidence_dir=evidence_dir,
        pe_dir=pe_dir,
        status_file_base=status_file_base,
        bad_file=bad_file
    )

    first_cut_ll_diff_lim = -2.0
    second_cut_ll_diff_lim = -2.0

    cut_info = dict(
        first_cut_ll_diff_lim=first_cut_ll_diff_lim,
        second_cut_ll_diff_lim=second_cut_ll_diff_lim
    )

    verbose = True

    m3_lims = [0.0, 16.0]
    e2_lims = [0.0, 0.985]
    opt_snr_lims = [0.0, 1e6]

    limits_info = dict(
        m3_lims=m3_lims,
        e2_lims=e2_lims,
        opt_snr_lims=opt_snr_lims,
        chirp_mass_lims=[0.001, 1.05],
    )

    search_settings = dict(
        nwalkers=50,
        ntemps=10,
        ngroups=500,
        data_length=8192,
        convergence_iter_count=25,
        nsteps_per_check=20,
        progress=True,
    )

    evidence_settings = dict(
        nwalkers=40,
        ntemps=200,
        ngroups=10,
        data_length=8192,
        total_steps_for_evidence=100,
        number_old_evidences=6,
        nsteps=10,
        thin_by=20,
        progress=True,
        p_base_to_third=0.5,  # for product space mcmc
    )

    pe_settings = dict(
        nwalkers=100,
        ntemps=10,
        ngroups=500,
        data_length=8192,
        nsteps=200,
        burn=100,
        thin_by=25,
        progress=True,
    )

    sampler_settings = dict(
        search=search_settings,
        evidence=evidence_settings,
        pe=pe_settings
    )

    return dict(
        limits_info=limits_info,
        dir_info=directory_info,
        waveform_kwargs=waveform_kwargs,
        verbose=verbose,
        cut_info=cut_info,
        sampler_settings=sampler_settings
    )




