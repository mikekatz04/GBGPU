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
        oversample=oversample
    )

    base_string = "run_test_1"
    main_dir = "./"
    population_directory_list = [main_dir + f"Realization_{i}/" for i in range(1, 4)]

    triples_setup_directory = main_dir + "populations_for_search/"
    search_dir = main_dir + "search_info/"
    evidence_dir = main_dir + "evidence_info/"
    pe_dir = main_dir + "pe_info/"
    status_file_base = "status_file"
    
    directory_info = dict(
        base_string=base_string,
        population_directory_list=population_directory_list,
        triples_setup_directory=triples_setup_directory,
        main_dir=main_dir,
        search_dir=search_dir,
        evidence_dir=evidence_dir,
        pe_dir=pe_dir,
        status_file_base=status_file_base
    )

    first_cut_ll_diff_lim = -2.0
    second_cut_ll_diff_lim = -2.0

    cut_info = dict(
        first_cut_ll_diff_lim=first_cut_ll_diff_lim,
        second_cut_ll_diff_lim=second_cut_ll_diff_lim
    )

    verbose = True

    return dict(
        dir_info=directory_info,
        waveform_kwargs=waveform_kwargs,
        verbose=verbose,
        cut_info=cut_info
    )




