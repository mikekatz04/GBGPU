import numpy as np
import time

from gbgpu.new_gbgpu import GBGPU

YEAR = 31457280.0

if __name__ == "__main__":

    use_gpu = False
    gb = GBGPU(use_gpu=use_gpu)

    num_bin = 2
    amp = 1e-22
    f0 = 2e-3
    fdot = 1e-14
    fddot = 0.0
    phi0 = 0.1
    iota = 0.2
    psi = 0.3
    lam = 0.4
    beta_sky = 0.5
    e1 = 0.3
    beta1 = 0.5
    A2 = 19.5
    omegabar = 0.0
    e2 = 0.3
    P2 = 0.6
    T2 = 0.0

    amp_in = np.full(num_bin, amp)
    f0_in = np.full(num_bin, f0)
    fdot_in = np.full(num_bin, fdot)
    fddot_in = np.full(num_bin, fddot)
    phi0_in = np.full(num_bin, phi0)
    iota_in = np.full(num_bin, iota)
    psi_in = np.full(num_bin, psi)
    lam_in = np.full(num_bin, lam)
    beta_sky_in = np.full(num_bin, beta_sky)
    e1_in = np.full(num_bin, e1)
    beta1_in = np.full(num_bin, beta1)
    A2_in = np.full(num_bin, A2)
    P2_in = np.full(num_bin, P2)
    omegabar_in = np.full(num_bin, omegabar)
    e2_in = np.full(num_bin, e2)
    T2_in = np.full(num_bin, T2)
    N = int(256)

    modes = np.array([1])

    params = np.array([f0, fdot, beta_sky, lam, amp, iota, psi, phi0])

    try:
        print("\n\n\n\n")
        import FastGB as FB

        Tobs = 4.0 * YEAR
        dt = 15.0

        fastGB = FB.FastGB("Test", dt=dt, Tobs=Tobs, orbit="analytic")
        num = 1
        st = time.perf_counter()
        for i in range(num):
            X, Y, Z = fastGB.onefourier(
                simulator="synthlisa",
                params=params[:8],
                buffer=None,
                T=Tobs,
                dt=dt,
                algorithm="Michele",
                oversample=1,
            )
        et = time.perf_counter()
        print("fastGB time per waveform:", (et - st) / num)

    except:
        pass

    num = 1
    st = time.perf_counter()
    for _ in range(num):
        gb.run_wave(
            amp_in,
            f0_in,
            fdot_in,
            fddot_in,
            phi0_in,
            iota_in,
            psi_in,
            lam_in,
            beta_sky_in,
            e1_in,
            beta1_in,
            A2_in,
            omegabar_in,
            e2_in,
            P2_in,
            T2_in,
            modes=modes,
            N=N,
            dt=dt,
        )
    et = time.perf_counter()

    print(
        "time:",
        (et - st) / num,
        "sec",
        "time per bin:",
        (et - st) / (num * num_bin),
        "sec",
    )

    check = np.load("test_fin_j1.npy")
    breakpoint()
