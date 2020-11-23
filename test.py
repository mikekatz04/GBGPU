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
    beta = 0.5
    A2 = 1.0
    omegabar = 0.0
    e = 0.1
    n2 = 1.0
    T2 = 1.0

    amp_in = np.full(num_bin, amp)
    f0_in = np.full(num_bin, f0)
    fdot_in = np.full(num_bin, fdot)
    fddot_in = np.full(num_bin, fddot)
    phi0_in = np.full(num_bin, phi0)
    iota_in = np.full(num_bin, iota)
    psi_in = np.full(num_bin, psi)
    lam_in = np.full(num_bin, lam)
    beta_in = np.full(num_bin, beta)
    e_in = np.full(num_bin, e)
    A2_in = np.full(num_bin, A2)
    n2_in = np.full(num_bin, n2)
    omegabar_in = np.full(num_bin, omegabar)
    T2_in = np.full(num_bin, T2)
    N = int(256)

    modes = np.array([2])

    params = np.array([f0, fdot, beta, lam, amp, iota, psi, phi0])

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
            beta_in,
            A2_in,
            omegabar_in,
            e_in,
            n2_in,
            T2_in,
            modes=modes,
            N=N,
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

    breakpoint()
