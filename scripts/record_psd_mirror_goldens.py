"""Record the psd-mirror goldens for ``tests/test_psd_mirror_kernels.py``.

Run this with the wheel built BEFORE the mirror kernel args were added (or
any wheel whose OFF path is known-good) so the rebuilt OFF path
``(invC_Nf=0, invC_row=empty)`` can be pinned to it bit-for-bit
(``test_off_args_are_identity``; plan risk R1 -- recompilation-induced
FP drift is the one thing reading cannot prove).

CPU (laptop)::

    OMP_NUM_THREADS=1 python scripts/record_psd_mirror_goldens.py --backend cpu --pre-mirror-wheel

GPU (cluster, one GPU)::

    python scripts/record_psd_mirror_goldens.py --backend cuda12x --pre-mirror-wheel

writes ``tests/data/psd_mirror_goldens_{cpu,gpu}.npz`` (GPU name is
``gpu`` regardless of the cuda flavour). Every entry point is recorded
on BOTH the narrow-slab and the full-band per-slot holders, at
m_band_half_width 1 and 2.

``--pre-mirror-wheel``: the installed ``cgbgpu`` predates the mirror args
(its ``gb_wdm_het_*`` bindings do not take ``invC_Nf, invC_row``). The
Python engine on the current source tree appends those args
unconditionally (``GBWDMComputations._PSD_MIRROR_KERNELS = True``), so
this flag flips the class constant OFF for the recording: the engine then
appends nothing and the old binding runs its unchanged per-slot path --
exactly the computation the goldens are meant to pin. Omit the flag when
re-recording against a rebuilt wheel (only for a NEW baseline, never to
"refresh" a failing identity test).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

from psd_mirror_fixture import build_fixture, run_all_entry_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pre-mirror-wheel", action="store_true",
                    help="installed cgbgpu predates the mirror kernel args: "
                         "record through the engine with _PSD_MIRROR_KERNELS "
                         "forced False (appends nothing)")
    args = ap.parse_args()
    if args.pre_mirror_wheel:
        from gbgpu.gbcomps import GBWDMComputations
        GBWDMComputations._PSD_MIRROR_KERNELS = False
        print("recording with _PSD_MIRROR_KERNELS=False (pre-mirror wheel)")
    tag = "cpu" if args.backend == "cpu" else "gpu"
    out = args.out or os.path.join(HERE, "..", "tests", "data",
                                   f"psd_mirror_goldens_{tag}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fx = build_fixture(backend=args.backend)
    rec = {}
    for hname in ("narrow_perslot", "full_perslot"):
        for m_half in (1, 2):
            res = run_all_entry_points(fx, fx["holders"][hname], m_half=m_half)
            for k, v in res.items():
                rec[f"{hname}/m{m_half}/{k}"] = v
    # the inputs too, so a mismatch can be diagnosed against the recorded
    # fixture rather than a rebuilt one
    rec["params"] = fx["params"]
    rec["params_remove"] = fx["params_remove"]
    rec["di"] = fx["di"]
    rec["slab_min_f"] = fx["slab_min_f"]
    rec["psd_rows"] = fx["psd_rows"]
    rec["backend"] = np.array(args.backend)
    rec["pre_mirror_wheel"] = np.array(bool(args.pre_mirror_wheel))
    np.savez(out, **rec)
    n_nonzero = sum(int(np.any(v != 0)) for k, v in rec.items()
                    if k.count("/") == 2)
    print(f"wrote {out}: {len(rec)} arrays, {n_nonzero} nonzero outputs")
    for k in sorted(rec):
        if k.endswith("/ll"):
            print(f"  {k}: {rec[k]}")


if __name__ == "__main__":
    main()
