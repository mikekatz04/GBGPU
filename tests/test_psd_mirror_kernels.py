"""Shared-psd MIRROR kernel args on the chunked-het family (2026-09-09).

The GB ``SubBandBuffer`` used to copy each slot's inverse-covariance slab
out of the parent ACA's per-WALKER psd plane on every unit fill. In mirror
mode the buffer keeps ONE per-device replica of that plane and hands the
kernels a per-slot walker-row map; ``get_ll`` / ``swap_ll`` /
``get_fstat_ll`` then index ``(row, absolute layer)`` instead of
``(slot, slab-local layer)``. Values, dtype and time axis are untouched --
this is deduplication of COPIES -- so the only acceptable outcome is
``np.array_equal`` between the two layouts, on every accumulator the
kernels return, never ``allclose``.

Every parity test here also flips the row map (slot 0 <-> slot 1 walkers)
and asserts the outputs CHANGE: the 2-walker fixture carries DISTINCT invC
per walker precisely so a wrong row is visible, and a comparison that
cannot fail proves nothing.

``test_off_args_are_identity`` pins the REBUILT wheel's OFF path
``(invC_Nf=0, invC_row=empty)`` to goldens recorded with the pre-mirror
wheel (``scripts/record_psd_mirror_goldens.py``): recompilation-induced FP
drift is the one hazard reading the diff cannot rule out (plan risk R1).

CPU-only; run ONE CLASS PER PROCESS on the 8 GB laptop (the fixture is
built once per process and cached at module level).
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_mirror_fixture import (  # noqa: E402
    MirrorSlotHolder, NarrowSlabHolder, build_fixture, run_all_entry_points,
)

#: Backend the fixture is built on: ``PSD_MIRROR_TEST_BACKEND`` (default
#: ``cpu``; ``cuda12x`` on the cluster). The goldens file follows the backend
#: FAMILY -- ``gpu`` for any cuda flavour -- matching what
#: ``scripts/record_psd_mirror_goldens.py --backend <same>`` writes. Cluster
#: recipe: record the GPU goldens with the OLD wheel (``--pre-mirror-wheel``)
#: BEFORE rebuilding GBGPU, rebuild, then
#: ``PSD_MIRROR_TEST_BACKEND=cuda12x python -m unittest tests.test_psd_mirror_kernels``.
TEST_BACKEND = os.environ.get("PSD_MIRROR_TEST_BACKEND", "cpu")
GOLDENS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    f"psd_mirror_goldens_{'cpu' if TEST_BACKEND == 'cpu' else 'gpu'}.npz")

_FX = {}

#: Keys of ``run_all_entry_points`` grouped by kernel entry point.
_GET_LL = ("ll", "d_h", "h_h", "d_h_im")
_SWAP = ("sw_like_add", "sw_like_rem", "sw_dha", "sw_dhr", "sw_aa", "sw_rr",
         "sw_ar", "sw_dha_im", "sw_ar_im")
_FSTAT = ("fs0_N", "fs0_M", "fs1_N", "fs1_M")


def _fixture():
    if "fx" not in _FX:
        _FX["fx"] = build_fixture(TEST_BACKEND)
    return _FX["fx"]


def _assert_equal(a, b, keys, msg):
    for k in keys:
        np.testing.assert_array_equal(
            a[k], b[k], err_msg=f"{msg}: {k} differs (mirror vs per-slot must "
                                "be bit-identical)")


def _assert_some_differ(a, b, keys):
    """The negative control: at least one accumulator must change."""
    return any(not np.array_equal(a[k], b[k]) for k in keys)


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _fixture()

    def _swapped_rows_holder(self, template):
        """Same holder, walker rows of slot 0 and 1 exchanged."""
        fx = self.fx
        rows = np.asarray(fx["psd_rows"])[::-1].copy()
        if getattr(template, "band_slab_Nf", None) is not None:
            return MirrorSlotHolder(fx["xp"], fx["slabs_narrow"], fx["mirror"],
                                    rows, fx["slab_Nf"], fx["slab_min_f"])
        return MirrorSlotHolder(fx["xp"], fx["slabs_full"], fx["mirror"], rows)


class OffPathIdentityTest(_Base):
    """The rebuilt wheel's OFF path == the pre-mirror wheel (goldens)."""

    def test_off_args_are_identity(self):
        self.assertTrue(os.path.exists(GOLDENS),
                        f"goldens missing: {GOLDENS} (record them with the "
                        "PRE-mirror wheel via scripts/record_psd_mirror_goldens.py)")
        g = np.load(GOLDENS)
        # the recorded inputs must be the ones we rebuild here
        np.testing.assert_array_equal(g["params"], self.fx["params"])
        np.testing.assert_array_equal(g["params_remove"], self.fx["params_remove"])
        np.testing.assert_array_equal(g["slab_min_f"], self.fx["slab_min_f"])
        np.testing.assert_array_equal(g["psd_rows"], self.fx["psd_rows"])
        n_checked = 0
        for hname in ("narrow_perslot", "full_perslot"):
            for m_half in (1, 2):
                got = run_all_entry_points(self.fx, self.fx["holders"][hname],
                                           m_half=m_half)
                for k, v in got.items():
                    key = f"{hname}/{m_half and 'm' + str(m_half)}/{k}"
                    self.assertIn(key, g.files, key)
                    np.testing.assert_array_equal(
                        v, g[key],
                        err_msg=f"{key}: rebuilt OFF path != pre-mirror golden "
                                "(recompilation changed the arithmetic)")
                    n_checked += 1
        self.assertGreaterEqual(n_checked, 4 * (4 + 9 + 4 + 1))
        # ...and the goldens are not trivially zero
        self.assertGreater(float(np.max(np.abs(g["narrow_perslot/m1/h_h"]))), 0.0)

    def test_off_args_are_the_off_state(self):
        """A holder without psd_row_index gets (0, empty) -- never a row map."""
        ch = self.fx["chunked"]
        args = ch._psd_kernel_args(self.fx["holders"]["narrow_perslot"])
        self.assertEqual(len(args), 2)
        self.assertEqual(int(args[0]), 0)
        self.assertEqual(int(np.asarray(args[1]).size), 0)
        self.assertEqual(np.asarray(args[1]).dtype, np.int32)


class MirrorParityTest(_Base):
    """Narrow-slab buffers (the production layout), mirror vs per-slot."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fx = cls.fx
        cls.ps = run_all_entry_points(fx, fx["holders"]["narrow_perslot"],
                                      m_half=1, fill=False)
        cls.mi = run_all_entry_points(fx, fx["holders"]["narrow_mirror"],
                                      m_half=1, fill=False)
        # the negative control: rows exchanged -> a wrong walker per slot
        cls.bad = run_all_entry_points(
            fx, cls._swapped_rows_holder(cls, fx["holders"]["narrow_mirror"]),
            m_half=1, fill=False)

    def test_mirror_args_are_armed(self):
        ch = self.fx["chunked"]
        args = ch._psd_kernel_args(self.fx["holders"]["narrow_mirror"])
        self.assertEqual(int(args[0]), int(self.fx["Nf_active"]))
        np.testing.assert_array_equal(np.asarray(args[1]), self.fx["psd_rows"])
        # the fixture's walkers really differ (else a wrong row is invisible)
        self.assertFalse(np.array_equal(self.fx["mirror"][0], self.fx["mirror"][1]))
        self.assertFalse(np.array_equal(self.fx["psd_rows"], np.arange(2)),
                         "row map must not be the identity (slot == row would "
                         "hide a 'row = slot' bug)")

    def test_get_ll_mirror_equals_per_slot(self):
        _assert_equal(self.mi, self.ps, _GET_LL, "get_ll")
        self.assertGreater(float(np.max(np.abs(self.ps["h_h"]))), 0.0)
        self.assertTrue(_assert_some_differ(self.bad, self.ps, _GET_LL),
                        "swapping the walker rows changed nothing in get_ll: "
                        "the test cannot see a wrong row")

    def test_swap_ll_mirror_equals_per_slot(self):
        _assert_equal(self.mi, self.ps, _SWAP, "swap_ll")
        self.assertGreater(float(np.max(np.abs(self.ps["sw_aa"]))), 0.0)
        self.assertTrue(_assert_some_differ(self.bad, self.ps, _SWAP))

    def test_fstat_mirror_equals_per_slot(self):
        _assert_equal(self.mi, self.ps, _FSTAT, "get_fstat_ll (fold 0 and 1)")
        self.assertGreater(float(np.max(np.abs(self.ps["fs0_M"]))), 0.0)
        self.assertGreater(float(np.max(np.abs(self.ps["fs1_M"]))), 0.0)
        self.assertTrue(_assert_some_differ(self.bad, self.ps, _FSTAT))

    def test_fill_global_untouched(self):
        """fill_global never reads invC; its output is layout-independent."""
        fx = self.fx
        a = run_all_entry_points(fx, fx["holders"]["narrow_perslot"], m_half=1)
        b = run_all_entry_points(fx, fx["holders"]["narrow_mirror"], m_half=1)
        np.testing.assert_array_equal(a["fill"], b["fill"])
        self.assertGreater(float(np.max(np.abs(a["fill"]))), 0.0)


class MirrorParityWideBandTest(_Base):
    """m_band_half_width=2: slot 1's band runs off its slab edge on BOTH
    sides of the clamp (fixture guard below), so the data-slab clamp and the
    mirror's full-band addressing are exercised where they differ most."""

    def test_band_really_runs_off_the_slab(self):
        fx = self.fx
        m_car = np.floor(fx["params_ref"][:, 1] / fx["layer_df"]).astype(int)
        lo = fx["slab_min_f"][1]
        hi = lo + fx["slab_Nf"]           # exclusive
        self.assertTrue(m_car[1] + 2 + 1 > hi,
                        "slot 1's +-2 band must run past its slab edge")

    def test_all_entry_points_m2(self):
        fx = self.fx
        ps = run_all_entry_points(fx, fx["holders"]["narrow_perslot"], m_half=2,
                                  fill=False)
        mi = run_all_entry_points(fx, fx["holders"]["narrow_mirror"], m_half=2,
                                  fill=False)
        _assert_equal(mi, ps, _GET_LL + _SWAP + _FSTAT, "m_half=2")
        bad = run_all_entry_points(
            fx, self._swapped_rows_holder(fx["holders"]["narrow_mirror"]),
            m_half=2, fill=False)
        self.assertTrue(_assert_some_differ(bad, ps, _GET_LL))


class FullBandLayoutMirrorTest(_Base):
    """Full-band buffers (slab_min_f=None): origin = ind_min_f path."""

    def test_full_band_layout_mirror(self):
        fx = self.fx
        ps = run_all_entry_points(fx, fx["holders"]["full_perslot"], m_half=1,
                                  fill=False)
        mi = run_all_entry_points(fx, fx["holders"]["full_mirror"], m_half=1,
                                  fill=False)
        _assert_equal(mi, ps, _GET_LL + _SWAP + _FSTAT, "full-band")
        self.assertGreater(float(np.max(np.abs(ps["h_h"]))), 0.0)
        bad = run_all_entry_points(
            fx, self._swapped_rows_holder(fx["holders"]["full_mirror"]),
            m_half=1, fill=False)
        self.assertTrue(_assert_some_differ(bad, ps, _GET_LL + _SWAP + _FSTAT))


class EdgeSlabBoundsTest(_Base):
    """A slab pinned at the TOP of the active band, scoring a source whose
    m-band runs past ``ind_max_f``: both layouts must clip identically (the
    data slab governs which layers score; the mirror row is read at the
    same absolute layers) -- the global-extent bound (plan risk R5)."""

    def _edge_case(self):
        fx = self.fx
        xp, ch, W = fx["xp"], fx["chunked"], fx["slab_Nf"]
        ws = fx["wdm_set"]
        ind_max_f = int(ws.ind_max_f)
        ilo = fx["ind_min_f"]
        Nt_a = fx["Nt_active"]
        # source near the top edge: m_floor = ind_max_f - 1
        E = fx["params_ref"][1].copy()
        E[1] = (ind_max_f - 1 + 0.5) * fx["layer_df"]
        lo_abs = ind_max_f + 1 - W         # the clamp ceiling in _compute_slab_min_f
        # per-slot narrow data slab for E, filled directly in slab layout
        buf = xp.zeros(3 * W * Nt_a)
        ch.fill_global_wdm(xp.asarray(E[None, :]), buf, data_index=xp.zeros(1, dtype=np.int32),
                           m_band_half_width=2, band_slab_Nf=W,
                           slab_min_f=xp.asarray(np.array([lo_abs], dtype=np.int32)))
        slab = np.asarray(buf).reshape(1, 3, W, Nt_a)
        self.assertGreater(float(np.max(np.abs(slab))), 0.0)
        row = int(fx["psd_rows"][1])
        mirror = fx["mirror"]
        invC_slab = mirror[row][:, :, lo_abs - ilo:lo_abs - ilo + W, :][None]
        per_slot = NarrowSlabHolder(xp, slab, invC_slab, W, np.array([lo_abs]))
        mirr = MirrorSlotHolder(xp, slab, mirror, np.array([row]), W, np.array([lo_abs]))
        rng = np.random.default_rng(7)
        params = np.stack([E, E, E])
        params[1, 4] += 0.4
        params[2, 0] *= 1.2
        params[2, 1] += 0.03 * fx["layer_df"] * rng.standard_normal()
        return params, per_slot, mirr, ind_max_f, E

    def test_edge_slab_bounds(self):
        fx = self.fx
        params, per_slot, mirr, ind_max_f, E = self._edge_case()
        m_floor = int(np.floor(E[1] / fx["layer_df"]))
        self.assertGreater(m_floor + 2 + 1, ind_max_f + 1,
                           "the +-2 band must run past the band top")
        sub = dict(fx)
        sub["params"] = params
        sub["params_remove"] = params[:, :] * np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 1])
        sub["di"] = np.zeros(3, dtype=np.int32)
        ps = run_all_entry_points(sub, per_slot, m_half=2, fill=False)
        mi = run_all_entry_points(sub, mirr, m_half=2, fill=False)
        _assert_equal(mi, ps, _GET_LL + _SWAP + _FSTAT, "edge slab")
        self.assertGreater(float(np.max(np.abs(ps["h_h"]))), 0.0)
        # negative control: read the OTHER walker's row
        other = MirrorSlotHolder(fx["xp"], np.asarray(per_slot.linear_data_arr[0]).reshape(1, 3, -1, fx["Nt_active"]),
                                 fx["mirror"], np.array([1 - int(fx["psd_rows"][1])]),
                                 fx["slab_Nf"], np.array([ind_max_f + 1 - fx["slab_Nf"]]))
        bad = run_all_entry_points(sub, other, m_half=2, fill=False)
        self.assertTrue(_assert_some_differ(bad, ps, _GET_LL))


class GuardTest(_Base):
    """Loud failures: no silent per-slot fallback, no OOB launch."""

    def test_bad_origin_raises(self):
        """Under GB_INDEX_ASSERTS a row >= nwalkers or a slab past the band
        top is refused BEFORE the kernel launches."""
        import lisatools.chunked_het as chm
        fx = self.fx
        xp = fx["xp"]
        with mock.patch.object(chm, "_GB_INDEX_ASSERTS", True):
            # row out of range
            bad_rows = MirrorSlotHolder(xp, fx["slabs_narrow"], fx["mirror"],
                                        np.array([fx["nwalkers"], 0]),
                                        fx["slab_Nf"], fx["slab_min_f"])
            with self.assertRaises(AssertionError):
                fx["chunked"]._psd_kernel_args(bad_rows)
            # origin + W past the active band
            top = int(fx["wdm_set"].ind_max_f) + 2 - fx["slab_Nf"]
            bad_org = MirrorSlotHolder(xp, fx["slabs_narrow"], fx["mirror"],
                                       fx["psd_rows"], fx["slab_Nf"],
                                       np.array([fx["slab_min_f"][0], top]))
            with self.assertRaises(AssertionError):
                fx["chunked"]._psd_kernel_args(bad_org)
            # ...and the good holder passes the same checks
            args = fx["chunked"]._psd_kernel_args(fx["holders"]["narrow_mirror"])
            self.assertEqual(int(args[0]), int(fx["Nf_active"]))

    def test_non_mirror_class_refuses_mirror_holder(self):
        from gbgpu.gbcomps import GBWDMComputations

        class _Stale(GBWDMComputations):
            _PSD_MIRROR_KERNELS = False

        stale = _Stale.__new__(_Stale)
        stale.__dict__.update(self.fx["chunked"].__dict__)
        with self.assertRaises(RuntimeError):
            stale._psd_kernel_args(self.fx["holders"]["narrow_mirror"])
        # and it appends NOTHING for a per-slot holder (bbhx contract)
        self.assertEqual(stale._psd_kernel_args(self.fx["holders"]["narrow_perslot"]), ())

    def test_flag_order_is_enforced_at_class_creation(self):
        from lisatools.chunked_het import WDMComputationsBase
        with self.assertRaises(TypeError):
            type("_Bad", (WDMComputationsBase,),
                 dict(_PSD_MIRROR_KERNELS=True, _FUSED_QUAD_KERNELS=False))

    def test_mirror_plane_size_check_in_binding(self):
        """A mirror plane whose size is not a whole number of full-band rows
        is refused by the C++ binding (per-row size keyed off invC_Nf)."""
        fx = self.fx
        xp = fx["xp"]
        ragged = MirrorSlotHolder(xp, fx["slabs_narrow"],
                                  fx["mirror"].ravel()[:-1], fx["psd_rows"],
                                  fx["slab_Nf"], fx["slab_min_f"])
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            fx["chunked"].get_ll_wdm(xp.asarray(fx["params"]), ragged,
                                     data_index=xp.asarray(fx["di"]),
                                     noise_index=xp.asarray(fx["di"]))


if __name__ == "__main__":
    unittest.main()
