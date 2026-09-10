"""Sig-het reference builds under the shared-psd MIRROR (2026-09-09).

``GBSignalHetComputations.setup_in_model`` / ``setup_fstat_references``
read the buffer's inverse-covariance Python-side (the ``invC_w`` window
that feeds the bin-fold). In mirror mode the holder's ``linear_psd_arr[0]``
is the parent's per-WALKER full-band plane and the slot's walker row is
``psd_row_index[slot]``; the gathered window must be the SAME bytes the
per-slot copy held, so every stash array and every score must be
``np.array_equal`` between the two layouts. Each parity test also flips
the row map and asserts the stash CHANGES (the fixture's walkers carry
distinct invC), so a comparison that could not fail is ruled out.

This is the plan's ``MirrorSlabTest`` -- placed here on the LIGHT
``psd_mirror_fixture`` (coarse orbit grid, ~0.3 GB) instead of
``test_sighet_inmodel_window._Grid`` (dense orbit grid, ~3 GB), which the
8 GB laptop's memory watchdog cannot host. CPU-only; one class per process.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_mirror_fixture import MirrorSlotHolder, build_fixture  # noqa: E402

from gbgpu.gbsignalhetcomputations import GBSignalHetComputations  # noqa: E402

V5_KNOBS = dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)

_INMODEL_STASH = ("c0_sparse_all", "c0_mask_all", "A0_all", "A1_all",
                  "B0_all", "B1_all", "B0nc_all", "B1nc_all",
                  "params_ref_all")
_FSTAT_STASH = ("A0", "A1", "B0", "B1", "B0nc", "B1nc", "mask", "refs",
                "w_lo")

_FX = {}


def _fixture():
    if "fx" not in _FX:
        _FX["fx"] = build_fixture("cpu")
    return _FX["fx"]


def _windowed(fn, *a, **k):
    saved = os.environ.get("GB_SIGHET_INMODEL_WINDOWED")
    os.environ["GB_SIGHET_INMODEL_WINDOWED"] = "1"
    try:
        return fn(*a, **k)
    finally:
        if saved is None:
            os.environ.pop("GB_SIGHET_INMODEL_WINDOWED", None)
        else:
            os.environ["GB_SIGHET_INMODEL_WINDOWED"] = saved


def _snap(comp, keys, src=None):
    out = {}
    for k in keys:
        v = getattr(comp, k) if src is None else src[k]
        out[k] = np.array(v, copy=True)
    out["_stash_w_lo"] = np.array(comp._stash_w_lo, copy=True) if src is None else None
    return out


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _fixture()

    def _comp(self):
        return _windowed(GBSignalHetComputations.for_band_engine,
                         self.fx["chunked"], **V5_KNOBS)

    def _swapped(self, template):
        fx = self.fx
        rows = np.asarray(fx["psd_rows"])[::-1].copy()
        if getattr(template, "band_slab_Nf", None) is not None:
            return MirrorSlotHolder(fx["xp"], fx["slabs_narrow"], fx["mirror"],
                                    rows, fx["slab_Nf"], fx["slab_min_f"])
        return MirrorSlotHolder(fx["xp"], fx["slabs_full"], fx["mirror"], rows)

    def _assert_stash_equal(self, a, b, msg):
        self.assertEqual(set(a), set(b))
        for k in a:
            if a[k] is None:
                continue
            self.assertEqual(a[k].shape, b[k].shape, k)
            np.testing.assert_array_equal(
                a[k], b[k], err_msg=f"{msg}: stash {k} differs")

    def _stash_differs(self, a, b):
        return any(a[k] is not None and not np.array_equal(a[k], b[k]) for k in a)


class InModelNarrowSlabMirrorTest(_Base):
    """setup_in_model on the production narrow-slab layout."""

    def _build_and_score(self, holder):
        fx = self.fx
        comp = self._comp()
        _windowed(comp.setup_in_model, holder, fx["params_ref"], fx["slots"])
        try:
            stash = _snap(comp, _INMODEL_STASH)
            ll = np.asarray(comp.get_ll(fx["params"], data_index=fx["di"]))
            h_h = np.asarray(comp.last_h_h).copy()
            d_h_im = np.asarray(comp.last_d_h_im).copy()
        finally:
            comp.clear_in_model()
        return stash, ll, h_h, d_h_im

    def test_stash_and_scores_mirror_equal_per_slot(self):
        fx = self.fx
        s_ps, ll_ps, hh_ps, im_ps = self._build_and_score(fx["holders"]["narrow_perslot"])
        s_mi, ll_mi, hh_mi, im_mi = self._build_and_score(fx["holders"]["narrow_mirror"])
        self._assert_stash_equal(s_mi, s_ps, "narrow slab")
        np.testing.assert_array_equal(ll_mi, ll_ps)
        np.testing.assert_array_equal(hh_mi, hh_ps)
        np.testing.assert_array_equal(im_mi, im_ps)
        self.assertGreater(float(np.max(np.abs(hh_ps))), 0.0)
        self.assertTrue(np.all(np.isfinite(ll_ps)))
        # the window really is the slab (narrow branch taken)
        self.assertEqual(int(s_mi["A0_all"].shape[2]), int(fx["slab_Nf"]))

    def test_wrong_row_is_visible(self):
        fx = self.fx
        s_ps, ll_ps, _, _ = self._build_and_score(fx["holders"]["narrow_perslot"])
        s_bad, ll_bad, _, _ = self._build_and_score(
            self._swapped(fx["holders"]["narrow_mirror"]))
        self.assertTrue(self._stash_differs(s_bad, s_ps),
                        "swapping the walker rows left the stash unchanged: "
                        "the test cannot see a wrong row")
        self.assertFalse(np.array_equal(ll_bad, ll_ps))


class InModelFullBandMirrorTest(_Base):
    """setup_in_model on a full-band holder (carrier-centred window branch)."""

    def _build(self, holder):
        fx = self.fx
        comp = self._comp()
        _windowed(comp.setup_in_model, holder, fx["params_ref"], fx["slots"])
        try:
            stash = _snap(comp, _INMODEL_STASH)
            ll = np.asarray(comp.get_ll(fx["params"], data_index=fx["di"]))
        finally:
            comp.clear_in_model()
        return stash, ll

    def test_full_band_holder_mirror_equals_per_slot(self):
        fx = self.fx
        s_ps, ll_ps = self._build(fx["holders"]["full_perslot"])
        s_mi, ll_mi = self._build(fx["holders"]["full_mirror"])
        self._assert_stash_equal(s_mi, s_ps, "full band")
        np.testing.assert_array_equal(ll_mi, ll_ps)
        self.assertGreater(float(np.max(np.abs(ll_ps))), 0.0)
        s_bad, _ = self._build(self._swapped(fx["holders"]["full_mirror"]))
        self.assertTrue(self._stash_differs(s_bad, s_ps))


class FStatReferencesMirrorTest(_Base):
    """setup_fstat_references picks ONE walker's row: mirror vs per-slot."""

    def _refs(self):
        fx = self.fx
        A, C = fx["params_ref"][0], fx["params_ref"][1]
        A2 = A.copy()
        A2[1] += 0.05 * fx["layer_df"]
        return np.stack([A, C, A2])

    def _build(self, holder, noise_index):
        comp = self._comp()
        comp.setup_fstat_references(self._refs(), holder, data_index=1,
                                    noise_index=noise_index)
        return _snap(comp, _FSTAT_STASH, src=comp._fstat)

    def test_fstat_stash_mirror_equals_per_slot(self):
        fx = self.fx
        for ni in (0, 1):
            s_ps = self._build(fx["holders"]["full_perslot"], ni)
            s_mi = self._build(fx["holders"]["full_mirror"], ni)
            self._assert_stash_equal(s_mi, s_ps, f"fstat refs noise_index={ni}")
            self.assertGreater(float(np.max(np.abs(s_ps["B0"]))), 0.0)
        # negative control: the other walker's row
        s_ps = self._build(fx["holders"]["full_perslot"], 1)
        s_bad = self._build(self._swapped(fx["holders"]["full_mirror"]), 1)
        self.assertTrue(self._stash_differs(s_bad, s_ps))


if __name__ == "__main__":
    unittest.main()
