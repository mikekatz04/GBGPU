"""WDM ``get_swap_ll``: host index arrays must reach the device first.

The WDM engine's bounds-keep mask is built from ``params_*_phys``, so on a
GPU run ``keep`` is a cupy array. Indexing a HOST numpy ``data_index`` with
it makes numpy call ``np.asarray`` on the mask and cupy refuses:

    TypeError: Implicit conversion to a NumPy array is not allowed.

Host indices are the normal case on the multi-shard path -- the router
forwards its shard-local ``intra`` maps, which are built with numpy -- and
``get_swap_ll`` was the one entry point that never moved them to the
device (the FD engine converts in both of its calls; the comps convert
internally, but only after the Python-level masking here). The single
victim was the GB_ORTHO premise check, the only production caller: the
1-yr v8 run (job 466) logged 1,702 skips and produced zero orthogonality
data, the same way the 2026-08-29 v7 run lost its dataset.

This laptop has no GPU, so a numpy-only test cannot fail: with
``xp = numpy`` the mask is host and nothing raises. The device is stood in
for by :class:`_DeviceOnlyArray`, which reproduces exactly the one cupy
behaviour that matters -- H2D is allowed, implicit D2H raises -- so
``test_stub_reproduces_the_cupy_failure`` pins that the fixture can still
fail, and the rest of the suite fails on the pre-fix code and passes after.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gbgpu.gb_likelihood import WDMBandLikelihoodEngine

#: cupy's own message, verbatim -- what production actually saw.
CUPY_MSG = ("Implicit conversion to a NumPy array is not allowed. "
            "Please use .get() to construct a NumPy array explicitly.")


def _unwrap(x):
    if isinstance(x, _DeviceOnlyArray):
        return x._a
    if isinstance(x, tuple):
        return tuple(_unwrap(k) for k in x)
    return x


def _wrap(a):
    return _DeviceOnlyArray(a) if isinstance(a, np.ndarray) else a


class _DeviceOnlyArray:
    """cupy stand-in: computes with numpy, but never converts to host.

    Only the operators the WDM ``get_swap_ll`` body (and the phase-max
    mixin) actually use are implemented -- anything else raising
    AttributeError is a feature, not a gap: it means the function grew a
    host/device path this test does not cover.
    """

    def __init__(self, a):
        self._a = np.asarray(a)          # no copy: in-place scatter is real

    # the guard the whole test hangs on
    def __array__(self, *args, **kwargs):
        raise TypeError(CUPY_MSG)

    def get(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def real(self):
        return _wrap(self._a.real)

    @property
    def imag(self):
        return _wrap(self._a.imag)

    def astype(self, dt):
        return _wrap(self._a.astype(dt))

    def copy(self):
        return _wrap(self._a.copy())

    def any(self):
        return bool(self._a.any())

    def __len__(self):
        return len(self._a)

    def __getitem__(self, k):
        return _wrap(self._a[_unwrap(k)])

    def __setitem__(self, k, v):
        self._a[_unwrap(k)] = _unwrap(v)

    def __add__(self, o):
        return _wrap(self._a + _unwrap(o))

    __radd__ = __add__

    def __sub__(self, o):
        return _wrap(self._a - _unwrap(o))

    def __rsub__(self, o):
        return _wrap(_unwrap(o) - self._a)

    def __mul__(self, o):
        return _wrap(self._a * _unwrap(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return _wrap(self._a / _unwrap(o))

    def __and__(self, o):
        return _wrap(self._a & _unwrap(o))

    def __ge__(self, o):
        return _wrap(self._a >= _unwrap(o))

    def __le__(self, o):
        return _wrap(self._a <= _unwrap(o))

    def __gt__(self, o):
        return _wrap(self._a > _unwrap(o))


class _DeviceXP:
    """Minimal ``cupy`` module stand-in (the comp's ``xp``)."""

    int32 = np.int32
    float64 = np.float64

    @staticmethod
    def asarray(x, dtype=None):
        # H2D is allowed (this is exactly what the fix leans on); the
        # result is device-only from here on.
        return _DeviceOnlyArray(np.asarray(_unwrap(x), dtype=dtype))

    @staticmethod
    def full(n, v, dtype=None):
        return _DeviceOnlyArray(np.full(n, v, dtype=dtype))

    @staticmethod
    def zeros(n, dtype=None):
        return _DeviceOnlyArray(np.zeros(n, dtype=dtype))

    @staticmethod
    def sqrt(x):
        return _wrap(np.sqrt(_unwrap(x)))

    @staticmethod
    def maximum(a, b):
        return _wrap(np.maximum(_unwrap(a), _unwrap(b)))

    @staticmethod
    def arctan2(a, b):
        return _wrap(np.arctan2(_unwrap(a), _unwrap(b)))

    @staticmethod
    def exp(x):
        return _wrap(np.exp(_unwrap(x)))

    @staticmethod
    def conj(x):
        return _wrap(np.conj(_unwrap(x)))

    @staticmethod
    def where(c, a, b):
        return _wrap(np.where(_unwrap(c), _unwrap(a), _unwrap(b)))


class _RecordingSwapComp:
    """Records what ``get_swap_ll_wdm`` was handed, returns fixed products.

    Values are row-index-linear so the engine's swap algebra can be
    checked exactly (nothing here needs a real waveform).
    """

    xp = _DeviceXP

    #: per-kept-row bases for (d_h_add, d_h_remove, aa, rr, ar)
    BASES = (100.0, 10.0, 40.0, 20.0, 5.0)

    def __init__(self):
        self.calls = []

    def get_swap_ll_wdm(self, params_add, params_remove, buffer_aca, *,
                        data_index, noise_index):
        self.calls.append(SimpleNamespace(
            data_index=data_index, noise_index=noise_index,
            n_rows=int(params_add.shape[0])))
        k = int(params_add.shape[0])
        out = tuple(_DeviceOnlyArray(b + np.arange(k, dtype=float))
                    for b in self.BASES)
        # fused quadrature stashes (exercises the fused branch + phase max)
        self.d_h_add_im_out = _DeviceOnlyArray(0.5 + np.arange(k, dtype=float))
        self.add_remove_im_out = _DeviceOnlyArray(
            0.25 + np.arange(k, dtype=float))
        # engine signature: (like_add, like_remove, d_h_a, d_h_r, aa, rr, ar)
        return (None, None) + out


LAYER_DF = 1e-5
IND_MIN_F, IND_MAX_F = 100, 300


def _engine():
    comp = _RecordingSwapComp()
    basis = SimpleNamespace(layer_df=LAYER_DF, ind_min_f=IND_MIN_F,
                            ind_max_f=IND_MAX_F)
    return WDMBandLikelihoodEngine(comp, basis, 3, "XYZ"), comp


def _params(f0s):
    """(n, 9) physical params on the 'device'; only column 1 (f0) matters."""
    p = np.zeros((len(f0s), 9))
    p[:, 0] = 1e-21
    p[:, 1] = np.asarray(f0s, dtype=float)
    return _DeviceOnlyArray(p)


# rows 0, 2 in band; row 1 out via the ADD f0; row 3 out via the REMOVE f0.
F0_ADD = [150 * LAYER_DF, 900 * LAYER_DF, 250 * LAYER_DF, 200 * LAYER_DF]
F0_REM = [160 * LAYER_DF, 210 * LAYER_DF, 260 * LAYER_DF, 10 * LAYER_DF]
KEEP_EXPECTED = np.array([True, False, True, False])

#: the router's shard-local ``intra`` maps: HOST numpy, default int dtype
HOST_DATA_INDEX = np.array([0, 1, 1, 0], dtype=int)
HOST_NOISE_INDEX = np.array([1, 0, 0, 1], dtype=int)

KW = dict(N_vals=None, waveform_kwargs={})


class StubFidelityTest(unittest.TestCase):
    """Without this, a passing suite proves nothing on a CPU-only box."""

    def test_stub_reproduces_the_cupy_failure(self):
        mask = _DeviceOnlyArray(KEEP_EXPECTED)
        with self.assertRaises(TypeError) as cm:
            HOST_DATA_INDEX[mask]
        self.assertIn("Implicit conversion", str(cm.exception))

    def test_host_to_device_is_allowed(self):
        dev = _DeviceXP.asarray(HOST_DATA_INDEX)
        self.assertIsInstance(dev, _DeviceOnlyArray)
        np.testing.assert_array_equal(dev.get(), HOST_DATA_INDEX)


class SwapLLHostIndexTest(unittest.TestCase):
    """The regression: host indices + device mask must not blow up."""

    def _call(self, di, ni, phase_maximize=False):
        eng, comp = _engine()
        res = eng.get_swap_ll(
            None, _params(F0_REM), _params(F0_ADD),
            data_index=di, noise_index=ni,
            phase_maximize=phase_maximize, **KW)
        return eng, comp, res

    def test_host_numpy_indices_do_not_raise(self):
        _, _, res = self._call(HOST_DATA_INDEX, HOST_NOISE_INDEX)
        np.testing.assert_array_equal(res.kept.get(), KEEP_EXPECTED)

        # exact swap algebra on the kept rows, sentinel on the rejected ones
        d_h_a, d_h_r, aa, rr, ar = (
            b + np.arange(KEEP_EXPECTED.sum(), dtype=float)
            for b in _RecordingSwapComp.BASES)
        ll = np.full(4, -1e300)
        ll[KEEP_EXPECTED] = (d_h_a - d_h_r) - 0.5 * (aa - rr) - (ar - rr)
        np.testing.assert_allclose(res.ll_diff.get(), ll, rtol=1e-12)
        np.testing.assert_allclose(
            res.opt_snr_add.get()[KEEP_EXPECTED], np.sqrt(aa), rtol=1e-12)
        np.testing.assert_array_equal(
            res.opt_snr_add.get()[~KEEP_EXPECTED], 0.0)

    def test_indices_reach_the_comp_on_device_as_int32(self):
        _, comp, _ = self._call(HOST_DATA_INDEX, HOST_NOISE_INDEX)
        self.assertEqual(len(comp.calls), 1)
        call = comp.calls[0]
        for name, host in (("data_index", HOST_DATA_INDEX),
                           ("noise_index", HOST_NOISE_INDEX)):
            got = getattr(call, name)
            self.assertIsInstance(got, _DeviceOnlyArray, name)
            self.assertEqual(got.dtype, np.int32, name)
            np.testing.assert_array_equal(got.get(), host[KEEP_EXPECTED], name)

    def test_device_indices_still_work(self):
        """Single-shard callers already pass device arrays; conversion is a
        no-op there (and must not smuggle them back to the host)."""
        _, comp, res = self._call(_DeviceXP.asarray(HOST_DATA_INDEX),
                                  _DeviceXP.asarray(HOST_NOISE_INDEX))
        np.testing.assert_array_equal(res.kept.get(), KEEP_EXPECTED)
        self.assertEqual(comp.calls[0].data_index.dtype, np.int32)

    def test_float_indices_are_cast(self):
        """The int32 cast is part of the contract the comps rely on."""
        _, comp, _ = self._call(HOST_DATA_INDEX.astype(float),
                                HOST_NOISE_INDEX.astype(float))
        self.assertEqual(comp.calls[0].data_index.dtype, np.int32)

    def test_phase_max_route_converts_too(self):
        """The conversion sits AFTER the phase-max early return, which is
        safe only because that path recurses back through this body."""
        _, comp, res = self._call(HOST_DATA_INDEX, HOST_NOISE_INDEX,
                                  phase_maximize=True)
        self.assertEqual(len(comp.calls), 1, "fused stash -> one comp call")
        self.assertEqual(comp.calls[0].data_index.dtype, np.int32)
        np.testing.assert_array_equal(res.kept.get(), KEEP_EXPECTED)
        # phase max only ever raises the kept rows' ll_diff
        self.assertTrue(np.all(res.ll_diff.get()[~KEEP_EXPECTED] == -1e300))
        self.assertTrue(np.all(res.ll_diff.get()[KEEP_EXPECTED] > -1e290))


if __name__ == "__main__":
    unittest.main()
