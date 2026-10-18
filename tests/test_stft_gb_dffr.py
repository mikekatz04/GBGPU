"""Tests of the DFFR response of galactic binaries: the gb_dffr_* entry points of the backend.

The response runs on the consumers of the existing STFT response (lat_stft_kernels.hh) with the
DFFR tile (lat_dffr_kernels.hh) and GBDFFRSource. Until ``STFTGBComputations`` selects it, the tests
call the entry points through ``_DFFRCase``, with the arguments the comp will pass.

* Self-consistency, as ``test_stft_gb.py`` checks the existing response: fill_global then the
  domain inner product equals get_ll; swap(A, A) equals get_ll(A) and relabelling add and remove
  swaps the terms; the F-statistic's N and M equal the get_ll and swap terms; both gradients equal
  host central differences, and eps <= 0 freezes a component.
* The compact template equals the fill_global grid bit for bit, and swap on a union band of several
  bins equals the inner products of compact templates on that band.
* DFFR against the existing response on the same fixture, both TDI generations: a sign, a
  conjugation or a factor 2 in either would show in the overlap or the norm ratio.
* Argument checks, and the CUDA backend against the CPU backend (skipped without a usable GPU).
"""

import unittest

import numpy as np
from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS
from lisatools.domains import get_stft_settings
from lisatools.response.tdiconfig import TDIConfig

try:
    from gbgpu.gbcomps import STFTGBComputations

    HAVE_STFT_GB = True
except (ImportError, ModuleNotFoundError):
    HAVE_STFT_GB = False

# GB params (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta), those of test_stft_gb.py.
GB_PARAMS = np.array(
    [1e-23, 4.2300812341e-3, 1e-18, 0.0, 0.892342, 1.230980, 3.009081, 4.827342, -0.509234]
)
GB_PARAMS_B = np.array(
    [2e-23, 4.2800123000e-3, 2e-18, 0.0, 1.100000, 0.700000, 2.200000, 3.500000, 0.300000]
)
# * 2.6 bins of the six-hour segments above GB_PARAMS: its band overlaps that of GB_PARAMS in part,
# * so swap sums both over a union band two to three bins wider than either.
GB_PARAMS_C = np.array(
    [1.5e-23, 4.2300812341e-3 + 2.6 / 21600.0, -3e-18, 0.0, 0.3, 2.0, 0.4, 1.9, 0.9]
)
GRAD_EPS = np.array([1e-26, 1e-11, 1e-21, 1e-28, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])

#: 24 six-hour segments of 10 s samples, the band 3.9 to 4.6 mHz.
SAMPLE_DT = 10.0
SEGMENT_DT = 6 * 3600.0
NUM_SAMPLES = int(24 * SEGMENT_DT / SAMPLE_DT)
N_SIDE_BINS = 3
CHIRP_RATE_FLOOR = 1.0e-22
# ? Bounds of the comparison with the existing response, set before measuring: a sign or a
# ? conjugation gives an overlap near zero or negative, and the norm ratio catches a factor 2 (a
# ? ratio of 4 or 1/4); the two responses' own approximations stay far inside both.
OVERLAP_BOUND = 0.99
NORM_RATIO_BOUNDS = (0.9, 1.1)
# ? CUDA against CPU, provisional until a first GPU run measures the differences.
GPU_RELATIVE_BOUND = 1e-12


def _host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _settings(backend_name):
    return get_stft_settings(
        np.arange(NUM_SAMPLES) * SAMPLE_DT,
        SEGMENT_DT,
        min_freq=3.9e-3,
        max_freq=4.6e-3,
        force_backend=backend_name,
    )


class _StftCompsShim:
    """Stand-in for STFTComputationGroup: ``cpp_fresnel``, ``cpp_domain`` and ``d_d``."""

    def __init__(self, cpp_fresnel, cpp_domain, d_d=None):
        self.cpp_fresnel = cpp_fresnel
        self.cpp_domain = cpp_domain
        self.d_d = d_d


class _DFFRCase:
    """One fixture on one backend: the comp, its domain and Fresnel shim, the kinematics table of
    the grid, and the gb_dffr_* entry points with the arguments STFTGBComputations will pass."""

    def __init__(
        self,
        backend_name,
        window_alpha=0.0,
        use_midpoint=False,
        tdi="1st generation",
        tdi_type="XYZ",
        tdi_generation=None,
        first_midpoint=None,
        n_side_bins=N_SIDE_BINS,
        linear_envelope=True,
        first_order_delays=True,
    ):
        self.settings = _settings(backend_name)
        self.comp = STFTGBComputations(
            stft_comps=None,
            T=NUM_SAMPLES * SAMPLE_DT,
            t_ref=0.0,
            orbits=None,
            tdi_config=TDIConfig(tdi, force_backend=backend_name),
            force_backend=backend_name,
            n_side_bins=n_side_bins,
        )
        self.xp = self.comp.xp
        self.shim = self._shim(window_alpha, use_midpoint, tdi_type)
        self.comp.stft_comps = self.shim
        self.table = self._table(first_midpoint)
        generation = self.comp.tdi_config.generation if tdi_generation is None else tdi_generation
        self.knobs = (
            self.comp.t_ref,
            n_side_bins,
            generation,
            linear_envelope,
            first_order_delays,
            CHIRP_RATE_FLOOR,
        )
        self.n_side_bins = n_side_bins
        self.wrap = self.comp.backend.GBComputationGroupWrap()

    def _shim(self, window_alpha, use_midpoint, tdi_type):
        """Domain (seeded data, identity inverse covariance) and Fresnel object of the grid."""
        settings, backend, xp = self.settings, self.comp.backend, self.xp
        nt, nf, nch = settings.NT, settings.NF_active, 3
        rng = np.random.default_rng(1234)
        data = rng.standard_normal((1, nch, nt, nf)) + 1j * rng.standard_normal((1, nch, nt, nf))
        if tdi_type == "XYZ":
            inv_c = np.zeros((1, nch, nch, nt, nf), dtype=np.complex128)
            for channel in range(nch):
                inv_c[:, channel, channel] = 1.0
        else:
            inv_c = np.ones((1, nch, nt, nf), dtype=np.complex128)
        data = xp.asarray(np.ascontiguousarray(data.astype(np.complex128)))
        inv_c = xp.asarray(np.ascontiguousarray(inv_c))
        domain = backend.STFTDomainWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            data.reshape(-1),
            inv_c.reshape(-1),
            1,
            1,
            backend.TDITypeDict[tdi_type],
        )
        fresnel = backend.STFTFresnelWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            window_alpha=window_alpha,
            use_midpoint=use_midpoint,
        )
        shim = _StftCompsShim(fresnel, domain)
        shim.keepalive = (data, inv_c)
        shim.data = data
        return shim

    def _table(self, first_midpoint):
        """The kinematics table of the grid from the comp's equal-arm orbits."""
        settings = self.settings
        self.table_arrays = {
            name: self.xp.zeros(int(np.prod(shape)) * settings.NT)
            for name, shape in KINEMATICS_TABLE_FIELDS
        }
        midpoint = settings.t0 + 0.5 * settings.dt if first_midpoint is None else first_midpoint
        table = self.comp.backend.KinematicsTableWrap(
            midpoint, settings.dt, settings.NT, *self.table_arrays.values()
        )
        table.build(self.comp.cpp_orbits)
        return table

    def _grid(self):
        return self.table, self.shim.cpp_fresnel, self.shim.cpp_domain

    def _params(self, params):
        return self.xp.ascontiguousarray(self.xp.asarray(np.atleast_2d(params), dtype=np.float64))

    def _indices(self, num_bin):
        return (self.xp.zeros(num_bin, dtype=np.int32), self.xp.zeros(num_bin, dtype=np.int32))

    def _no_cells(self):
        return self.xp.zeros(0, dtype=np.int32)

    def get_ll(self, params):
        batch = self._params(params)
        num_bin = batch.shape[0]
        d_h = self.xp.zeros(num_bin, dtype=np.complex128)
        h_h = self.xp.zeros(num_bin, dtype=np.complex128)
        self.wrap.gb_dffr_get_ll(
            d_h,
            h_h,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *self.knobs,
            self._no_cells(),
        )
        return _host(d_h), _host(h_h)

    def swap(self, params_add, params_remove, rows_per_pass=32):
        batch_add, batch_remove = self._params(params_add), self._params(params_remove)
        num_bin = batch_add.shape[0]
        terms = [self.xp.zeros(num_bin, dtype=np.complex128) for _ in range(5)]
        self.wrap.gb_dffr_swap_ll(
            *terms,
            *self._grid(),
            batch_add.reshape(-1),
            batch_remove.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *self.knobs,
            rows_per_pass,
            self._no_cells(),
        )
        return [_host(term) for term in terms]

    def fstat(self, params, rows_per_pass=32):
        batch = self._params(params)
        num_bin = batch.shape[0]
        outs = [self.xp.zeros(num_bin * size) for size in (4, 4, 10, 10)]
        self.wrap.gb_dffr_get_fstat_ll(
            *outs,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *self.knobs,
            rows_per_pass,
            self._no_cells(),
        )
        n_re, n_im, m_re, m_im = (_host(out).reshape(num_bin, -1) for out in outs)
        return n_re + 1j * n_im, m_re + 1j * m_im

    def fill_global(self, params):
        """The template grid (1, 3, NT, NF_active) of the sources, all with factor one."""
        settings = self.settings
        batch = self._params(params)
        num_bin = batch.shape[0]
        templates = self.xp.zeros((1, 3, settings.NT, settings.NF_active), dtype=np.complex128)
        self.wrap.gb_dffr_fill_global(
            templates.reshape(-1),
            *self._grid(),
            batch.reshape(-1),
            self.xp.zeros(num_bin, dtype=np.int32),
            self.xp.ones(num_bin),
            num_bin,
            9,
            *self.knobs,
            self._no_cells(),
        )
        return templates

    def get_ll_grad(self, params, eps):
        batch = self._params(params)
        num_bin = batch.shape[0]
        grad = self.xp.zeros(num_bin * 9)
        self.wrap.gb_dffr_get_ll_grad(
            grad,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            self.xp.asarray(eps, dtype=np.float64),
            num_bin,
            9,
            *self.knobs,
            self._no_cells(),
        )
        return _host(grad).reshape(num_bin, 9)

    def swap_ll_grad(self, params_add, params_remove, eps_add, eps_remove, rows_per_pass=32):
        batch_add, batch_remove = self._params(params_add), self._params(params_remove)
        num_bin = batch_add.shape[0]
        grad_add, grad_remove = self.xp.zeros(num_bin * 9), self.xp.zeros(num_bin * 9)
        self.wrap.gb_dffr_swap_ll_grad(
            grad_add,
            grad_remove,
            *self._grid(),
            batch_add.reshape(-1),
            batch_remove.reshape(-1),
            *self._indices(num_bin),
            self.xp.asarray(eps_add, dtype=np.float64),
            self.xp.asarray(eps_remove, dtype=np.float64),
            num_bin,
            9,
            *self.knobs,
            rows_per_pass,
            self._no_cells(),
        )
        return _host(grad_add).reshape(num_bin, 9), _host(grad_remove).reshape(num_bin, 9)

    def fill_template(self, params):
        """Compact templates (num_bin, 3, NT, band) and band starts (num_bin, NT)."""
        batch = self._params(params)
        num_bin = batch.shape[0]
        num_times = self.settings.NT
        band = 2 * self.n_side_bins + 1
        template = self.xp.zeros(num_bin * 3 * num_times * band, dtype=np.complex128)
        starts = self.xp.zeros(num_bin * num_times, dtype=np.int32)
        self.wrap.gb_dffr_fill_template(
            template, starts, *self._grid(), batch.reshape(-1), num_bin, 9, *self.knobs
        )
        return (
            _host(template).reshape(num_bin, 3, num_times, band),
            _host(starts).reshape(num_bin, num_times),
        )

    def domain_inner_product(self, templates):
        """(d|h) and (h|h) of a template grid through the domain's own inner product."""
        settings = self.settings
        d_h = self.xp.zeros(1, dtype=np.complex128)
        h_h = self.xp.zeros(1, dtype=np.complex128)
        self.shim.cpp_domain.compute_likelihood_terms(
            d_h,
            h_h,
            templates.reshape(-1),
            self.xp.asarray([0.0]),
            self.xp.asarray([settings.min_freq]),
            1,
            self.xp.zeros(1, dtype=np.int32),
            self.xp.zeros(1, dtype=np.int32),
            settings.NT,
            settings.NF_active,
            False,
        )
        return _host(d_h), _host(h_h)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRSelfConsistencyTest(unittest.TestCase):
    """The consumer identities of test_stft_gb.py for the DFFR response, CPU backend."""

    backend_name = "cpu"

    def _case(self, **options):
        return _DFFRCase(self.backend_name, **options)

    def test_fill_global_matches_get_ll(self):
        """get_ll equals fill_global followed by the domain inner product."""
        for window_alpha in (0.0, 0.5):
            for tdi in ("1st generation", "2nd generation"):
                with self.subTest(window_alpha=window_alpha, tdi=tdi):
                    case = self._case(window_alpha=window_alpha, tdi=tdi)
                    d_h, h_h = case.get_ll(GB_PARAMS)
                    d_h_fill, h_h_fill = case.domain_inner_product(case.fill_global(GB_PARAMS))
                    self.assertGreater(h_h[0].real, 0.0)
                    np.testing.assert_allclose(d_h_fill, d_h, rtol=1e-10, atol=0)
                    np.testing.assert_allclose(h_h_fill, h_h, rtol=1e-10, atol=0)

    def test_compact_template_equals_fill_global(self):
        """The compact template is the fill_global grid on its band, bit for bit; zero elsewhere."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        settings = case.settings
        grid = _host(case.fill_global(GB_PARAMS))[0]
        compact, starts = (array[0] for array in case.fill_template(GB_PARAMS))
        rebuilt = np.zeros_like(grid)
        for segment in range(settings.NT):
            for band_bin in range(compact.shape[-1]):
                bin_index = starts[segment] + band_bin
                if 0 <= bin_index < settings.NF_active:
                    rebuilt[:, segment, bin_index] = compact[:, segment, band_bin]
        self.assertGreater(np.abs(grid).max(), 0.0)
        np.testing.assert_array_equal(rebuilt, grid)

    def test_swap_degenerate_matches_get_ll(self):
        """swap(A, A): every term equals the (d|h) or (h|h) of get_ll(A)."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        d_h, h_h = case.get_ll(GB_PARAMS)
        d_h_a, d_h_r, aa, rr, ar = case.swap(GB_PARAMS, GB_PARAMS)
        for value, reference in ((d_h_a, d_h), (d_h_r, d_h), (aa, h_h), (rr, h_h), (ar, h_h)):
            np.testing.assert_allclose(value, reference, rtol=1e-10, atol=0)

    def test_swap_add_remove_symmetry(self):
        """swap(B, A) holds the terms of swap(A, B) relabelled, the cross term conjugated."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        ab = case.swap(GB_PARAMS, GB_PARAMS_B)
        ba = case.swap(GB_PARAMS_B, GB_PARAMS)
        np.testing.assert_allclose(ba[0], ab[1], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[1], ab[0], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[2], ab[3], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[3], ab[2], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[4], np.conj(ab[4]), rtol=1e-10, atol=0)
        self.assertGreater(np.abs(ab[4][0]), 0.0)

    def test_swap_union_band_matches_compact_templates(self):
        """swap(A, C) sums both templates over the union of their bands, in every segment."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        wide_side_bins = N_SIDE_BINS + 6
        wide = self._case(window_alpha=0.5, tdi="2nd generation", n_side_bins=wide_side_bins)
        settings = case.settings
        templates, starts = wide.fill_template(np.stack([GB_PARAMS, GB_PARAMS_C]))
        carriers = starts + wide_side_bins
        data = _host(case.shim.data)[0]
        sums = np.zeros(5, dtype=np.complex128)
        for segment in range(settings.NT):
            low = carriers[:, segment].min() - N_SIDE_BINS
            high = carriers[:, segment].max() + N_SIDE_BINS
            for bin_index in range(max(low, 0), min(high, settings.NF_active - 1) + 1):
                add = templates[0, :, segment, bin_index - starts[0, segment]]
                remove = templates[1, :, segment, bin_index - starts[1, segment]]
                data_pixel = data[:, segment, bin_index]
                sums += [
                    np.vdot(data_pixel, add),
                    np.vdot(data_pixel, remove),
                    np.vdot(add, add),
                    np.vdot(remove, remove),
                    np.vdot(add, remove),
                ]
        reference = 4.0 * settings.df * sums
        # * The carriers differ by more than a bin, so the union band is a real one.
        self.assertGreater(np.abs(carriers[0] - carriers[1]).max(), 1)
        values = np.array([term[0] for term in case.swap(GB_PARAMS, GB_PARAMS_C)])
        np.testing.assert_allclose(values, reference, rtol=1e-10, atol=0)

    def test_fstat_matches_get_ll_and_swap(self):
        """N_i = (d|A_i) and M_ij = (A_i|A_j) equal the get_ll and swap terms of the filters."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        n_values, m_values = (array[0] for array in case.fstat(GB_PARAMS))
        basis = []
        for amplitude, iota, psi, phi0 in zip(
            (2.0,) * 4,
            (np.pi / 2,) * 4,
            (0.0, np.pi / 4, 0.0, np.pi / 4),
            (0.0, np.pi, 3 * np.pi / 2, np.pi / 2),
        ):
            filter_params = GB_PARAMS.copy()
            filter_params[[0, 5, 6, 4]] = amplitude, iota, psi, phi0
            basis.append(filter_params)
        n_reference = np.zeros(4, dtype=np.complex128)
        m_reference = np.zeros((4, 4), dtype=np.complex128)
        for first in range(4):
            d_h, h_h = case.get_ll(basis[first])
            n_reference[first], m_reference[first, first] = d_h[0], h_h[0]
            for second in range(first + 1, 4):
                m_reference[first, second] = case.swap(basis[first], basis[second])[4][0]
        np.testing.assert_allclose(n_values, n_reference, rtol=1e-9, atol=0)
        np.testing.assert_allclose(m_values, m_reference[np.triu_indices(4)], rtol=1e-9, atol=0)

    def test_get_ll_grad_matches_host_differences(self):
        """The gradient kernel equals central differences of get_ll; eps <= 0 freezes."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        eps = GRAD_EPS.copy()
        eps[4] = -1.0
        grad = case.get_ll_grad(GB_PARAMS, eps)[0]
        reference = np.zeros(9)
        for parameter in range(9):
            if eps[parameter] <= 0.0:
                continue
            values = []
            for sign in (1.0, -1.0):
                shifted = GB_PARAMS.copy()
                shifted[parameter] += sign * eps[parameter]
                d_h, h_h = case.get_ll(shifted)
                values.append(d_h[0].real - 0.5 * h_h[0].real)
            reference[parameter] = (values[0] - values[1]) / (2.0 * eps[parameter])
        self.assertEqual(grad[4], 0.0)
        self.assertGreater(np.abs(grad).max(), 0.0)
        np.testing.assert_allclose(grad, reference, rtol=1e-6, atol=1e-25)

    def test_swap_ll_grad_matches_host_differences(self):
        """Both swap gradients equal central differences of the swap scalar."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")

        def swap_scalar(params_add, params_remove):
            d_h_a, d_h_r, aa, rr, ar = (term[0] for term in case.swap(params_add, params_remove))
            return (d_h_a - d_h_r - 0.5 * aa - 0.5 * rr + ar).real

        grad_add, grad_remove = (
            grad[0] for grad in case.swap_ll_grad(GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS)
        )
        for grad, side in ((grad_add, 0), (grad_remove, 1)):
            reference = np.zeros(9)
            for parameter in range(9):
                values = []
                for sign in (1.0, -1.0):
                    pair = [GB_PARAMS.copy(), GB_PARAMS_B.copy()]
                    pair[side][parameter] += sign * GRAD_EPS[parameter]
                    values.append(swap_scalar(*pair))
                reference[parameter] = (values[0] - values[1]) / (2.0 * GRAD_EPS[parameter])
            with self.subTest(side=("add", "remove")[side]):
                np.testing.assert_allclose(grad, reference, rtol=1e-6, atol=1e-25)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRAgainstExistingResponseTest(unittest.TestCase):
    """DFFR and the existing response model the same source: their templates nearly agree."""

    def test_templates_overlap(self):
        for tdi in ("1st generation", "2nd generation"):
            with self.subTest(tdi=tdi):
                case = _DFFRCase(
                    "cpu", window_alpha=0.5, use_midpoint=True, tdi=tdi, linear_envelope=False
                )
                settings = case.settings
                dffr = _host(case.fill_global(GB_PARAMS)).ravel()
                column = case.xp.zeros((1, 3, settings.NT, settings.NF_active), dtype=np.complex128)
                case.comp.fill_global_stft(GB_PARAMS, column)
                column = _host(column).ravel()
                overlap = np.vdot(dffr, column).real / np.sqrt(
                    np.vdot(dffr, dffr).real * np.vdot(column, column).real
                )
                norm_ratio = np.vdot(dffr, dffr).real / np.vdot(column, column).real
                self.assertGreater(overlap, OVERLAP_BOUND, msg=f"overlap {overlap:.6f}")
                self.assertTrue(
                    NORM_RATIO_BOUNDS[0] < norm_ratio < NORM_RATIO_BOUNDS[1],
                    msg=f"norm ratio {norm_ratio:.6f}",
                )


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRArgumentTest(unittest.TestCase):
    """Setups DFFR cannot represent raise ValueError instead of returning wrong values."""

    def test_unsupported_setups_raise(self):
        with self.assertRaises(ValueError):
            _DFFRCase("cpu").swap(GB_PARAMS, GB_PARAMS_B, rows_per_pass=24)
        with self.assertRaises(ValueError):
            _DFFRCase("cpu", tdi_generation=3).get_ll(GB_PARAMS)
        # ! A table of another grid would place every segment at the wrong time.
        settings = _settings("cpu")
        with self.assertRaises(ValueError):
            _DFFRCase("cpu", first_midpoint=settings.t0 + 3600.0).get_ll(GB_PARAMS)
        with self.assertRaises(ValueError):
            _DFFRCase("cpu", tdi_type="AET").get_ll(GB_PARAMS)


def _cuda_backend_name():
    """Name of a usable CUDA backend, or None."""
    try:
        import cupy
        import lisatools

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        for name in ("cuda13x", "cuda12x", "cuda11x"):
            if lisatools.has_backend(name):
                cupy.zeros(1)  # * a device that cannot allocate is not usable
                return name
    except Exception:
        return None
    return None


@unittest.skipUnless(HAVE_STFT_GB and _cuda_backend_name(), "requires a usable CUDA backend")
class DFFRGPUTest(DFFRSelfConsistencyTest):
    """The same tests on the CUDA backend, and the CUDA values against the CPU values."""

    backend_name = "gpu"

    def test_gpu_matches_cpu(self):
        """Compact templates and inner products agree with the CPU, and the 16-row and 32-row
        tiles of swap and the F-statistic with each other, within GPU_RELATIVE_BOUND of their
        peaks."""
        gpu = self._case(window_alpha=0.5, tdi="2nd generation")
        cpu = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation")
        pairs = {
            "compact template": (gpu.fill_template(GB_PARAMS)[0], cpu.fill_template(GB_PARAMS)[0]),
            "get_ll": (
                np.concatenate(gpu.get_ll(GB_PARAMS)),
                np.concatenate(cpu.get_ll(GB_PARAMS)),
            ),
            "swap": (gpu.swap(GB_PARAMS, GB_PARAMS_B, 32)[4], cpu.swap(GB_PARAMS, GB_PARAMS_B)[4]),
            "swap rows": (
                gpu.swap(GB_PARAMS, GB_PARAMS_B, 16)[4],
                gpu.swap(GB_PARAMS, GB_PARAMS_B, 32)[4],
            ),
            "fstat rows": (gpu.fstat(GB_PARAMS, 16)[1], gpu.fstat(GB_PARAMS, 32)[1]),
        }
        for name, (first, second) in pairs.items():
            with self.subTest(quantity=name):
                self.assertLessEqual(
                    np.abs(first - second).max(), GPU_RELATIVE_BOUND * np.abs(second).max()
                )


if __name__ == "__main__":
    unittest.main()
