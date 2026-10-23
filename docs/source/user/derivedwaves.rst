Extending GBGPU Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`gbgpu.gbgpu.GBGPUBase` (documented in :doc:`main`) holds the FastGB machinery. A
derived class adds astrophysics through four methods that
:meth:`~gbgpu.gbgpu.GBGPUBase.run_wave` calls by name. :class:`~gbgpu.gbgpu.GBGPUBase`
does not declare them (it has no abstract methods), so a derived class must define all four:

* ``prepare_additional_args(*args)``: converts the extra per-binary parameters given to
  ``run_wave`` after the nine GB ones and returns them as a tuple (``add_args``);
* ``special_get_N(amp, f0, T, *args, oversample=1)``: points per waveform when ``run_wave``
  gets no ``N``;
* ``shift_frequency(fi, xi, *add_args)``: the instantaneous frequency at each spacecraft;
* ``add_to_phasing(arg_phasing, f0, fdot, fddot, xi, *add_args)``: an extra phase in the
  link-response phasing.

:class:`gbgpu.gbgpu.GBGPU` implements them as identities (circular binaries) and is the
template to copy. Only the Python path of ``run_wave`` applies the hooks' physics:
``run_wave`` calls ``prepare_additional_args`` and ``special_get_N`` on both of its paths,
but ``shift_frequency`` and ``add_to_phasing`` run only on the Python path, and the
shared-memory kernels (the likelihoods, ``generate_global_template`` and
``use_c_implementation=True``) take the nine base parameters only. The waveform models that
already extend the base are described below.

Third-body inclusion
--------------------

:class:`gbgpu.thirdbody.GBGPUThirdBody` adds a third body on a long outer orbit around the
inner binary: the outer orbit Doppler-shifts the frequency and adds the integrated phase.
This waveform was first built in
`arXiv:1806.00500 <https://arxiv.org/abs/1806.00500>`_ (Robson et al. 2018). Please cite
it, as well as the FastGB and GBGPU references (the ``citation`` property), if you use
this waveform.

.. automodule:: gbgpu.thirdbody

Third-body waveform
*******************

.. autoclass:: gbgpu.thirdbody.GBGPUThirdBody
    :members:
    :show-inheritance:

Third-body utility functions
****************************

:func:`~gbgpu.thirdbody.third_body_factors` and :func:`~gbgpu.thirdbody.get_T2` convert
physical triple parameters to the ``(A2, varpi, T2)`` inputs. See the warning in
:func:`~gbgpu.thirdbody.third_body_factors`: as written it stops with ``NameError``.

.. autofunction:: gbgpu.thirdbody.third_body_factors

.. autofunction:: gbgpu.thirdbody.get_T2
