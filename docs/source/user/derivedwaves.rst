Extending GBGPU Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~

The base class :class:`gbgpu.gbgpu.InheritGBGPU` can be inherited in order to build other waveforms in the model of FastGB. To do this, there are required methods that need to be added to the new waveform class. They are described in the Abstract Base Class :class:`gbgpu.gbgpu.InheritGBGPU`. After the base class, waveform models that have already been extended beyond the base are described. 


``InheritGBGPU`` base class
-----------------------------

.. autoclass:: gbgpu.gbgpu.InheritGBGPU
    :members:
    :show-inheritance:


Third-body inclusion
-----------------------

:class:`gbgpu.gbgpu.GBGPU` has been extended to include a third-body in long orbit around the inner binary. This waveform was first built in `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_. It was more recently used and adapted into this code base for # TODO: add arxiv. Please cite both papers, as well as the base FastGB papers, if you use this waveform. 

Third-body waveform
******************************

.. autoclass:: gbgpu.thirdbody.GBGPUThirdBody
    :members:
    :show-inheritance:


Third-body utility functions
******************************

.. autofunction:: gbgpu.thirdbody.third_body_factors

.. autofunction:: gbgpu.thirdbody.get_T2

