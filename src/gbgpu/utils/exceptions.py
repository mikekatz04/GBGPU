"""Definition of GBGPU package common exceptions.

:class:`GBGPUException` is the root of the classes defined here. ``ExceptionGroup`` is
re-exported (the builtin on Python >= 3.11, the ``exceptiongroup`` backport otherwise).
The module is star-imported into :mod:`gbgpu.cutils` AFTER
``gpubackendtools.exceptions``, so ``gbgpu.cutils.CudaException``,
``gbgpu.cutils.CuPyException`` and ``gbgpu.cutils.MissingDependency`` are the GBGPU
classes below, not the same-named classes of :mod:`gpubackendtools.exceptions` (an
``except`` on one does not catch the other).

Note:
    No GBGPU code raises any of these classes. The configuration, file-manager and
    trajectory families are copies from the FastEMRIWaveforms template, used only by the
    dead :mod:`gbgpu.utils.config` / :mod:`gbgpu.utils.citations` modules or not at all.
"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class GBGPUException(Exception):
    """Base class for GBGPU package exceptions."""

    pass


class CudaException(GBGPUException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(GBGPUException):
    """Base class for CuPy-related exceptions."""

    pass


class MissingDependency(GBGPUException):
    """Exception raised when a required dependency is missing."""

    pass


# TODO check: keep/remove/document -- raised only by the dead utils/citations.py
class InvalidInputFile(GBGPUException):
    """Exception raised when the content of an input file does not match expectations."""


# TODO check: keep/remove/document -- used only by the unimportable utils/config.py
class ConfigurationError(GBGPUException):
    """Exception raised when configuration setup fails."""


# TODO check: keep/remove/document -- used only by the unimportable utils/config.py
class ConfigurationMissing(ConfigurationError):
    """Exception raised when an expected configuration entry is missing."""


# TODO check: keep/remove/document -- used only by the unimportable utils/config.py
class ConfigurationValidationError(ConfigurationError):
    """Exception raised when a configuration entry is invalid."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileManagerException(GBGPUException):
    """Exception raised by the FileManager."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileNotInRegistry(FileManagerException):
    """Exception raised when a requested file is not in file registry."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileNotFoundLocally(FileManagerException):
    """Exception raised when file not found locally but expected to be."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileInvalidChecksum(FileManagerException):
    """Exception raised when file has invalid checksum."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileDownloadException(FileManagerException):
    """Exception raised if file download failed."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileDownloadNotFound(FileDownloadException):
    """Exception raised if file is not found at expected URL."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileDownloadConnectionError(FileDownloadException):
    """Exception raised in case of connection error during download."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileDownloadIntegrityError(FileDownloadException):
    """Exception raised in case of integrity error after download."""


# TODO check: keep/remove/document -- GBGPU has no file manager; never raised (FEW template)
class FileManagerDisabledAccess(FileManagerException):
    """Exception raised when trying to access a file whose tags are disabled"""

    disabled_tag: str
    file_name: str

    def __init__(self, /, *args, disabled_tag: str, file_name: str, **kwargs):
        self.disabled_tag = disabled_tag
        self.file_name = file_name
        super().__init__(*args, **kwargs)


### Trajectory-related exceptions
# TODO check: keep/remove/document -- GBGPU has no trajectories; never raised (FEW template)
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off-grid (except for the lower boundary in p)."""
