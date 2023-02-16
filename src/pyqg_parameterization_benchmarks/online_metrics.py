import pyqg
import pyqg.diagnostic_tools
import numpy as np
from scipy.stats import wasserstein_distance
from pyqg_parameterization_benchmarks.utils import FeatureExtractor


# TODO: use pythonic variable names
def diagnostic_differences(ds1, ds2, num_ending_timesteps=10):
    """Distance metrics between dataset diagnostics.

    Compute distributional and spectral differences between two
    xarray.Dataset objects representing ensembles of pyqg simulations.
    Variable shape should be (time, lev, y, x) with an optional leading
    "run" dimension.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Dataset of pyqg.QGModel runs
    ds2 : xarray.Datasset
        Dataset of pyqg.QGModel runs
    num_ending_timesteps : int
        When computing distributional differences, marginalize over this many
        timesteps (starting from the end and going backwards). Increasing this
        will smooth out differences but require more computational time. Note
        that this should not be increased until before simulations transition
        to quasi-steady state (exact crossover point depends on simulation
        parameters).


    Returns
    -------
    differences : Dict[str, float]
        Distance metrics between different diagnostics. See paper for details.

    Notes
    -----
    This method is similar to
    https://pyqg.readthedocs.io/en/latest/api.html#pyqg.diagnostic_tools.diagnostic_differences,
    but includes distributional metrics and has been adapted to work with
    xarray.Dataset objects."""

    if "run" not in ds1.dims:
        ds1 = ds1.expand_dims("run")
    if "run" not in ds2.dims:
        ds2 = ds2.expand_dims("run")

    # Define our quantities using our feature extraction DSL. Note that some
    # are off by a factor of 2 which does affect Wasserstein distance, but this
    # isn't important for distributional similarity metrics since we normalize.
    distribution_quantities = dict(
        q="q",
        u="ufull",
        v="vfull",
        KE="add(pow(ufull,2),pow(vfull,2))",  # u^2 + v^2
        Ens="pow(curl(ufull,vfull),2)",  # (u_y - v_x)^2
    )

    spectral_quantities = [
        "KEspec",
        "Ensspec",
        "KEflux",
        "APEflux",
        "APEgenspec",
        "KEfrictionspec",
    ]

    differences = {}

    for label, expr in distribution_quantities.items():
        for z in [0, 1]:
            # Flatten over space and the last T timesteps
            ts = slice(-num_ending_timesteps, None)
            q1 = FeatureExtractor(ds1.isel(lev=z, time=ts))(expr).ravel()
            q2 = FeatureExtractor(ds2.isel(lev=z, time=ts))(expr).ravel()
            # Compute the empirical wasserstein distance
            differences[f"distrib_diff_{label}{z+1}"] = wasserstein_distance(q1, q2)

    # TODO: Does this function belong inside here (too-many-locals as is).
    def twothirds_nyquist(m):
        return m.k[0][np.argwhere(np.array(m.filtr)[0] < 1)[0, 0]]

    # TODO: Does this function belong inside here (too-many-locals as is).
    def spectral_rmse(spec1, spec2):
        # Initialize pyqg models so we can use pyqg's calc_ispec helper
        m1 = pyqg.QGModel(nx=spec1.data.shape[-2], log_level=0)
        m2 = pyqg.QGModel(nx=spec2.data.shape[-2], log_level=0)
        # Compute isotropic spectra
        kr1, ispec1 = pyqg.diagnostic_tools.calc_ispec(m1, spec1.data)
        kr2, ispec2 = pyqg.diagnostic_tools.calc_ispec(m2, spec2.data)
        # Take error over wavenumbers below 2/3 of both models' Nyquist freqs
        kmax = min(twothirds_nyquist(m1), twothirds_nyquist(m2))
        nk = (kr1 < kmax).sum()
        return np.sqrt(np.mean((ispec1[:nk] - ispec2[:nk]) ** 2))

    for spec in ["KEspec", "Ensspec"]:
        for z in [0, 1]:
            spec1 = ds1[spec].isel(lev=z).mean("run")
            spec2 = ds2[spec].isel(lev=z).mean("run")
            differences[f"spectral_diff_{spec}{z+1}"] = spectral_rmse(spec1, spec2)

    for spec in ["KEflux", "APEflux"]:
        spec1 = (ds1[spec] + ds1[f"paramspec_{spec}"]).mean("run")
        spec2 = (ds2[spec] + ds2[f"paramspec_{spec}"]).mean("run")
        differences[f"spectral_diff_{spec}"] = spectral_rmse(spec1, spec2)

    for spec in ["APEgenspec", "KEfrictionspec"]:
        spec1 = ds1[spec].mean("run")
        spec2 = ds2[spec].mean("run")
        differences[f"spectral_diff_{spec}"] = spectral_rmse(spec1, spec2)

    return differences


def diagnostic_similarities(model, target, baseline, **kwargs):
    """Similarity metrics between dataset diagnostics.

    Like `diagnostic_differences`, but returning a dictionary of similarity
    scores between negative infinity and 1 which quantify how much closer the
    diagnostics of a given `model` are to a `target` with respect to a
    `baseline`. Scores approach 1 when the distance between the model and the
    target is small compared to the baseline and are negative when that
    distance is greater.

    Parameters
    ----------
    model : xarray.Dataset
        Dataset representing runs of a pyqg.QGModel to be evaluated for
        similarity with respect to the ``target``. Typically, this is a dataset
        of parameterized low-resolution pyqg.QGModel runs.
    target : xarray.Dataset
        Another dataset of pyqg.QGModel runs; we are attempting to quantify how
        similar ``model`` is to this dataset. Typically, this is a dataset of
        high-resolution pyqg.QGModel runs.
    baseline : xarray.Dataset
        A third dataset of pyqg.QGModel runs; we are quantifying how much more
        similar ``model`` is to ``target`` than ``baseline`` is to ``target``.
        Typically, this is a dataset of unparameterized low-resolution
        pyqg.QGModel runs.

    Returns
    -------
    sims : Dict[str, float]
        Dictionary of similarity scores (between negative infinity and 1) of
        ``model`` to ``target`` (relative to ``baseline``) for each diagnostic.

    """
    d1 = diagnostic_differences(model, target, **kwargs)
    d2 = diagnostic_differences(baseline, target, **kwargs)
    sims = dict((k, 1 - d1[k] / d2[k]) for k in d1.keys())
    return sims
