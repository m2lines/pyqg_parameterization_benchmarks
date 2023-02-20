import numpy as np
import xarray as xr
import pyqg
from functools import cached_property
try:
    import gcm_filters
except ImportError:
    print("Could not import gcm_filters; Operator3 coarsening will not work")

def config_for(model: pyqg.QGModel) -> dict:
    """Return all parameters needed to initialize a new pyqg.QGModel similar to
    an existing ``model``, except for ``nx`` and ``ny``, so that the new model
    can be given a different resolution.

    Parameters
    ----------
    model : pyqg.QGModel
        The existing model whose configuration parameters we wish to extract

    Returns
    -------
    config : Dict[str, Any]
        A dictionary holding the configuration parameters

    """
    config = dict(H1=model.Hi[0])
    for prop in ["L", "W", "dt", "rek", "g", "beta", "delta", "U1", "U2", "rd"]:
        config[prop] = getattr(model, prop)
    return config


class Coarsener:
    """Common code for defining filtering and coarse-graining operators.

    Parameters
    ----------
    high_res_model : pyqg.QGModel
        A quasigeostrophic model.
    low_res_nx : int
        A new resolution for the model. Must be lower than
        ``high_res_model.nx`` and evenly divisible by 2.

    """

    def __init__(self, high_res_model, low_res_nx):
        assert low_res_nx < high_res_model.nx
        assert low_res_nx % 2 == 0
        # TODO: Switch to proper pythonic variable names.
        self.m1 = high_res_model
        self.m1._invert()
        self.m2 = pyqg.QGModel(nx=low_res_nx, **config_for(high_res_model))
        self.m2.q = self.coarsen(self.m1.q)
        self.m2._invert()  # recompute psi, u, and v
        self.m2._calc_derived_fields()

    @property
    def q_forcing_total(self) -> np.ndarray:
        """Return the "total" subgrid forcing of the potential vorticity as the
        difference between PV tendencies when passing high-res and coarsened
        initial conditions through high-res and low-res models, respectively

        Returns
        -------
        numpy.ndarray
            The total difference between high-res and low-res tendencies.

        """
        for model in [self.m1, self.m2]:
            model._invert()
            model._do_advection()
            model._do_friction()
        return self.coarsen(self.m1.dqhdt) - self.to_real(self.m2.dqhdt)

    def to_real(self, var: np.ndarray) -> np.ndarray:
        """Convert variable to real space, if needed.

        Parameters
        ----------
        var: numpy.ndarray
            Real or complex array variable.

        Returns
        -------
        numpy.ndarray
            The array variable converted to real space.

        """
        for m in [self.m1, self.m2]:
            if var.shape == m.qh.shape:
                return m.ifft(var)
        return var

    def to_spec(self, var: np.ndarray) -> np.ndarray:
        """Convert variable to spectral space, if needed.

        Parameters
        ----------
        var: numpy.ndarray
            Real or complex array variable.

        Returns
        -------
        numpy.ndarray
            The array variable converted to spectral space.

        """
        for m in [self.m1, self.m2]:
            if var.shape == m.q.shape:
                return m.fft(var)
        return var

    def subgrid_forcing(self, var_name: str) -> np.ndarray:
        """Compute subgrid forcing of a given `var` (as string).

        Parameters
        ----------
        var_name : str
            A string representing a variable, which must be an attribute of a
            pyqg.QGModel.

        Returns
        -------
        numpy.ndarray
            The subgrid forcing of that variable as the difference between the
            coarsened advected quantity and the advected coarsened quantity.

        """
        q1 = getattr(self.m1, var_name)
        q2 = getattr(self.m2, var_name)
        adv1 = self.coarsen(self.m1._advect(q1))
        adv2 = self.to_real(self.m2._advect(q2))
        return adv1 - adv2

    def subgrid_fluxes(self, var_name : str) -> tuple[np.ndarray, np.ndarray]:
        """Compute subgrid fluxes (wrt. u and v) of a given `var`.

        Parameters
        ----------
        var_name : str
            A string representing a variable, which must be an attribute of a
            pyqg.QGModel.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple of two arrays representing the subgrid fluxes of the
            variable in the x- and y-directions. The divergence of these fluxes
            equals the ``subgrid_forcing``.

        """
        q1 = getattr(self.m1, var_name)
        q2 = getattr(self.m2, var_name)
        u_flux = self.coarsen(self.m1.ufull * q1) - self.m2.ufull * q2
        v_flux = self.coarsen(self.m1.vfull * q1) - self.m2.vfull * q2
        return u_flux, v_flux

    @property
    def length_ratio(self) -> float:
        """Ratio of high-res to low-res grid length.

        Returns
        -------
        length_ratio : float

        """
        return self.m1.nx / self.m2.nx

    def coarsen(self, var) -> np.ndarray:
        """Filter and coarse-grain a variable (as array).

        Parameters
        ----------
        var : numpy.ndarray
            An array representing a spatial field to filter and coarsen.

        """
        raise NotImplementedError()

    @cached_property
    def ds1(self) -> xr.Dataset:
        """xarray representation of the high-res model.

        Returns
        -------
        xarray.Dataset

        """
        return self.m1.to_dataset()


class SpectralCoarsener(Coarsener):
    """Spectral truncation with a configurable filter."""

    def coarsen(self, var: np.ndarray) -> np.ndarray:
        """Filter and coarse-grain a variable by converting to spectral space,
        truncating modes, multiplying remaining modes by a spectral filter, and
        converting back to real space.

        Parameters
        ----------
        var : numpy.ndarray
            An array representing a spatial field to filter and coarsen. Can be
            either in real or spectral space.

        Returns
        -------
        numpy.ndarray
            An array in real-space representing the original variable filtered
            and coarse-grained in spectral space.
        """
        vh = self.to_spec(var)
        nk = self.m2.qh.shape[1] // 2
        trunc = np.hstack((vh[:, :nk, : nk + 1], vh[:, -nk:, : nk + 1]))
        filtered = trunc * self.spectral_filter / self.length_ratio**2
        return self.to_real(filtered)

    @property
    def spectral_filter(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            The spectral filter to multiply modes.
        """
        raise NotImplementedError()


class Operator1(SpectralCoarsener):
    """Spectral truncation with a sharp filter."""

    @property
    def spectral_filter(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            The piecewise sharp filter (1 below cutoff mode, double-exponential
            after) used internally by pyqg to dissipate high-frequency modes
            for stability.
        """
        return self.m2.filtr


class Operator2(SpectralCoarsener):
    """Spectral truncation with a softer Gaussian filter."""

    @property
    def spectral_filter(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            A soft exponential filter that reduces the intensity of all modes,
            but especially those above a cutoff lengthscale relative to the
            low-resolution model's grid size.
        """
        return np.exp(-self.m2.wv**2 * (2 * self.m2.dx) ** 2 / 24)


class Operator3(Coarsener):
    """Diffusion-based filter, then real-space coarsening."""

    def coarsen(self, var: np.ndarray) -> np.ndarray:
        """Filter and coarse-grain a variable by coarsening in real space
        (averaging over regions) after filtering with gcm_filters, which uses a
        diffusion-based method for smoothing out high-frequency variation

        Parameters
        ----------
        var : numpy.ndarray
            An array representing a spatial field to filter and coarsen. Can be
            either in real or spectral space.

        Returns
        -------
        numpy.ndarray
            An array in real-space representing the original variable filtered
            with gcm_filters and coarse-grained in real space.
        """
        f = gcm_filters.Filter(
            dx_min=1,
            filter_scale=self.length_ratio,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR,
        )
        d = self.m1.to_dataset().isel(time=-1)
        q = d.q * 0 + self.to_real(var)  # hackily convert to data array
        r = int(self.length_ratio)
        assert r == self.length_ratio
        return f.apply(q, dims=["y", "x"]).coarsen(y=r, x=r).mean().data


if __name__ == "__main__":
    from scipy.stats import pearsonr

    m1 = pyqg.QGModel(nx=256)

    for _ in range(10000):
        m1._step_forward()

    op1 = Operator1(m1, 64)
    op2 = Operator2(m1, 64)
    op3 = Operator3(m1, 64)

    for op in [op1, op2, op3]:
        q_forcing = op.subgrid_forcing("q")

        uq_flux, vq_flux = op.subgrid_fluxes("q")

        q_forcing2 = op.m2.ifft(
            op.m2.ik * op.m2.fft(uq_flux) + op.m2.il * op.m2.fft(vq_flux)
        )

        corr = pearsonr(q_forcing.ravel(), q_forcing2.ravel())[0]

        print(op.__class__.__name__, corr)

        assert corr > 0.5
