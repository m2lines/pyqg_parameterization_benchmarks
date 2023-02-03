import numpy as np
import pyqg
from functools import cached_property

# TODO: Add type-hinting and proper variable name for ``m``.
# TODO: Complete docstring.
def config_for(m):
    """Return parameters needed to initialize new pyqg.QGModel.

    Does not return the ``nx`` and ``ny`` parameters. What are these
    parameters?

    Parameters
    ----------
    m : ???

    Returns
    -------
    config : Dict[str, ???]
        A dictionary holding the parameters???

    """
    # """Return the parameters needed to initialize a new
    # pyqg.QGModel, except for nx and ny."""
    config = dict(H1=m.Hi[0])
    for prop in ["L", "W", "dt", "rek", "g", "beta", "delta", "U1", "U2", "rd"]:
        config[prop] = getattr(m, prop)
    return config


class Coarsener:
    """Common code for defining filtering and coarse-graining operators.

    Parameters
    ----------
    high_res_model : ???
        Description of ``high_res_model``.
    low_res_nx : ???
        Description of ``low_res_nx``.

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

    # TODO: Add a docstring.
    # TODO: Use pythonic variable names.
    @property
    def q_forcing_total(self):
        """Add a docstring."""
        for m in [self.m1, self.m2]:
            m._invert()
            m._do_advection()
            m._do_friction()
        return self.coarsen(self.m1.dqhdt) - self.to_real(self.m2.dqhdt)

    # TODO: Update docstring and type-hints.
    def to_real(self, var):
        """Convert variable to real space, if needed.

        Parameters
        ----------
        var: ???
            What is ``var``?

        Returns
        -------
        float
            Is this surely a float?

        """
        for m in [self.m1, self.m2]:
            if var.shape == m.qh.shape:
                return m.ifft(var)
        return var

    # TODO: Update docstring and type-hints.
    def to_spec(self, var):
        """Convert variable to spectral space, if needed.

        Parameters
        ----------
        var : ???
            Add description.

        Returns
        -------
        float : ???
            Is this a float?

        """
        for m in [self.m1, self.m2]:
            if var.shape == m.q.shape:
                return m.fft(var)
        return var

    # TODO: Fix docstring and type-hints.
    def subgrid_forcing(self, var):
        """Compute subgrid forcing of a given `var` (as string).

        Parameters
        ----------
        var : ???
            What is ``var``?

        Returns
        -------
        ???

        """
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        adv1 = self.coarsen(self.m1._advect(q1))
        adv2 = self.to_real(self.m2._advect(q2))
        return adv1 - adv2

    # TODO: Add type-hints and update docstring.
    def subgrid_fluxes(self, var):
        """Compute subgrid fluxes (wrt. u and v) of a given `var`.

        Parameters
        ----------
        var : ???
            What is ``var``.

        Returns
        -------
        Tuple[???, ???]
            ???

        """
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        u_flux = self.coarsen(self.m1.ufull * q1) - self.m2.ufull * q2
        v_flux = self.coarsen(self.m1.vfull * q1) - self.m2.vfull * q2
        return u_flux, v_flux

    # TODO: Can we give ``ratio`` more desciptive name?
    @property
    def ratio(self):
        """Ratio of high-res to low-res grid length.

        Returns
        -------
        ratio : ???
            A ratio of ? to ?.

        """
        return self.m1.nx / self.m2.nx

    # TODO: Type-hint and document.
    def coarsen(self, var):
        """Filter and coarse-grain a variable (as array).

        Parameters
        ----------
        var : ???
            Description of ``var``.

        """
        raise NotImplementedError()

    # TODO: Document.
    @cached_property
    def ds1(self):
        """xarray representation of the high-res model.

        Returns
        -------
        ???


        """
        return self.m1.to_dataset()


# TODO: Add docstrings and type-hints for this class.
class SpectralCoarsener(Coarsener):
    """Spectral truncation with a configurable filter."""

    def coarsen(self, var):
        # Truncate high-frequency indices & filter
        vh = self.to_spec(var)
        nk = self.m2.qh.shape[1] // 2
        trunc = np.hstack((vh[:, :nk, : nk + 1], vh[:, -nk:, : nk + 1]))
        filtered = trunc * self.spectral_filter / self.ratio**2
        return self.to_real(filtered)

    @property
    def spectral_filter(self):
        raise NotImplementedError()


# TODO: Add docstring and type-hints.
class Operator1(SpectralCoarsener):
    """Spectral truncation with a sharp filter."""

    @property
    def spectral_filter(self):
        return self.m2.filtr


# TODO: Add docstrings and type-hints.
class Operator2(SpectralCoarsener):
    """Spectral truncation with a softer Gaussian filter."""

    @property
    def spectral_filter(self):
        return np.exp(-self.m2.wv**2 * (2 * self.m2.dx) ** 2 / 24)


# TODO: add docstring and type-hints.
class Operator3(Coarsener):
    """Diffusion-based filter, then real-space coarsening."""

    def coarsen(self, var):
        # TODO: Place import in proper place (top-level), if allowed?
        import gcm_filters

        f = gcm_filters.Filter(
            dx_min=1,
            filter_scale=self.ratio,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR,
        )
        d = self.m1.to_dataset().isel(time=-1)
        q = d.q * 0 + self.to_real(var)  # hackily convert to data array
        r = int(self.ratio)
        assert r == self.ratio
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
