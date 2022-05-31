import re
import pyqg
import operator
import numpy as np
import xarray as xr

class Parameterization(pyqg.Parameterization):
    """Helper class for defining parameterizations. This extends the normal
    pyqg parameterization framework to handle prediction of either subgrid
    forcings or fluxes, as well as to apply to either pyqg.Models or
    xarray.Datasets."""
    
    @property
    def targets(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
    @property
    def nx(self):
        return 64 # Future work should generalize this.

    @property
    def parameterization_type(self):
        if any(q in self.targets[0] for q in ['q_forcing', 'q_subgrid']):
            return 'q_parameterization'
        else:
            return 'uv_parameterization'

    def __call__(self, m):
        def arr(x):
            if isinstance(x, xr.DataArray): x = x.data
            return x.astype(m.q.dtype)

        preds = self.predict(m)
        keys = list(sorted(preds.keys()))
        assert keys == self.targets
        if len(keys) == 1:
            return arr(preds[keys[0]])
        elif keys == ['uq_subgrid_flux', 'vq_subgrid_flux']:
            ex = FeatureExtractor(m)
            return arr(ex.ddx(preds['uq_subgrid_flux']) + ex.ddy(preds['vq_subgrid_flux']))
        elif 'uu_subgrid_flux' in keys and len(keys) == 3:
            ex = FeatureExtractor(m)
            return (arr(ex.ddx(preds['uu_subgrid_flux']) + ex.ddy(preds['uv_subgrid_flux'])),
                    arr(ex.ddx(preds['uv_subgrid_flux']) + ex.ddy(preds['vv_subgrid_flux'])))
        else:
            return tuple(arr(preds[k]) for k in keys)

    def run_online(self, sampling_freq=1000, **kw):
        """Run a parameterized pyqg.QGModel, saving snapshots every 1000h."""
        
        # Initialize a pyqg model with this parameterization
        params = dict(kw)
        params[self.parameterization_type] = self
        params['nx'] = self.nx
        m = pyqg.QGModel(**params)

        # Run it, saving snapshots
        snapshots = []
        while m.t < m.tmax:
            if m.tc % sampling_freq == 0:
                snapshots.append(m.to_dataset().copy(deep=True))
            m._step_forward()

        ds = xr.concat(snapshots, dim='time')
        
        # Diagnostics get dropped by this procedure since they're only present for
        # part of the timeseries; resolve this by saving the most recent
        # diagnostics (they're already time-averaged so this is ok)
        for k,v in snapshots[-1].variables.items():
            if k not in ds:
                ds[k] = v.isel(time=-1)

        # Drop complex variables since they're redundant and can't be saved
        complex_vars = [k for k,v in ds.variables.items() if np.iscomplexobj(v)]
        ds = ds.drop_vars(complex_vars)

        return ds

    def test_offline(self, dataset):
        """Evaluate the parameterization on an offline dataset,
        computing a variety of metrics."""
        
        test = dataset[self.targets]
        
        for key, val in self.predict(dataset).items():
            truth = test[key]
            test[f"{key}_predictions"] = truth*0 + val
            preds = test[f"{key}_predictions"]
            error = (truth - preds)**2

            true_centered = (truth - truth.mean())
            pred_centered = (preds - preds.mean())
            true_var = true_centered**2
            pred_var = pred_centered**2
            true_pred_cov = true_centered * pred_centered

            def dims_except(*dims):
                return [d for d in test[key].dims if d not in dims]
            
            time = dims_except('x','y','lev')
            space = dims_except('time','lev')
            both = dims_except('lev')

            test[f"{key}_spatial_mse"] = error.mean(dim=time)
            test[f"{key}_temporal_mse"] = error.mean(dim=space)
            test[f"{key}_mse"] = error.mean(dim=both)

            test[f"{key}_spatial_skill"] = 1 - test[f"{key}_spatial_mse"] / true_var.mean(dim=time)
            test[f"{key}_temporal_skill"] = 1 - test[f"{key}_temporal_mse"] / true_var.mean(dim=space)
            test[f"{key}_skill"] = 1 - test[f"{key}_mse"] / true_var.mean(dim=both)

            test[f"{key}_spatial_correlation"] = xr.corr(truth, preds, dim=time)
            test[f"{key}_temporal_correlation"] = xr.corr(truth, preds, dim=space)
            test[f"{key}_correlation"] = xr.corr(truth, preds, dim=both)

        for metric in ['correlation', 'mse', 'skill']:
            test[metric] = sum(
                test[f"{key}_{metric}"] for key in self.targets
            ) / len(self.targets)

        return test

class FeatureExtractor:
    """Helper class for taking spatial derivatives and translating string
    expressions into data. Works with either pyqg.Model or xarray.Dataset."""
    
    def __call__(self, feature_or_features, flat=False):
        arr = lambda x: x.data if isinstance(x, xr.DataArray) else x
        if isinstance(feature_or_features, str):
            res = arr(self.extract_feature(feature_or_features))
            if flat: res = res.reshape(-1)

        else:
            res = np.array([arr(self.extract_feature(f)) for f in feature_or_features])
            if flat: res = res.reshape(len(feature_or_features), -1).T
        return res

    def __init__(self, model_or_dataset):
        self.m = model_or_dataset
        self.cache = {}
        
        if hasattr(self.m, '_ik'):
            self.ik, self.il = np.meshgrid(self.m._ik, self.m._il)
        elif hasattr(self.m, 'fft'):
            self.ik = 1j * self.m.k
            self.il = 1j * self.m.l
        else:
            k, l = np.meshgrid(self.m.k, self.m.l)
            self.ik = 1j * k
            self.il = 1j * l

        self.nx = self.ik.shape[0]
        self.wv2 = self.ik**2 + self.il**2

    # Helpers for taking FFTs / deciding if we need to
    def fft(self, x):
        try:
            return self.m.fft(x)
        except:
            # Convert to data array
            dims = [dict(y='l',x='k').get(d,d) for d in self['q'].dims]
            coords = dict([(d, self[d]) for d in dims])
            return xr.DataArray(np.fft.rfftn(x, axes=(-2,-1)), dims=dims, coords=coords)

    def ifft(self, x):
        try:
            return self.m.ifft(x)
        except:
            return self['q']*0 + np.fft.irfftn(x, axes=(-2,-1))
    
    def is_real(self, arr):
        return len(set(arr.shape[-2:])) == 1
    
    def real(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return arr
        return self.ifft(arr)
    
    def compl(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return self.fft(arr)
        return arr

    # Spectral derivatrives
    def ddxh(self, f): return self.ik * self.compl(f)
    def ddyh(self, f): return self.il * self.compl(f)
    def divh(self, x, y): return self.ddxh(x) + self.ddyh(y)
    def curlh(self, x, y): return self.ddxh(y) - self.ddyh(x)
    def laplacianh(self, x): return self.wv2 * self.compl(x)
    def advectedh(self, x_):
        x = self.real(x_)
        return self.ddxh(x * self.m.ufull) + self.ddyh(x * self.m.vfull)

    # Real counterparts
    def ddx(self, f): return self.real(self.ddxh(f))
    def ddy(self, f): return self.real(self.ddyh(f))
    def laplacian(self, x): return self.real(self.laplacianh(x))
    def advected(self, x): return self.real(self.advectedh(x))
    def curl(self, x, y): return self.real(self.curlh(x,y))
    def div(self, x, y): return self.real(self.divh(x,y))

    # Main function: interpreting a string as a feature
    def extract_feature(self, feature):
        """Evaluate a string feature, e.g. laplacian(advected(curl(u,v)))."""
        
        # Helper to recurse on each side of an arity-2 expression
        def extract_pair(s):
            depth = 0
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    return self.extract_feature(s[:i].strip()), self.extract_feature(s[i+1:].strip())
            raise ValueError(f"string {s} is not a comma-separated pair")

        real_or_spectral = lambda arr: arr + [a+'h' for a in arr]
            
        if not self.extracted(feature):
            # Check if the feature looks like "function(expr1, expr2)"
            # (better would be to write a grammar + use a parser,
            # but this is a very simple DSL)
            match = re.search(f"^([a-z]+)\((.*)\)$", feature)
            if match:
                op, inner = match.group(1), match.group(2)
                if op in ['mul', 'add', 'sub', 'pow']:
                    self.cache[feature] = getattr(operator, op)(*extract_pair(inner))
                elif op in ['neg', 'abs']:
                    self.cache[feature] = getattr(operator, op)(self.extract_feature(inner))
                elif op in real_or_spectral(['div', 'curl']):
                    self.cache[feature] = getattr(self, op)(*extract_pair(inner))
                elif op in real_or_spectral(['ddx', 'ddy', 'advected', 'laplacian']):
                    self.cache[feature] = getattr(self, op)(self.extract_feature(inner))
                else:
                    raise ValueError(f"could not interpret {feature}")
            elif re.search(f"^[\-\d\.]+$", feature):
                # ensure numbers still work
                return float(feature)
            elif feature == 'streamfunction':
                # hack to make streamfunctions work in both datasets & pyqg.Models
                self.cache[feature] = self.ifft(self['ph'])
            else:
                raise ValueError(f"could not interpret {feature}")

        return self[feature]

    def extracted(self, key):
        return key in self.cache or hasattr(self.m, key)

    # A bit of additional hackery to allow for the reading of features or properties
    def __getitem__(self, q):
        if isinstance(q, str):
            if q in self.cache:
                return self.cache[q]
            elif re.search(f"^[\-\d\.]+$", q):
                return float(q)
            else:
                return getattr(self.m, q)
        elif any([isinstance(q, kls) for kls in [xr.DataArray, np.ndarray, int, float]]):
            return q
        else:
            raise KeyError(q)

def energy_budget_term(model, term):
    val = model[term]
    if 'paramspec_' + term in model:
        val += model['paramspec_' + term]
    return val.sum('l')

def energy_budget_figure(models, skip=0):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    vmax = 0
    for i,term in enumerate(['KEflux','APEflux','APEgenspec','KEfrictionspec']):
        plt.subplot(1,4,i+1,title=term)
        plt.axhline(0,color='gray', ls='--')
        
        for model, label in models:
            spec = energy_budget_term(model, term)
            plt.semilogx(model.k[skip:], spec[skip:],
                         label=label, lw=3, ls=('--' if '+' in label else '-'))
            vmax = max(vmax, spec[skip:].max())
        plt.grid(which='both',alpha=0.25)
        if i == 0: plt.ylabel("Energy transfer $[m^2 s^{-3}]$")
        else: plt.gca().set_yticklabels([])
        if i == 3: plt.legend()
        plt.xlabel("Zonal wavenumber $[m^{-1}]$")
    for ax in fig.axes:
        ax.set_ylim(-vmax, vmax)
    plt.tight_layout()
    return fig
