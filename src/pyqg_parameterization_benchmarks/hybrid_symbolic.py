import pyqg
import gplearn
import gplearn.genetic
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from pyqg_parameterization_benchmarks.utils import FeatureExtractor, Parameterization

def make_custom_gplearn_functions(ds):
    """Define custom gplearn functions for spatial derivatives that are specific
    to the spatial shape of that particular dataset"""

    extractor = FeatureExtractor(ds)

    def apply_spatial(func, x):
        r = func(x.reshape(ds.q.shape))
        if isinstance(r, xr.DataArray): r = r.data
        return r.reshape(x.shape)

    ddx = lambda x: apply_spatial(extractor.ddx, x)
    ddy = lambda x: apply_spatial(extractor.ddy, x)
    lap = lambda x: apply_spatial(extractor.laplacian, x)
    adv = lambda x: apply_spatial(extractor.advected, x)

    # create gplearn function objects to represent these transformations
    return [          
        gplearn.functions._Function(function=ddx, name='ddx', arity=1),
        gplearn.functions._Function(function=ddy, name='ddy', arity=1),
        gplearn.functions._Function(function=lap, name='laplacian', arity=1),
        gplearn.functions._Function(function=adv, name='advected', arity=1),
    ]

def run_gplearn_iteration(ds, target,
        base_features=['q','u','v'],
        base_functions=['add','mul'],
        **kwargs):
    """Run gplearn for one iteration using custom spatial derivatives."""

    # Flatten the input and target data
    x = np.array([ds[feature].data.reshape(-1)
                for feature in base_features]).T
    y = target.reshape(-1)
    
    gplearn_kwargs = dict(
        population_size=5000,
        generations=50,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.001,
        metric='pearson', # IMPORTANT: fit using pearson correlation, not MSE
        const_range=(-2,2),
    )
    
    gplearn_kwargs.update(**kwargs)

    # Configure gplearn to run with a relatively small population
    # and for relatively few generations (again for performance)
    sr = gplearn.genetic.SymbolicRegressor(
        feature_names=base_features,
        function_set=(
            base_functions + make_custom_gplearn_functions(ds) # use our custom ops
        ),
        **gplearn_kwargs
    )

    # Fit the model
    sr.fit(x, y)

    # Return the result
    return sr

class LinearSymbolicRegression(Parameterization):
    def __init__(self, lr1, lr2, inputs, target):
        self.lr1 = lr1
        self.lr2 = lr2
        self.inputs = inputs
        self.target = target

    @property
    def targets(self):
        return [self.target]
    
    @property
    def models(self):
        return [self.lr1, self.lr2]
        
    def predict(self, m):
        extract = FeatureExtractor(m)
        
        x = extract(self.inputs)

        preds = []
        
        # Do some slightly annoying reshaping to properly apply LR coefficients
        # to data that may or may not have extra batch dimensions 
        for z, lr in enumerate(self.models):
            data_indices = [slice(None) for _ in x.shape]
            data_indices[-3] = z
            x_z = x[tuple(data_indices)]
            coef_indices = [np.newaxis for _ in x_z.shape]
            coef_indices[0] = slice(None)
            c_z = lr.coef_[tuple(coef_indices)]
            pred_z = (x_z * c_z).sum(axis=0)
            preds.append(pred_z)
            
        preds = np.stack(preds, axis=-3)
        res = {}
        res[self.target] = preds
        return res
    
    @classmethod
    def fit(kls, ds, inputs, target='q_subgrid_forcing'):
        lrs = []
        for z in [0,1]:
            extract = FeatureExtractor(ds.isel(lev=z))
            lrs.append(
                LinearRegression(fit_intercept=False).fit(
                    extract(inputs, flat=True),
                    extract(target, flat=True)
                )
            )
        return kls(*lrs, inputs, target)  
    
def corr(a,b):
    return pearsonr(np.array(a.data).ravel(), np.array(b.data).ravel())[0]

def hybrid_symbolic_regression(ds, target='q_subgrid_forcing', max_iters=10, verbose=True, **kw):
    extract = FeatureExtractor(ds)
    residual = ds[target]
    terms = []
    vals = []
    lrs = []
    
    try:
        for i in range(max_iters):
            for lev in [0,1]:
                sr = run_gplearn_iteration(ds.isel(lev=lev),
                                           target=residual.isel(lev=lev).data,
                                           **kw)
                new_term = str(sr._program)
                new_vals = extract(new_term)
                # Prevent spurious duplicates, e.g. ddx(q) and ddx(add(1,q))
                if not any(corr(new_vals, v) > 0.99 for v in vals):
                    terms.append(new_term)
                    vals.append(new_vals)
            lrs.append(LinearSymbolicRegression.fit(ds, terms, target))
            preds = lrs[-1].test_offline(ds).load()
            residual = (ds[target] - preds[f"{target}_predictions"]).load()
            if verbose:
                print(f"Iteration {i+1}")
                print("Discovered terms:", terms)
                print("Train correlations:", preds.correlation.data)

    except KeyboardInterrupt:
        if verbose:
            print("Exiting early due to keyboard interrupt")

    return terms, lrs
