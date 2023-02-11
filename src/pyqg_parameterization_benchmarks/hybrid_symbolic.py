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
    to the spatial shape of that particular dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset generated from pyqg.QGModel runs

    Returns
    -------
    List[gplearn.functions._Function]
        List of functions representing spatial differential operators that can
        be evaluated over the dataset during genetic programming
    """

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
    """Run gplearn for one iteration using custom spatial derivatives.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset generated from pyqg.QGModel runs
    target : numpy.ndarray
        Target spatial field to be predicted from dataset attributes
    base_features : List[str]
        Features from the dataset to be used as the initial set of atomic
        inputs to genetic programming
    base_functions : List[str]
        Names of gplearn built-in functions to explore during genetic
        programming (in addition to differential operators)
    **kwargs : dict
        Additional arguments to pass to gplearn.genetic.SymbolicRegressor

    Returns
    -------
    gplearn.genetic.SymbolicRegressor
        A trained symbolic regression model that predicts the ``target`` based
        on functions of the dataset's ``base_features``

    """

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
    """Linear regression parameterization on top of symbolic expressions."""

    def __init__(self, lr1, lr2, inputs, target):
        """Initialize the parameterization. This is not intended to be called
        by the user directly; use the ``fit`` class method instead.

        Parameters
        ----------
        lr1 : sklearn.linear_model.LinearRegression
            Linear model to predict the upper layer's target based on input
            expressions
        lr2 : sklearn.linear_model.LinearRegression
            Linear model to predict the lower layer's target based on input
            expressions
        inputs : List[str]
            List of string input expressions and functions that can be
            extracted from a pyqg.QGModel or dataset using a
            ``FeatureExtractor``
        target : str
            String expression representing the target variable.

        """
        self.lr1 = lr1
        self.lr2 = lr2
        self.inputs = inputs
        self.target = target

    @property
    def targets(self):
        """
        Returns
        -------
        List[str]
            The targets of the parameterization

        """
        return [self.target]

    @property
    def models(self):
        """
        Returns
        -------
        List[sklearn.linear_model.LinearRegression]
            The linear models as a list

        """
        return [self.lr1, self.lr2]

    def predict(self, model_or_dataset):
        """Make a prediction for a given model or dataset.

        Parameters
        ----------
        model_or_dataset : Union[pyqg.QGModel, xarray.Dataset]
            The live-running model or dataset of stored model runs for which we
            want to make subgrid forcing predictions

        Returns
        -------
        result : Dict[str, numpy.ndarray]
            For each string target expression, the predicted subgrid forcing
            (as an array with the same shape as the corresponding inputs in the
            model or dataset)

        """
        extract = FeatureExtractor(model_or_dataset)

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
        result = {}
        result[self.target] = preds
        return result

    @classmethod
    def fit(kls, dataset, inputs, target='q_subgrid_forcing'):
        """Fit a linear regression parameterization on top of the given
        ``dataset`` in terms of symbolic ``inputs``.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset of pyqg.QGModel runs
        inputs : List[str]
            List of expressions that can be evaluated by a
            ``FeatureExtractor``, to be used as inputs for the linear
            regression models
        target : Optional[str]
            Target variable to predict. Defaults to the subgrid forcing of
            potential vorticity.

        Returns
        -------
        LinearSymbolicRegression
            Resulting linear regression parameterization

        """
        lrs = []
        for z in [0,1]:
            extract = FeatureExtractor(dataset.isel(lev=z))
            lrs.append(
                LinearRegression(fit_intercept=False).fit(
                    extract(inputs, flat=True),
                    extract(target, flat=True)
                )
            )
        return kls(*lrs, inputs, target)

def corr(a,b):
    """Return the Pearson correlation between two spatial data arrays.

    Parameters
    ----------
    a : xarray.DataArray
        First spatial data array
    b : xarray.DataArray
        Second spatial data array

    Returns
    -------
    float
        Correlation between the data arrays

    """
    return pearsonr(np.array(a.data).ravel(), np.array(b.data).ravel())[0]

def hybrid_symbolic_regression(ds, target='q_subgrid_forcing', max_iters=10, verbose=True, **kw):
    """Run hybrid symbolic and linear regression, using symbolic regression to
    find expressions correlated wit the output, then fitting linear regression
    to get an exact expression, then running symbolic regression again on the
    resulting residuals (repeating until ``max_iters``).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of pyqg.QGModel runs
    target : str
        String representing target variable to predict. Defaults to subgrid
        forcing of potential vorticity.
    max_iters : int
        Number of iterations to run. Defaults to 10.
    verbose : bool
        Whether to print intermediate outputs. Defaults to True.
    **kw : dict
        Additional arguments to pass to ``run_gplearn_iteration``.

    Returns
    -------
    Tuple[List[str], List[LinearSymbolicRegression]]
        List of terms discovered at each iteration alongside list of linear
        symbolic regression parameterization objects (each fit to all terms
        available at corresponding iteration)

    """
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
