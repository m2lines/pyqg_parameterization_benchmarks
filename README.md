# Benchmarking of machine learning ocean parameterizations in an idealized model

In this repository, we present code, data, and parameterizations to explore and reproduce results from _Benchmarking of machine learning ocean parameterizations in an idealized model_ (Submitted to JAMES, https://www.essoar.org/doi/10.1002/essoar.10511742.1).

## Main idea

Ocean and climate models attempt to simulate continuous processes, but are discrete and run at finite resolution. The error incurred by discretization on a finite grid, however, can be approximated by _subgrid parameterizations_ and corrected at every timestep.  Subgrid parameterizations are attempting to capture the effects of scales that are not resolved on the finite grid of the climate or ocean models we are using. Subgrid parameterizations can be formulated and derived in many ways, e.g. as equations derived by physical analysis, as a neural network learned from data, or as equations again but learned from data with symbolic regression. In this work, we evaluate parameterizations of each kind.

Because the field of learning data-driven parameterizations is relatively new, however, there isn't a clear consensus on how to evaluate them. So in addition to contributing new parameterizations, we also provide new datasets and evaluation schemes, defining various ways of measuring to what extent a parameterization brings characteristics of low-resolution simulations into closer alignment with those of high-resolution simulations (which are assumed to be a better approximation of the true continuous system we want to model). 

We develop these parameterizations and evaluation metrics with [`pyqg`](https://pyqg.readthedocs.io/en/latest/), an open-source Python framework for running quasi-geostrophic ocean simulations. We are building this tool as part of a model hierarchy to ensure robust testing and validation of subgrid parameterizations for ocean and climate models.

## Repository structure

- [`dataset_description.ipynb`](./notebooks/dataset_description.ipynb) provides documentation for our publicly accessible data.
- [`subgrid_forcing.ipynb`](./notebooks/subgrid_forcing.ipynb) demonstrates different forcings from [`coarsening_ops.py`](./src/pyqg_parameterization_benchmarks/coarsening_ops.py).
- [`hybrid_symbolic.ipynb`](./notebooks/hybrid_symbolic.ipynb) demonstrates running symbolic regression based on [`hybrid_symbolic.py`](./src/pyqg_parameterization_benchmarks/hybrid_symbolic.py).
- [`neural_networks.ipynb`](./notebooks/neural_networks.ipynb) demonstrates running fully convolutional neural network parameterizations.
- [`online_metrics.ipynb`](./notebooks/online_metrics.ipynb) demonstrates how to compute online similarity metrics between neural networks, symbolic regression, and baseline physical parameterizations based on [`online_metrics.py`](./src/pyqg_parameterization_benchmarks/online_metrics.py).

## Running the code

1. Clone the repository
2. Install the requirements, e.g. with `pip install -r requirements.txt`
3. Install locally as a package, e.g. with `pip install --editable .`
4. Ensure the tests pass by running `pytest`

After this, you should be able to `import pyqg_parameterization_benchmarks` and run all of the [notebooks](./notebooks).
