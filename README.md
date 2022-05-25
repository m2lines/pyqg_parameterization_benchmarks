# Benchmarking of machine learning ocean parameterizations in an idealized model

In this repository, we present code, data, and parameterizations to explore and reproduce results from [Benchmarking of machine learning ocean parameterizations in an idealized model](#).

## Main idea

Ocean models attempt to simulate continuous processes, but are discrete and run at finite resolution. The error incurred by discretization, however, can be approximated by _subgrid parameterizations_ and corrected at every timestep.  Subgrid parameterizations can be formulated/derived in many ways, e.g. as equations derived by physical analysis, as a neural network learned from data, or as equations again but learned from data with symbolic regression. In this work, we evaluate parameterizations of each kind.

Because the field of learning data-driven parameterizations is relatively new, however, there isn't a clear consensus on how to evaluate parameterizations. So in addition to contributing new parameterizations, we also provide new evaluation schemes, defining various ways of measuring to what extent a parameterization brings characteristics of low-resolution simulations into closer alignment with those of high-resolution simulations (which are assumed to be a better approximation of the true continuous system we want to model).

We develop these parameterizations and evaluation metrics with [`pyqg`](https://pyqg.readthedocs.io/en/latest/), an open-source Python framework for running quasi-geostrophic ocean simulations.

## Repository structure

- [`dataset_description.ipynb`](./notebooks/dataset_description.ipynb) provides documentation for our publicly accessible data.
- [`subgrid_forcing.ipynb`](./notebooks/subgrid_forcing.ipynb) demonstrates different forcings from [`coarsening_ops.py`](./src/pyqg_parameterization_benchmarks/coarsening_ops.py).

## Running the code

1. Clone the repository
1. Install locally as a package, e.g. with `pip install --editable .`
1. In Python, `import pyqg_parameterization_benchmarks`
