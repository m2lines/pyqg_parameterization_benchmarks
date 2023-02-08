"""Module containing neural networks."""
from typing import Tuple
import os
import glob
import pickle


import pyqg
import torch
from torch import (  # pylint: disable=no-name-in-module
    as_tensor,
    tensor,
    cuda,
    Tensor,
    save,
    load,
    no_grad,
)
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MSELoss
from torch import optim
from torch.autograd import grad, Variable

import numpy as np
from pyqg_parameterization_benchmarks.utils import FeatureExtractor, Parameterization


class FullyCNN(Sequential):
    """CNN model plus helpers for dealing with constraints and scaling.

    Parameters
    ----------
    inputs : List[str]
        The inputs to the model, represented as a list of string attributes
        that indicate spatial field variables present on associated datasets.
    targets : List[str]
        The targets of the model, represented as a list of string attributes
        that indicate spatial field variables present on associated datasets.
    padding : str, optional
        The padding argument. Should be one of ``None``, ``"circular"``
        or ``"same"``.
    zero_mean : bool
        Controls whether the mean of each channel is standardised to zero
        after being passed through the ``forward`` method.

    """

    # TODO: Supply types for ``inputs`` and ``targets`` in docstring.

    # TODO: Complete type-hinting in __init__ def.
    def __init__(
        self,
        inputs,
        targets,
        padding: str = "circular",
        zero_mean: bool = True,
    ):
        """Build ``FullyCNN``."""
        self.padding = padding
        self.is_zero_mean = zero_mean

        padding_3, padding_5 = self._process_padding()

        self.inputs = inputs
        self.targets = targets
        n_in = len(inputs)
        n_out = len(targets)
        self.n_in = n_in

        kwargs = {"padding_mode": "circular"} if padding == "circular" else {}

        super().__init__(
            ConvBlock(n_in, 128, 5, padding_5, **kwargs),
            ConvBlock(128, 64, 5, pad=padding_5, **kwargs),
            ConvBlock(64, 32, 3, pad=padding_3, **kwargs),
            ConvBlock(32, 32, 3, pad=padding_3, **kwargs),
            ConvBlock(32, 32, 3, pad=padding_3, **kwargs),
            ConvBlock(32, 32, 3, pad=padding_3, **kwargs),
            ConvBlock(32, 32, 3, pad=padding_3, **kwargs),
            Conv2d(32, n_out, 3, padding=padding_3),
        )

    def _process_padding(self) -> Tuple[int, int]:
        """Process the ``padding`` argument.

        Returns
        -------
        padding_3 : int
            The padding to use in a ``Conv2d`` if the kernel size is 3.
        padding_5 : int
            The padding to use in a ``Conv2d`` if the kernel size is 5.

        Raises
        ------
        ValueError
            If padding is not ``None``, ``"same"`` or ``circular``.

        """
        if self.padding is None:
            padding_5 = 0
            padding_3 = 0
        elif self.padding in ["same", "circular"]:
            padding_5 = 2
            padding_3 = 1
        else:
            msg = f"Unknown value for padding parameter. Choose from {None} "
            msg += f"{'circular'} or {'same'}."
            raise ValueError(msg)

        return padding_3, padding_5

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Pass ``x`` through the model.

        Parameters
        ----------
        x : Tensor
            A mini-batch of inputs.

        Returns
        -------
        r : Tensor
            The result of passing ``x`` through the model.

        """
        out = super().forward(x)
        if self.is_zero_mean:
            return out - out.mean(dim=(1, 2, 3), keepdim=True)

        return out

    def extract_vars(self, m, features, dtype=np.float32):
        """Take a list of string `features` and extract them from the
        xarray.Dataset or pyqg.QGModel `m` as a numpy.ndarray of the associated
        `dtype`."""
        # TODO: Provide docstring and type-hints.
        ex = FeatureExtractor(m)

        arr = np.stack([np.take(ex(feat), z, axis=-3) for feat, z in features], axis=-3)

        arr = arr.reshape((-1, len(features), ex.nx, ex.nx))
        arr = arr.astype(dtype)
        return arr

    def extract_inputs(self, m):
        """Take the input variable strings and extract a numpy array from the dataset or model `m`."""
        # TODO: Provide a docstring and type-hinting to explain this function.
        return self.extract_vars(m, self.inputs)

    def extract_targets(self, m):
        """Take the target variable strings and extract a numpy array from the dataset or model `m`."""
        # TODO: Add docstring and type-hinting.
        return self.extract_vars(m, self.targets)

    def input_gradients(self, inputs, output_channel, j, i, device=None):
        """The gradients of the model's output (at a given `output_channel` and
        `i`, `j` spatial position) with respect to a list of string
        `inputs`."""
        # TODO: Provide a docstring, types and maybe a 'Notes' section in doc.
        if device is None:
            device = torch.device("cuda:0" if cuda.is_available() else "cpu")
            self.to(device)

        X = self.input_scale.transform(self.extract_inputs(inputs))

        grads = []
        for (x_,) in minibatch(X, shuffle=False, return_tensor=False):
            x = Variable(tensor(x_), requires_grad=True).to(device)
            y = self.forward(x)[:, output_channel, j, i]
            grads.append(grad(y.sum(), x)[0].cpu().numpy())

        grads = self.output_scale.inverse_transform(np.vstack(grads))

        s = list(inputs.q.shape)
        grads = np.stack(
            [grads[:, i].reshape(s[:-3] + s[-2:]) for i in range(len(self.targets))],
            axis=-3,
        )

        if isinstance(inputs, pyqg.Model):
            return grads.astype(inputs.q.dtype)
        else:
            return grads

    # TODO: Confirm it's okay to decorate this full function with no_grad()
    @no_grad()
    def predict(self, inputs, device=None):
        """Predict using inputs?."""
        # TODO: Should we say `self.eval()` here to be careful?
        if device is None:
            device = torch.device("cuda:0" if cuda.is_available() else "cpu")
            self.to(device)

        X = self.input_scale.transform(self.extract_inputs(inputs))

        preds = []
        for (x,) in minibatch(X, shuffle=False):
            x = x.to(device)
            preds.append(self.forward(x).cpu().numpy())

        preds = self.output_scale.inverse_transform(np.vstack(preds))

        # TODO: What is ``s``? Use pythonic variable name.
        s = list(inputs.q.shape)
        preds = np.stack(
            [preds[:, i].reshape(s[:-3] + s[-2:]) for i in range(len(self.targets))],
            axis=-3,
        )

        try:
            return preds.astype(inputs.q.dtype)
        # TODO What sort of exception are you expecting here?
        except:
            return preds

    def mse(self, inputs, targets):
        """Return the mean-squared-error between ``inputs`` and ``targets``."""
        # TODO: Update docstring and type hint.
        # TODO: Do we need to re-implement this when PyTorch has done it?
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    def fit(self, inputs, targets, rescale=False, **kw):
        """Fit the model using ``inputs`` and ``targets``."""
        # TODO: Give more comprehensive docstrings and explanations of types.
        if rescale or not hasattr(self, "input_scale") or self.input_scale is None:
            # TODO: Attributes should not be set in this way outside the init.
            self.input_scale = ChannelwiseScaler(inputs)
        if rescale or not hasattr(self, "output_scale") or self.output_scale is None:
            # TODO: Attributes should not be set in this way outside the init.
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)
        train(
            self,
            self.input_scale.transform(inputs),
            self.output_scale.transform(targets),
            **kw,
        )

    def save(self, path):
        """Add a docstring explaining what's going here."""
        # TODO: Make it clear what we are saving here.
        # TODO: Consider using the glorious pathlib library.
        os.system(f"mkdir -p {path}")
        device = torch.device("cuda:0" if cuda.is_available() else "cpu")
        self.cpu()
        save(self.state_dict(), f"{path}/weights.pt")
        self.to(device)
        if hasattr(self, "input_scale") and self.input_scale is not None:
            with open(f"{path}/input_scale.pkl", "wb") as f:
                pickle.dump(self.input_scale, f)
        if hasattr(self, "output_scale") and self.output_scale is not None:
            with open(f"{path}/output_scale.pkl", "wb") as f:
                pickle.dump(self.output_scale, f)
        with open(f"{path}/inputs.pkl", "wb") as f:
            pickle.dump(self.inputs, f)
        with open(f"{path}/targets.pkl", "wb") as f:
            pickle.dump(self.targets, f)
        if self.is_zero_mean:
            open(f"{path}/zero_mean", "a").close()
        if hasattr(self, "padding"):
            with open(f"{path}/padding", "w") as f:
                f.write(self.padding)

    @classmethod
    def load(cls, path, set_eval=True, **kwargs):
        """Add a docstring explaining what's going on here."""
        # TODO: Add docstring and type-hinting.
        # TODO: Don't use ``f`` as a variable name in context managers.
        with open(f"{path}/inputs.pkl", "rb") as f:
            inputs = pickle.load(f)
        with open(f"{path}/targets.pkl", "rb") as f:
            targets = pickle.load(f)
        kw = {}
        kw.update(**kwargs)
        if os.path.exists(f"{path}/padding"):
            with open(f"{path}/padding", "r") as f:
                kw["padding"] = f.read().strip()
        model = cls(inputs, targets, **kw)
        model.load_state_dict(load(f"{path}/weights.pt"))
        if os.path.exists(f"{path}/input_scale.pkl"):
            with open(f"{path}/input_scale.pkl", "rb") as f:
                model.input_scale = pickle.load(f)
        if os.path.exists(f"{path}/output_scale.pkl"):
            with open(f"{path}/output_scale.pkl", "rb") as f:
                model.output_scale = pickle.load(f)
        if os.path.exists(f"{path}/zero_mean"):
            model.is_zero_mean = True
        if set_eval:
            model.eval()
        return model


class ConvBlock(Sequential):
    """Two-dimensional convolutional subblock.

    ``Conv2d`` -> ``ReLU`` -> ``BatchNorm2d``

    Parameters
    ----------
    in_chans : int
        The number of inputs channels the block should expect.
    out_chans : int
        The number of output channels the block should produce.
    kernel_size : int
        The size of the kernel to use in the ``Conv2d`` layer.
    pad : int
        The padding argument for the ``Conv2d``.
    **conv_keyword_args : Dict[str, Any]
        Keyword arguments for the ``Conv2d``.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        pad: int,
        **conv_kwargs,
    ):
        """Build ``_ConBlock``."""
        super().__init__(
            Conv2d(in_chans, out_chans, kernel_size, padding=pad, **conv_kwargs),
            ReLU(),
            BatchNorm2d(out_chans),
        )


class BasicScaler(object):
    """Basic scaling class for setting mean zero and std dev to 1.

    #TODO: Is this transformation to be applied to mini-batches using stats
    obtained over the dataset as a whole? If so, I'd modify the description.

    #TODO: Make it clear in the docstring how this is distinct from channel-
    wise rescaling.

    Parameters
    ----------
    mean : float
        The mean to subtract from inputs.
    std_dev : float
        The standard deviation to divide inputs by.

    """

    # TODO: Add type-hinting for this class.

    def __init__(self, mean=0, std_dev=1):
        """Construct ``BasicScaler``."""
        self.mean = mean
        self.std_dev = std_dev

    def transform(self, x):
        """Rescale ``x``."""
        # TODO: Update the docstring with descriptive variable names
        return (x - self.mean) / self.std_dev

    def inverse_transform(self, z):
        """Invert the basic rescaling transform."""
        # TODO: Update the docstring with descriptive variable names
        return z * self.std_dev + self.mean


class ChannelwiseScaler(BasicScaler):
    """Add a docstring with parameters and explanation."""

    # TODO: Add dosctring and type-hints.
    # TODO: See ``torchvision.transforms.Normalize``.

    def __init__(self, x, zero_mean=False):
        """Construct ``ChannelwiseScaler``."""

        # TODO: Expand on description of what's going on here in docstring.
        assert len(x.shape) == 4
        if zero_mean:
            mean = 0
        else:
            mean = np.array([x[:, i].mean() for i in range(x.shape[1])])[
                np.newaxis, :, np.newaxis, np.newaxis
            ]
        std_dev = np.array([x[:, i].std() for i in range(x.shape[1])])[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        super().__init__(mean, std_dev)


def minibatch(*arrays, batch_size=64, return_tensor=True, shuffle=True):
    """Add a docstring explaining what is going on here."""
    # TODO: Update docstring, explain what's going on and add type-hinting.
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    xform = as_tensor if return_tensor else lambda x: x
    for step in range(steps):
        idx = order[step * batch_size : (step + 1) * batch_size]
        yield tuple(xform(array[idx]) for array in arrays)


def train(
    net,
    inputs,
    targets,
    num_epochs=50,
    batch_size=64,
    learning_rate=0.001,
    device=None,
):
    """Add a descriptive docstring."""
    # TODO: Add a descriptive docstring and type-hints.
    if device is None:
        device = device("cuda:0" if cuda.is_available() else "cpu")
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(num_epochs / 2),
            int(num_epochs * 3 / 4),
            int(num_epochs * 7 / 8),
        ],
        gamma=0.1,
    )
    criterion = MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch, target in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net.forward(batch.to(device))
            ytrue = target.to(device)
            loss = criterion(yhat, ytrue)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
        print(f"Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}")
        scheduler.step()


class FCNNParameterization(Parameterization):
    """Add docstring explaining what this class is and does."""

    # TODO: Update dosctring and type-hints. Make it clear what this class does.
    def __init__(self, directory, models=None, **kw):
        """Construct ``FCNNParameterization``."""
        self.directory = directory
        self.models = (
            models
            if models is not None
            else [
                FullyCNN.load(f, **kw)
                for f in sorted(glob.glob(os.path.join(directory, "models/*")))
            ]
        )

    @property
    def targets(self):
        """What do?."""
        # TODO: Add a docstring and explain what we are doing here.
        targets = set()
        for model in self.models:
            for target, _ in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    def predict(self, m):
        """Predict ?."""
        # TODO: Provide docstring, type-hints and explanation.
        preds = {}

        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == len(m.q.shape)
            # Handle the arduous task of getting the indices right for many
            # possible input shapes (e.g. pyqg.Model or xr.Dataset snapshot
            # stack)
            for channel in range(pred.shape[-3]):
                # TODO: What is z? Please use a descriptive name.
                target, z = model.targets[channel]
                if target not in preds:
                    preds[target] = np.zeros_like(m.q)
                out_indices = [slice(None) for _ in m.q.shape]
                out_indices[-3] = slice(z, z + 1)
                in_indices = [slice(None) for _ in m.q.shape]
                in_indices[-3] = slice(channel, channel + 1)
                preds[target][tuple(out_indices)] = pred[tuple(in_indices)]

        return preds

    @classmethod
    def train_on(
        cls,
        dataset,
        directory,
        inputs=None,
        targets=None,
        num_epochs=50,
        zero_mean=True,
        padding="circular",
        **kw,
    ):
        """Provide doctring with explanation of this function."""
        # TODO: Provide docstring and type-hints.

        inputs = inputs if inputs is not None else ["q", "u", "v"]
        targets = targets if targets is not None else ["q_subgrid_forcing"]

        layers = range(len(dataset.lev))

        models = [
            FullyCNN(
                [(feat, zi) for feat in inputs for zi in layers],
                [(feat, z) for feat in targets],
                zero_mean=zero_mean,
                padding=padding,
            )
            for z in layers
        ]

        # Train models on dataset and save them
        trained = []
        for idx, model in enumerate(models):
            model_dir = os.path.join(directory, f"models/{idx}")
            if os.path.exists(model_dir):
                trained.append(FullyCNN.load(model_dir))
            else:
                x_items = model.extract_inputs(dataset)
                y_items = model.extract_targets(dataset)
                model.fit(x_items, y_items, num_epochs=num_epochs, **kw)
                model.save(os.path.join(directory, f"models/{idx}"))
                trained.append(model)

        # Return the trained parameterization
        return cls(directory, models=trained)
