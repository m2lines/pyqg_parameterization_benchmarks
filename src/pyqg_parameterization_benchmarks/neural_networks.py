"""Module containing neural networks."""
from typing import Dict, Tuple, Any
import os
import glob
import pickle


import pyqg
import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MSELoss
from torch import optim
from torch.autograd import grad, Variable

import numpy as np
from pyqg_parameterization_benchmarks.utils import FeatureExtractor, Parameterization


class FullyCNN(Sequential):
    """CNN model plus helpers for dealing with constraints and scaling.

    Parameters
    ----------
    inputs : ???
        The inputs to the model. Tensors?
    targets : ???
        The targets/ground truths. Tensors?
    padding : str, optional
        The padding argument. The options seem to be ``None``, ``"circular"``
        or ``"same"``, with zero explanation.
    zero_mean : bool
        Controls whether the mean of each channel is standardised to zero
        after being passed through the ``forward`` method.

    """

    def __init__(self, inputs, targets, padding="circular", zero_mean=True):
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

    def forward(self, x):
        r = super().forward(x)
        if self.is_zero_mean:
            return r - r.mean(dim=(1, 2, 3), keepdim=True)

        return r

    def extract_vars(self, m, features, dtype=np.float32):
        ex = FeatureExtractor(m)

        arr = np.stack([np.take(ex(feat), z, axis=-3) for feat, z in features], axis=-3)

        arr = arr.reshape((-1, len(features), ex.nx, ex.nx))
        arr = arr.astype(dtype)
        return arr

    def extract_inputs(self, m):
        return self.extract_vars(m, self.inputs)

    def extract_targets(self, m):
        return self.extract_vars(m, self.targets)

    def input_gradients(self, inputs, output_channel, j, i, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        X = self.input_scale.transform(self.extract_inputs(inputs))

        grads = []
        for (x_,) in minibatch(X, shuffle=False, as_tensor=False):
            x = Variable(torch.tensor(x_), requires_grad=True).to(device)
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

    def predict(self, inputs, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        X = self.input_scale.transform(self.extract_inputs(inputs))

        preds = []
        for (x,) in minibatch(X, shuffle=False):
            x = x.to(device)
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())

        preds = self.output_scale.inverse_transform(np.vstack(preds))

        s = list(inputs.q.shape)
        preds = np.stack(
            [preds[:, i].reshape(s[:-3] + s[-2:]) for i in range(len(self.targets))],
            axis=-3,
        )

        try:
            return preds.astype(inputs.q.dtype)
        except:
            return preds

    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    def fit(self, inputs, targets, rescale=False, **kw):
        if rescale or not hasattr(self, "input_scale") or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)
        if rescale or not hasattr(self, "output_scale") or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)
        train(
            self,
            self.input_scale.transform(inputs),
            self.output_scale.transform(targets),
            **kw,
        )

    def save(self, path):
        os.system(f"mkdir -p {path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu()
        torch.save(self.state_dict(), f"{path}/weights.pt")
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
        model.load_state_dict(torch.load(f"{path}/weights.pt"))
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
    def __init__(self, mean=0, std_dev=1):
        self.mean = mean
        self.std_dev = std_dev

    def transform(self, x):
        return (x - self.mean) / self.std_dev

    def inverse_transform(self, z):
        return z * self.std_dev + self.mean


class ChannelwiseScaler(BasicScaler):
    def __init__(self, x, zero_mean=False):
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


def minibatch(*arrays, batch_size=64, as_tensor=True, shuffle=True):
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    xform = torch.as_tensor if as_tensor else lambda x: x
    for step in range(steps):
        idx = order[step * batch_size : (step + 1) * batch_size]
        yield tuple(xform(array[idx]) for array in arrays)


def train(
    net, inputs, targets, num_epochs=50, batch_size=64, learning_rate=0.001, device=None
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        for x, y in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net.forward(x.to(device))
            ytrue = y.to(device)
            loss = criterion(yhat, ytrue)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
        print(f"Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}")
        scheduler.step()


class FCNNParameterization(Parameterization):
    def __init__(self, directory, models=None, **kw):
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
        targets = set()
        for model in self.models:
            for target, z in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    def predict(self, m):
        preds = {}

        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == len(m.q.shape)
            # Handle the arduous task of getting the indices right for many
            # possible input shapes (e.g. pyqg.Model or xr.Dataset snapshot
            # stack)
            for channel in range(pred.shape[-3]):
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
        inputs=["q", "u", "v"],
        targets=["q_subgrid_forcing"],
        num_epochs=50,
        zero_mean=True,
        padding="circular",
        **kw,
    ):

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
        for z, model in enumerate(models):
            model_dir = os.path.join(directory, f"models/{z}")
            if os.path.exists(model_dir):
                trained.append(FullyCNN.load(model_dir))
            else:
                X = model.extract_inputs(dataset)
                Y = model.extract_targets(dataset)
                model.fit(X, Y, num_epochs=num_epochs, **kw)
                model.save(os.path.join(directory, f"models/{z}"))
                trained.append(model)

        # Return the trained parameterization
        return cls(directory, models=trained)
