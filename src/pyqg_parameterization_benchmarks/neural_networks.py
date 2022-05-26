import os
import glob
import pyqg
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
from collections import OrderedDict
from torch.autograd import grad, Variable
from pyqg_parameterization_benchmarks.utils import FeatureExtractor, Parameterization

class FullyCNN(nn.Sequential):
    """Pytorch class defining our CNN architecture, plus some helpers for
    dealing with constraints and scaling."""
    def __init__(self, inputs, targets, padding='circular', zero_mean=True):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding in ['same', 'circular']:
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.padding = padding
        self.inputs = inputs
        self.targets = targets
        n_in = len(inputs)
        n_out = len(targets)
        self.n_in = n_in
        kw = {}
        if padding == 'circular':
            kw['padding_mode'] = 'circular'
        block1 = self._make_subblock(nn.Conv2d(n_in, 128, 5, padding=padding_5, **kw))
        block2 = self._make_subblock(nn.Conv2d(128, 64, 5, padding=padding_5, **kw))
        block3 = self._make_subblock(nn.Conv2d(64, 32, 3, padding=padding_3, **kw))
        block4 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3, **kw))
        block5 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3, **kw))
        block6 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3, **kw))
        block7 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3, **kw))
        conv8 = nn.Conv2d(32, n_out, 3, padding=padding_3)
        super().__init__(*block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        self.is_zero_mean = zero_mean

    def forward(self, x):
        r = super().forward(x)
        if self.is_zero_mean:
            return r - r.mean(dim=(1,2,3), keepdim=True)
        else:
            return r
        
    def _make_subblock(self, conv):
        return [conv, nn.ReLU(), nn.BatchNorm2d(conv.out_channels)]

    def extract_vars(self, m, features, dtype=np.float32):
        ex = FeatureExtractor(m)

        arr = np.stack([
            np.take(ex(feat), z, axis=-3) for feat, z in features
        ], axis=-3)

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
        for x_, in minibatch(X, shuffle=False, as_tensor=False):
            x = Variable(torch.tensor(x_), requires_grad=True).to(device)
            y = self.forward(x)[:,output_channel,j,i]
            grads.append(grad(y.sum(), x)[0].cpu().numpy())

        grads = self.output_scale.inverse_transform(np.vstack(grads))

        s = list(inputs.q.shape)
        grads = np.stack([
            grads[:,i].reshape(s[:-3] + s[-2:])
            for i in range(len(self.targets))
        ], axis=-3)

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
        for x, in minibatch(X, shuffle=False):
            x = x.to(device)
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())

        preds = self.output_scale.inverse_transform(np.vstack(preds))

        s = list(inputs.q.shape)
        preds = np.stack([
            preds[:,i].reshape(s[:-3] + s[-2:])
            for i in range(len(self.targets))
        ], axis=-3)

        try:
            return preds.astype(inputs.q.dtype)
        except:
            return preds

    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true)**2, axis=1))

    def fit(self, inputs, targets, rescale=False, **kw):
        if rescale or not hasattr(self, 'input_scale') or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)
        if rescale or not hasattr(self, 'output_scale') or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)
        train(self,
              self.input_scale.transform(inputs),
              self.output_scale.transform(targets),
              **kw)

    def save(self, path):
        os.system(f"mkdir -p {path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu()
        torch.save(self.state_dict(), f"{path}/weights.pt")
        self.to(device)
        if hasattr(self, 'input_scale') and self.input_scale is not None:
            with open(f"{path}/input_scale.pkl", 'wb') as f:
                pickle.dump(self.input_scale, f)
        if hasattr(self, 'output_scale')  and self.output_scale is not None:
            with open(f"{path}/output_scale.pkl", 'wb') as f:
                pickle.dump(self.output_scale, f)
        with open(f"{path}/inputs.pkl", 'wb') as f:
            pickle.dump(self.inputs, f)
        with open(f"{path}/targets.pkl", 'wb') as f:
            pickle.dump(self.targets, f)
        if self.is_zero_mean:
            open(f"{path}/zero_mean", 'a').close()
        if hasattr(self, 'padding'):
            with open(f"{path}/padding", 'w') as f:
                f.write(self.padding)

    @classmethod
    def load(cls, path, set_eval=True, **kwargs):
        with open(f"{path}/inputs.pkl", 'rb') as f:
            inputs = pickle.load(f)
        with open(f"{path}/targets.pkl", 'rb') as f:
            targets = pickle.load(f)
        kw = {}
        kw.update(**kwargs)
        if os.path.exists(f"{path}/padding"):
            with open(f"{path}/padding", 'r') as f:
                kw['padding'] = f.read().strip()
        model = cls(inputs, targets, **kw)
        model.load_state_dict(torch.load(f"{path}/weights.pt"))
        if os.path.exists(f"{path}/input_scale.pkl"):
            with open(f"{path}/input_scale.pkl", 'rb') as f:
                model.input_scale = pickle.load(f)
        if os.path.exists(f"{path}/output_scale.pkl"):
            with open(f"{path}/output_scale.pkl", 'rb') as f:
                model.output_scale = pickle.load(f)
        if os.path.exists(f"{path}/zero_mean"):
            model.is_zero_mean = True
        if set_eval:
            model.eval()
        return model

class BasicScaler(object):
    def __init__(self, mu=0, sd=1):
        self.mu = mu
        self.sd = sd
        
    def transform(self, x):
        return (x - self.mu) / self.sd
    
    def inverse_transform(self, z):
        return z * self.sd + self.mu

class ChannelwiseScaler(BasicScaler):
    def __init__(self, x, zero_mean=False):
        assert len(x.shape) == 4
        if zero_mean:
            mu = 0
        else:
            mu = np.array([x[:,i].mean()
                for i in range(x.shape[1])])[np.newaxis,:,np.newaxis,np.newaxis]
        sd = np.array([x[:,i].std()
            for i in range(x.shape[1])])[np.newaxis,:,np.newaxis,np.newaxis]
        super().__init__(mu, sd)

def minibatch(*arrays, batch_size=64, as_tensor=True, shuffle=True):
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    xform = torch.as_tensor if as_tensor else lambda x: x
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        yield tuple(xform(array[idx]) for array in arrays)

def train(net, inputs, targets, num_epochs=50, batch_size=64, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)
    criterion = nn.MSELoss()
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
        self.models = models if models is not None else [
            FullyCNN.load(f, **kw)
            for f in sorted(glob.glob(os.path.join(directory, "models/*")))
        ]

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
                out_indices[-3] = slice(z,z+1)
                in_indices = [slice(None) for _ in m.q.shape]
                in_indices[-3] = slice(channel,channel+1)
                preds[target][tuple(out_indices)] = pred[tuple(in_indices)]

        return preds

    @classmethod
    def train_on(cls, dataset, directory,
            inputs=['q','u','v'],
            targets=['q_subgrid_forcing'],
            num_epochs=50,
            zero_mean=True,
            padding='circular', **kw):

        layers = range(len(dataset.lev))

        models = [
            FullyCNN(
                [(feat, zi) for feat in inputs for zi in layers],
                [(feat, z) for feat in targets],
                zero_mean=zero_mean,
                padding=padding

            ) for z in layers
        ]

        # Train models on dataset and save them
        trained = []
        for z, model in enumerate(models):
            model_dir = os.path.join(directory, f"models/{z}")
            if os.path.exists(model_dir):
                models2.append(FullyCNN.load(model_dir))
            else:
                X = model.extract_inputs(dataset)
                Y = model.extract_targets(dataset)
                model.fit(X, Y, num_epochs=num_epochs, **kw)
                model.save(os.path.join(directory, f"models/{z}"))
                trained.append(model)

        # Return the trained parameterization
        return cls(directory, models=trained)
