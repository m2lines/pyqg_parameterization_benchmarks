import os
import pyqg
import numpy as np
import xarray as xr
from pyqg_parameterization_benchmarks.utils import FeatureExtractor
from pyqg_parameterization_benchmarks.hybrid_symbolic import LinearSymbolicRegression
from pyqg_parameterization_benchmarks.neural_networks import FCNNParameterization
import pytest

def test_feature_extractor():
    m = pyqg.QGModel()
    m._step_forward()

    fe1 = FeatureExtractor(m)
    fe2 = FeatureExtractor(m.to_dataset())

    np.testing.assert_allclose(fe1.extract_feature('laplacian(advected(q))'),
                               fe2.extract_feature('laplacian(advected(q))').data[0])

def test_linear_symbolic():
    m = pyqg.QGModel()
    m._step_forward()
    ds = m.to_dataset()
    fe = FeatureExtractor(ds)

    ds['q_forcing_target'] = ds.u + ds.v + 2*fe.ddx('q')

    lsr = LinearSymbolicRegression.fit(ds, ['u','v','q','ddx(q)','ddy(q)'], target='q_forcing_target')

    for m in lsr.models:
        np.testing.assert_allclose(m.coef_, np.array([1, 1, 0, 2, 0]), atol=1e-2)

    preds = lsr.test_offline(ds)

    np.testing.assert_allclose(preds.skill.mean(), 1.0, atol=1e-2)

    m2 = pyqg.QGModel(parameterization=lsr)
    m2._step_forward()

def test_neural_network():
    path = os.path.join(os.path.dirname(__file__), '../../../models/fcnn_q_to_Sqtot1')

    cnn_param = FCNNParameterization(path)

    assert len(cnn_param.models) == 2

    np.testing.assert_allclose(
            cnn_param.models[0].input_scale.sd[0,0,0,0],
            7.871194e-06,
            rtol=1e-2)

    np.testing.assert_allclose(
            cnn_param.models[0].input_scale.sd[0,1,0,0],
            1.039401e-06,
            rtol=1e-2)

    m = pyqg.QGModel()

    preds = cnn_param.predict(m)

    assert preds['q_forcing_total'].shape == (2,64,64)

    preds2 = cnn_param.predict(m.to_dataset())

    assert preds2['q_forcing_total'].shape == (1,2,64,64)
