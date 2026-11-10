import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.gan.cmmd import CMMD


def reference_cmmd(x, y, sigma=10.0, scale=1000.0):
    """Biased V-statistic CMMD computed with numpy, as in Jayasumana et al. 2024 (arXiv:2401.09603)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma**2)

    def rbf(a, b):
        dist = -2.0 * a @ b.T + (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :]
        return np.exp(-gamma * dist)

    k_xx = rbf(x, x).mean()
    k_yy = rbf(y, y).mean()
    k_xy = rbf(x, y).mean()
    return scale * (k_xx + k_yy - 2.0 * k_xy)


def test_compute_cmmd_parity_with_numpy(available_device):
    torch.manual_seed(2)
    n, m, num_features = 13, 9, 7
    real_samples = torch.rand(n, num_features)
    fake_samples = torch.rand(m, num_features)

    metric = CMMD(num_features=num_features, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.update([real_samples[:5], fake_samples[:4]])
    metric.update([real_samples[5:], fake_samples[4:]])

    tol = 5e-4 if available_device == "mps" else 1e-5
    assert pytest.approx(reference_cmmd(real_samples.numpy(), fake_samples.numpy()), rel=tol) == metric.compute()


def test_cmmd_zero_for_identical_sets(available_device):
    torch.manual_seed(3)
    samples = torch.rand(10, 8)

    metric = CMMD(num_features=8, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.update([samples, samples.clone()])
    assert metric.compute() == 0.0


def test_cmmd_monotonic_with_noise_level(available_device):
    torch.manual_seed(4)
    real_samples = torch.randn(20, 16)
    real_samples = real_samples / real_samples.norm(dim=1, keepdim=True)

    metric = CMMD(num_features=16, feature_extractor=torch.nn.Identity(), device=available_device)
    values = []
    for noise in (0.0, 0.05, 0.1, 0.2, 0.4):
        fake_samples = real_samples + noise * torch.randn_like(real_samples)
        fake_samples = fake_samples / fake_samples.norm(dim=1, keepdim=True)

        metric.reset()
        metric.update([real_samples, fake_samples])
        values.append(metric.compute())

    assert all(value_b > value_a for value_a, value_b in zip(values[:-1], values[1:]))


@pytest.mark.parametrize("n_batches", [1, 2, 5])
def test_cmmd_accumulation_invariance(available_device, n_batches):
    torch.manual_seed(5)
    n, m, num_features = 10, 7, 12
    real_samples = torch.rand(n, num_features)
    fake_samples = torch.rand(m, num_features)

    metric = CMMD(num_features=num_features, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.update([real_samples, fake_samples])
    single_batch = metric.compute()

    # several batches, also of uneven sizes between real and fake
    metric.reset()
    for real_batch, fake_batch in zip(real_samples.tensor_split(n_batches), fake_samples.tensor_split(n_batches)):
        metric.update([real_batch, fake_batch])

    assert pytest.approx(single_batch, rel=1e-5, abs=1e-12) == metric.compute()


def test_cmmd_unequal_set_sizes(available_device):
    torch.manual_seed(6)
    real_samples = torch.rand(11, 6)
    fake_samples = torch.rand(4, 6)

    metric = CMMD(num_features=6, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.update([real_samples, fake_samples])

    assert pytest.approx(reference_cmmd(real_samples.numpy(), fake_samples.numpy()), rel=1e-5) == metric.compute()


def test_cmmd_custom_sigma_and_scale(available_device):
    torch.manual_seed(7)
    real_samples = torch.rand(6, 5)
    fake_samples = torch.rand(6, 5)

    metric = CMMD(
        num_features=5,
        feature_extractor=torch.nn.Identity(),
        sigma=1.0,
        scale=1.0,
        device=available_device,
    )
    metric.update([real_samples, fake_samples])

    expected = reference_cmmd(real_samples.numpy(), fake_samples.numpy(), sigma=1.0, scale=1.0)
    assert pytest.approx(expected, rel=1e-5) == metric.compute()


def test_not_computable():
    metric = CMMD(num_features=8, feature_extractor=torch.nn.Identity())
    with pytest.raises(NotComputableError, match=r"CMMD must have at least one example before it can be computed."):
        metric.compute()


def test_state_dict_roundtrip(available_device):
    torch.manual_seed(8)
    real_samples = torch.rand(6, 4)
    fake_samples = torch.rand(5, 4)

    metric = CMMD(num_features=4, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.update([real_samples, fake_samples])

    restored_metric = CMMD(num_features=4, feature_extractor=torch.nn.Identity(), device=available_device)
    restored_metric.load_state_dict(metric.state_dict())

    assert pytest.approx(metric.compute(), rel=1e-5) == restored_metric.compute()


def test_integrated_with_engine(available_device):
    from ignite.engine import Engine
    from ignite.metrics import CMMD as CMMDPublic

    torch.manual_seed(9)
    real_samples = torch.rand(12, 6)
    fake_samples = torch.rand(9, 6)

    def update(_, i):
        return (
            real_samples[i * 4 : (i + 1) * 4, :],
            fake_samples[i * 3 : (i + 1) * 3, :],
        )

    engine = Engine(update)
    metric = CMMDPublic(num_features=6, feature_extractor=torch.nn.Identity(), device=available_device)
    metric.attach(engine, "cmmd")

    engine.run(data=list(range(3)), max_epochs=1)

    assert "cmmd" in engine.state.metrics
    assert (
        pytest.approx(reference_cmmd(real_samples.numpy(), fake_samples.numpy()), rel=1e-5)
        == engine.state.metrics["cmmd"]
    )


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Argument num_features must be provided"):
        CMMD(feature_extractor=torch.nn.Identity())

    with pytest.raises(ValueError, match=r"Argument num_features must be greater to zero"):
        CMMD(num_features=-1, feature_extractor=torch.nn.Identity())

    with pytest.raises(TypeError, match=r"Argument feature_extractor must be of type torch.nn.Module"):
        CMMD(num_features=1, feature_extractor=lambda x: x)

    with pytest.raises(ValueError, match=r"feature_extractor output must be a tensor of dim 2, got: 1"):
        CMMD(num_features=1, feature_extractor=torch.nn.Identity()).update(torch.tensor([[], []]))

    with pytest.raises(ValueError, match=r"Batch size should be greater than one, got: 0"):
        CMMD(num_features=1, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0, 0))

    with pytest.raises(ValueError, match=r"num_features returned by feature_extractor should be 3, got: 5"):
        CMMD(num_features=3, feature_extractor=torch.nn.Identity()).update((torch.rand(2, 3), torch.rand(2, 5)))


def _test_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        n_iters = 6
        s = 4
        offset = n_iters * s

        num_features = 10

        y_pred = torch.rand(offset * idist.get_world_size(), num_features)
        y_true = torch.rand(offset * idist.get_world_size(), num_features)

        def update(_, i):
            return (
                y_pred[i * s + rank * offset : (i + 1) * s + rank * offset, :],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset, :],
            )

        engine = Engine(update)
        m = CMMD(num_features=num_features, feature_extractor=torch.nn.Identity(), device=metric_device)
        m.attach(engine, "cmmd")

        engine.run(data=list(range(n_iters)), max_epochs=1)

        assert "cmmd" in engine.state.metrics
        assert pytest.approx(reference_cmmd(y_pred.numpy(), y_true.numpy()), rel=1e-5) == m.compute()

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        _test(metric_device=metric_device)


def _test_distrib_uneven_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        s = 4
        num_features = 10

        # rank r sees (world_size - r) batches: gathered embeddings have uneven per-rank counts
        n_iters = idist.get_world_size() - rank
        total = s * sum(range(idist.get_world_size() + 1))

        y_pred = torch.rand(total, num_features)
        y_true = torch.rand(total, num_features)

        start = s * sum(idist.get_world_size() - r for r in range(rank))

        def update(_, i):
            return (
                y_pred[start + i * s : start + (i + 1) * s, :],
                y_true[start + i * s : start + (i + 1) * s, :],
            )

        engine = Engine(update)
        m = CMMD(num_features=num_features, feature_extractor=torch.nn.Identity(), device=metric_device)
        m.attach(engine, "cmmd")

        engine.run(data=list(range(n_iters)), max_epochs=1)

        assert "cmmd" in engine.state.metrics
        assert pytest.approx(reference_cmmd(y_pred.numpy(), y_true.numpy()), rel=1e-5) == m.compute()

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        _test(metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_integration(device)
    _test_distrib_uneven_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)
    _test_distrib_uneven_integration(device)
