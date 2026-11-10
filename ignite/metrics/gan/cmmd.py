from collections.abc import Callable, Sequence
from typing import cast

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.gan.utils import _BaseInceptionMetric
from ignite.metrics.metric import reinit__is_reduced

__all__ = [
    "CMMD",
]


class CMMD(_BaseInceptionMetric):
    r"""Calculates the CLIP Maximum Mean Discrepancy (CMMD) between reference and generated images.

    .. math::
       \text{CMMD} = s \cdot \left( \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n k(\mathbf{x}_i, \mathbf{x}_j)
       + \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m k(\mathbf{y}_i, \mathbf{y}_j)
       - \frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m k(\mathbf{x}_i, \mathbf{y}_j) \right)

    where :math:`\mathbf{x}_i` and :math:`\mathbf{y}_j` are embeddings of the reference and the generated
    images, :math:`k(\mathbf{x}, \mathbf{y}) = \exp(-\| \mathbf{x} - \mathbf{y} \|^2 / 2\sigma^2)` is a
    Gaussian RBF kernel and :math:`s` is a scale factor.

    More details can be found in `Jayasumana et al. 2024`__, where CMMD is proposed as a replacement
    for :class:`~ignite.metrics.gan.FID` for the evaluation of generative models.

    __ https://arxiv.org/abs/2401.09603

    The kernel means include the diagonal terms, i.e. the biased V-statistic estimator is used. This
    matches the reference implementations and makes the output comparable to the values reported in
    the paper. Unlike FID, the maximum mean discrepancy has no fixed-size sufficient statistics:
    all embeddings are accumulated and the kernel means are computed over the whole reference and
    generated sets in ``compute``. The two sets may contain a different number of embeddings.

    Remark:

        This implementation is a clean-room reimplementation from the paper above. We consulted the
        following Apache-2.0 licensed reference implementations, which are not copied into this
        BSD-3 licensed codebase: `sayakpaul/cmmd-pytorch`__ and `google-research cmmd`__.

        __ https://github.com/sayakpaul/cmmd-pytorch
        __ https://github.com/google-research/google-research/tree/master/cmmd

    .. note::

        This metric is different from :class:`~ignite.metrics.MaximumMeanDiscrepancy`: the latter
        computes an unbiased U-statistic (diagonal excluded) on every batch, requires ``x`` and ``y``
        of the same shape, averages the per-batch estimates and returns the square root of MMD.
        CMMD computes the biased V-statistic over the whole accumulated sets of embeddings and
        multiplies it by ``scale``.

    .. note::

        CMMD is defined on CLIP image embeddings. The metric does not provide a default feature
        extractor and does not add a dependency on ``transformers``: pass a ``feature_extractor``
        returning CLIP embeddings, or feed precomputed embeddings while ``feature_extractor``
        is assigned to Identity function (the default when only ``num_features`` is defined).
        To reproduce the paper's values, the following wrapper can be used:

        .. code::

            import torch
            import torch.nn as nn
            from torchvision import transforms
            from transformers import CLIPVisionModelWithProjection

            # wrapper class as feature_extractor
            class ClipImageEmbedder(nn.Module):

                def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
                    super().__init__()
                    self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).eval()

                @torch.no_grad()
                def forward(self, x):
                    image_embeds = self.model(x).image_embeds
                    # CMMD uses L2-normalized CLIP embeddings
                    return nn.functional.normalize(image_embeds, dim=-1)

            # bicubic resize to 336x336, as in the paper
            preprocess = transforms.Compose([
                transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(336),
                transforms.ToTensor(),
            ])

            clip_embedder = ClipImageEmbedder()

            # comparable metric
            cmmd_metric = CMMD(num_features=768, feature_extractor=clip_embedder)

    Args:
        num_features: number of features of the embeddings, e.g. 768 for CLIP ViT-L/14 image
            embeddings. Must be provided: unlike FID, there is no default Inception model.
        feature_extractor: a torch Module for extracting the features from the input data.
            It returns a tensor of shape (batch_size, num_features). If only ``num_features`` is
            defined but ``feature_extractor`` is not defined, ``feature_extractor`` is assigned
            Identity function. Please note that the model will be implicitly converted to device
            mentioned in the ``device`` argument.
        sigma: the bandwidth :math:`\sigma` of the Gaussian RBF kernel. Default value is 10.0, the
            paper's value for L2-normalized CLIP embeddings.
        scale: factor the MMD value is multiplied by to report numbers of a convenient magnitude.
            Default value is 1000.0, the paper's value.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(real, fake)``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = CMMD(num_features=1, feature_extractor=default_model)
            metric.attach(default_evaluator, "cmmd")
            y_real = torch.ones(10, 4)
            y_fake = torch.ones(10, 4)
            state = default_evaluator.run([[y_real, y_fake]])
            print(state.metrics["cmmd"])

        .. testoutput::

            0.0

    .. versionadded:: 0.6.1
    """

    _state_dict_all_req_keys = ("_real_features", "_fake_features")

    def __init__(
        self,
        num_features: int | None = None,
        feature_extractor: torch.nn.Module | None = None,
        sigma: float = 10.0,
        scale: float = 1000.0,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
    ) -> None:
        self._sigma = sigma
        self._scale = scale

        super().__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._real_features: torch.Tensor = torch.zeros(
            (0, self._num_features), dtype=self._double_dtype, device=self._device
        )
        self._fake_features: torch.Tensor = torch.zeros(
            (0, self._num_features), dtype=self._double_dtype, device=self._device
        )

        super().reset()  # type: ignore

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        real, fake = output
        self._real_features = torch.cat([self._real_features, self._extract_features(real)], dim=0)
        self._fake_features = torch.cat([self._fake_features, self._extract_features(fake)], dim=0)

    def compute(self) -> float:
        if self._real_features.shape[0] < 1 or self._fake_features.shape[0] < 1:
            raise NotComputableError("CMMD must have at least one example before it can be computed.")

        real, fake = self._real_features, self._fake_features

        if idist.get_world_size() > 1:
            real = self._all_gather_features(real)
            fake = self._all_gather_features(fake)

        xx, yy, xy = torch.mm(real, real.t()), torch.mm(fake, fake.t()), torch.mm(real, fake.t())
        rx = xx.diag().unsqueeze(0)
        ry = yy.diag().unsqueeze(0)

        gamma = 1.0 / (2.0 * self._sigma**2)
        # biased V-statistic: kernel means include the diagonal terms
        k_xx = torch.exp(-gamma * (rx.t() + rx - 2.0 * xx)).mean()
        k_yy = torch.exp(-gamma * (ry.t() + ry - 2.0 * yy)).mean()
        k_xy = torch.exp(-gamma * (rx.t() + ry - 2.0 * xy)).mean()

        return (self._scale * (k_xx + k_yy - 2.0 * k_xy)).item()

    @staticmethod
    def _all_gather_features(features: torch.Tensor) -> torch.Tensor:
        # MMD has no sufficient statistics: gather the full embeddings, padding for uneven ranks
        counts = cast(torch.Tensor, idist.all_gather(torch.tensor([features.shape[0]], device=features.device)))
        shapes = [[count, features.shape[1]] for count in counts.tolist()]
        return torch.cat(idist.all_gather_tensors_with_shapes(features, shapes))
