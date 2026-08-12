"""Co-DINO head with a fixed-simplex ETF loss on matched positive queries."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from codetr.co_dino_head import CoDINOHead
from mmdet.registry import MODELS


def build_simplex_etf(feat_dim: int, num_classes: int) -> Tensor:
    """Return ``num_classes`` unit vectors with pairwise cosine ``-1/(C-1)``."""
    if feat_dim < num_classes - 1:
        raise ValueError(
            f"feat_dim={feat_dim} cannot embed a {num_classes}-class simplex"
        )

    generator = torch.Generator().manual_seed(0)
    basis, _ = torch.linalg.qr(
        torch.randn(feat_dim, num_classes, generator=generator)
    )
    centering = torch.eye(num_classes) - torch.ones(
        num_classes, num_classes
    ) / num_classes
    return math.sqrt(num_classes / (num_classes - 1)) * basis @ centering


@MODELS.register_module()
class CoDINOFixedETFHead(CoDINOHead):
    """Add class-weighted dot regression to Co-DINO's standard detector loss."""

    def __init__(
        self,
        etf_loss_weight: float = 0.1,
        etf_feat_dim: int | None = None,
        etf_scale: float = 1.0,
        etf_class_weight: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.etf_loss_weight = float(etf_loss_weight)
        self.etf_scale = float(etf_scale)

        feat_dim = int(etf_feat_dim or self.embed_dims)
        self.etf_proj = nn.Linear(self.embed_dims, feat_dim, bias=False)
        nn.init.normal_(self.etf_proj.weight, std=0.02)
        self.register_buffer(
            "etf_targets",
            build_simplex_etf(feat_dim, self.num_classes),
            persistent=False,
        )

        if etf_class_weight is None:
            self.etf_class_weight = None
        else:
            if len(etf_class_weight) != self.num_classes:
                raise ValueError("etf_class_weight must contain one value per class")
            self.register_buffer(
                "etf_class_weight",
                torch.tensor(etf_class_weight, dtype=torch.float32),
                persistent=False,
            )

        self._captured_hidden_states: Tensor | None = None
        self.transformer.register_forward_hook(self._capture_hidden_states)

    def _capture_hidden_states(self, _module, _inputs, output) -> None:
        if self.training:
            self._captured_hidden_states = output[0]

    def loss(self, x, batch_data_samples):
        """Compute the published Co-DINO losses and the ETF auxiliary loss."""
        assert self.dn_generator is not None, '"dn_cfg" must be set'

        batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = self.dn_generator(
            batch_data_samples
        )
        outs = self(x, batch_img_metas, dn_label_query, dn_bbox_query, attn_mask)
        loss_inputs = outs[:-1] + (batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        losses["loss_etf"] = self.etf_loss_weight * self._loss_fixed_etf(
            outs[0], outs[1], batch_gt_instances, batch_img_metas, dn_meta
        )
        return losses, outs[-1]

    def _matching_hidden_states(self, dn_meta: dict[str, int] | None) -> Tensor:
        hidden = self._captured_hidden_states
        if hidden is None:
            raise RuntimeError("Co-DINO transformer hidden states were not captured")
        hidden = hidden.permute(0, 2, 1, 3)
        num_denoising = int((dn_meta or {}).get("num_denoising_queries", 0))
        return hidden[-1, :, num_denoising:, :]

    def _loss_fixed_etf(
        self,
        outputs_classes: Tensor,
        outputs_coords: Tensor,
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        dn_meta: dict[str, int] | None,
    ) -> Tensor:
        hidden = self._matching_hidden_states(dn_meta)
        matching_cls, matching_bbox, _, _ = self.split_outputs(
            outputs_classes, outputs_coords, dn_meta
        )
        last_cls, last_bbox = matching_cls[-1], matching_bbox[-1]

        positive_features: list[Tensor] = []
        positive_labels: list[Tensor] = []
        for image_index, (gt_instances, img_meta) in enumerate(
            zip(batch_gt_instances, batch_img_metas)
        ):
            labels, _, _, _, pos_inds, _ = self._get_targets_single(
                last_cls[image_index],
                last_bbox[image_index],
                gt_instances,
                img_meta,
            )
            if pos_inds.numel():
                positive_features.append(hidden[image_index, pos_inds])
                positive_labels.append(labels[pos_inds].long())

        if not positive_features:
            return self.etf_proj.weight.sum() * 0.0

        features = F.normalize(self.etf_proj(torch.cat(positive_features)), dim=1)
        labels = torch.cat(positive_labels)
        targets = self.etf_targets.to(features.dtype).t()[labels]
        per_sample = 0.5 * (self.etf_scale * (features * targets).sum(1) - 1).pow(2)

        if self.etf_class_weight is None:
            return per_sample.mean()
        weights = self.etf_class_weight.to(per_sample)[labels]
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
