"""Inference pipeline for biometric prediction from checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

from biometric.data.transforms import get_fingerprint_transforms, get_iris_transforms
from biometric.models.multimodal_net import MultimodalBiometricNet

logger = logging.getLogger(__name__)


class BiometricPredictor:
    """Production inference pipeline for biometric identity recognition.

    Load once from a checkpoint, then call predict() for individual samples
    or predict_batch() for efficient batched inference.
    """

    def __init__(
        self,
        model: MultimodalBiometricNet,
        cfg: DictConfig,
        inv_label_map: dict[int, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.inv_label_map = inv_label_map
        self.device = device
        self.fp_transform = get_fingerprint_transforms(cfg.data, is_train=False)
        self.iris_transform = get_iris_transforms(cfg.data, is_train=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device | None = None,
    ) -> "BiometricPredictor":
        """Load a predictor from a training checkpoint.

        Args:
            checkpoint_path: Path to a .pt checkpoint file.
            device: Device to load the model onto. Auto-detected if None.

        Returns:
            Ready-to-use BiometricPredictor instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        cfg = OmegaConf.create(checkpoint["config"])
        model = MultimodalBiometricNet(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        label_map: dict[int, int] = checkpoint["label_map"]
        inv_label_map = {v: k for k, v in label_map.items()}

        logger.info(
            "Loaded predictor from %s (epoch %d, %d classes)",
            checkpoint_path.name,
            checkpoint["epoch"],
            len(label_map),
        )

        return cls(model, cfg, inv_label_map, device)

    def predict(
        self,
        fingerprint: Union[str, Path, Image.Image],
        iris: Union[str, Path, Image.Image],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Predict identity from a fingerprint + iris pair.

        Args:
            fingerprint: Path or PIL Image of a fingerprint.
            iris: Path or PIL Image of an iris.
            top_k: Number of top predictions to return.

        Returns:
            Dict with predicted_subject_id, confidence, and top_k list.
        """
        fp_tensor = self._load_and_transform_fp(fingerprint)
        iris_tensor = self._load_and_transform_iris(iris)

        with torch.no_grad():
            logits = self.model(fp_tensor, iris_tensor)
            probs = torch.softmax(logits, dim=1)

        k = min(top_k, probs.shape[1])
        top_probs, top_indices = probs.topk(k, dim=1)

        top_results = [
            {
                "subject_id": self.inv_label_map[idx.item()],
                "confidence": round(p.item(), 4),
            }
            for idx, p in zip(top_indices[0], top_probs[0])
        ]

        return {
            "predicted_subject_id": top_results[0]["subject_id"],
            "confidence": top_results[0]["confidence"],
            "top_k": top_results,
        }

    def predict_batch(
        self,
        samples: list[dict[str, Union[str, Path, Image.Image]]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Process multiple samples with batched inference.

        Args:
            samples: List of dicts with 'fingerprint' and 'iris' keys.
            top_k: Number of top predictions per sample.

        Returns:
            List of prediction dicts.
        """
        if not samples:
            return []

        fp_tensors = []
        iris_tensors = []
        for sample in samples:
            fp_tensors.append(self._load_and_transform_fp(sample["fingerprint"]))
            iris_tensors.append(self._load_and_transform_iris(sample["iris"]))

        fp_batch = torch.cat(fp_tensors, dim=0)
        iris_batch = torch.cat(iris_tensors, dim=0)

        with torch.no_grad():
            logits = self.model(fp_batch, iris_batch)
            probs = torch.softmax(logits, dim=1)

        results = []
        k = min(top_k, probs.shape[1])
        top_probs, top_indices = probs.topk(k, dim=1)

        for i in range(len(samples)):
            top_results = [
                {
                    "subject_id": self.inv_label_map[idx.item()],
                    "confidence": round(p.item(), 4),
                }
                for idx, p in zip(top_indices[i], top_probs[i])
            ]
            results.append({
                "predicted_subject_id": top_results[0]["subject_id"],
                "confidence": top_results[0]["confidence"],
                "top_k": top_results,
            })

        return results

    def _load_and_transform_fp(
        self, source: Union[str, Path, Image.Image]
    ) -> Tensor:
        if isinstance(source, Image.Image):
            img = source
        else:
            img = Image.open(source)
        tensor = self.fp_transform(img)
        return tensor.unsqueeze(0).to(self.device)

    def _load_and_transform_iris(
        self, source: Union[str, Path, Image.Image]
    ) -> Tensor:
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        tensor = self.iris_transform(img)
        return tensor.unsqueeze(0).to(self.device)
