"""Model architectures for biometric recognition."""

from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import MultimodalFusion
from biometric.models.iris_encoder import IrisEncoder
from biometric.models.multimodal_net import MultimodalBiometricNet

__all__ = [
    "FingerprintEncoder",
    "IrisEncoder",
    "MultimodalFusion",
    "MultimodalBiometricNet",
]
