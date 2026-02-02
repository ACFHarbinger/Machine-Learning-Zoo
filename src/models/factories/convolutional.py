from __future__ import annotations

from typing import Any

from torch import nn

from ..convolutional.capsule import CapsuleLayer
from ..convolutional.cnn import RollingWindowCNN
from ..convolutional.dcign import DCIGN
from ..convolutional.dcn import DeepConvNet
from ..convolutional.deconv import DeconvNet
from ..convolutional.resnet import DeepResNet
from ..factories.base import NeuralComponentFactory


class ConvolutionalFactory(NeuralComponentFactory):
    """Factory for convolutional networks."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get convolutional model by name."""
        name = name.lower()
        if "resnet" in name:
            return DeepResNet(**kwargs)
        elif "capsule" in name:
            return CapsuleLayer(**kwargs)
        elif "dcign" in name:
            return DCIGN(**kwargs)
        elif "deconv" in name:
            return DeconvNet(**kwargs)
        elif "dcn" in name:
            return DeepConvNet(**kwargs)
        elif "cnn" in name:
            return RollingWindowCNN(**kwargs)
        else:
            raise ValueError(
                f"Unknown convolutional model: {name}. "
                f"Available: cnn, resnet, capsule, dcign, deconv, dcn"
            )
