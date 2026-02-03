from __future__ import annotations

from typing import Any

from torch import nn

from ..recurrent.esn import EchoStateNetwork as ESN
from ..recurrent.lsm import LiquidStateMachine
from ..recurrent.rnn import GRU, LSTM
from ..recurrent.tsmamba import TSMamba
from ..recurrent.xlstm import xLSTM
from .base import NeuralComponentFactory


class RecurrentFactory(NeuralComponentFactory):
    """Factory for recurrent neural networks."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get recurrent model by name."""
        name = name.lower()
        if "lstm" in name and "xlstm" not in name:
            return LSTM(**kwargs)
        elif "gru" in name:
            return GRU(**kwargs)
        elif "xlstm" in name:
            return xLSTM(**kwargs)
        elif "mamba" in name:
            return TSMamba(**kwargs)
        elif "esn" in name:
            return ESN(**kwargs)
        elif "lsm" in name:
            return LiquidStateMachine(**kwargs)
        else:
            raise ValueError(
                f"Unknown recurrent model: {name}. Available: lstm, gru, xlstm, mamba, esn, lsm"
            )
